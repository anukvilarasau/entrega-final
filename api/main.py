"""
api/main.py — API de producción del asesor de viajes multi-agente (FastAPI, 100% async).

Endpoints (todos con esquemas Pydantic de entrada y salida, ver /docs):
    POST /tasks                    encola una pregunta (o un seguimiento con thread_id) → 202 al instante
    GET  /tasks/{job_id}           estado, progreso, pausa HITL, resultado o error
    POST /tasks/{job_id}/approve   responde la pausa HITL: {"aprobado": true|false}
    GET  /tasks                    los últimos jobs
    GET  /threads/{thread_id}      la conversación: sus turnos, leídos del checkpoint de LangGraph
    GET  /rag/buscar               el retriever de guías solo (diagnóstico: fragmentos y puntajes)
    GET  /health                   Redis, checkpointer, RAG, modo, LangSmith y jobs activos

Ningún endpoint espera al grafo: POST /tasks guarda el job en Redis (redis.asyncio)
y le pasa el trabajo al worker (una tarea de asyncio). Lo sync y lento al arrancar
(cargar ChromaDB y los modelos ONNX) va a un hilo con run_in_threadpool; la ingesta
del RAG (si el índice está vacío o cambió el corpus) es async.

Levantar:
    docker compose up                                   # app + Redis 8 (todo con un comando)
    uvicorn api.main:app --port 8000                    # usa el .env
    MODO_MOCK=1 uvicorn api.main:app --port 8000        # sin claves (LLM con políticas + Tavily falso + fakeredis)
"""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from langgraph.types import Command

import config
from api.schemas import (
    AprobacionAceptada,
    AprobacionRequest,
    ConversacionInfo,
    Error,
    EstadoJob,
    EstadoRedis,
    HealthInfo,
    JobInfo,
    TaskCreada,
    TaskRequest,
    TurnoInfo,
)
from api.worker import JobStore, Worker
from graph.builder import Componentes, construir_grafo_api, crear_componentes
from graph.checkpointer import abrir_checkpointer
from graph.state import ahora
from observability import monitoreo as observability
from rag.retriever import ResultadoBusqueda

log = logging.getLogger(__name__)

ERRORES_JOB: dict[int | str, dict[str, Any]] = {404: {"model": Error, "description": "No existe el job."}}


# ---------------------------------------------------------------------------
# Configuración de la API (sale de config.AJUSTES: variables de entorno validadas)
# ---------------------------------------------------------------------------
@dataclass
class Ajustes:
    modo_mock: bool = field(default_factory=lambda: config.AJUSTES.modo_mock)
    redis_url: str = field(default_factory=lambda: config.AJUSTES.redis_url.strip())
    max_concurrentes: int = field(default_factory=lambda: config.AJUSTES.max_jobs_concurrentes)
    timeout_s: float = field(default_factory=lambda: config.AJUSTES.job_timeout_s)
    # Tope de conexiones a Redis (Redis Cloud gratis admite 30 clientes): 1/3 para los jobs, el resto checkpoints.
    redis_max_conexiones: int = field(default_factory=lambda: config.AJUSTES.redis_max_conexiones)
    ingesta_al_arrancar: bool = field(default_factory=lambda: config.AJUSTES.ingesta_al_arrancar)
    guardar_trazas: bool | None = None  # None = solo en modo real (traces/api/)

    @classmethod
    def desde_entorno(cls) -> "Ajustes":
        return cls()


def _job_info(job: dict[str, Any]) -> JobInfo:
    return JobInfo(job_id=job["id"], **{k: v for k, v in job.items() if k != "id" and v is not None})


# ---------------------------------------------------------------------------
# Arranque y apagado
# ---------------------------------------------------------------------------
async def _conectar_redis(ajustes: Ajustes) -> tuple[Any, str]:
    if ajustes.redis_url:
        from redis.asyncio import BlockingConnectionPool, Redis

        pool = BlockingConnectionPool.from_url(ajustes.redis_url, decode_responses=True, timeout=60,
                                               max_connections=max(4, ajustes.redis_max_conexiones // 3))
        cliente = Redis(connection_pool=pool, auto_close_connection_pool=True)
        await cliente.ping()
        return cliente, "redis"
    import fakeredis

    log.warning("Sin REDIS_URL: el estado de los jobs vive en fakeredis (en memoria, se pierde al reiniciar).")
    return fakeredis.FakeAsyncRedis(decode_responses=True), "fakeredis"


def create_app(ajustes: Ajustes | None = None, componentes: Componentes | None = None) -> FastAPI:
    """Fábrica de la app (los tests le pasan ajustes y componentes propios)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        a = ajustes or Ajustes.desde_entorno()
        config.configurar_logs(logging.INFO)
        config.validar_entorno(a.modo_mock)
        app.state.langsmith = observability.configurar_langsmith(a.modo_mock)
        async with AsyncExitStack() as pila:
            redis, tipo_redis = await _conectar_redis(a)
            pila.push_async_callback(redis.aclose)
            checkpointer = await pila.enter_async_context(abrir_checkpointer(
                a.redis_url or None, max_conexiones=max(4, a.redis_max_conexiones - a.redis_max_conexiones // 3)))
            # Chroma y los modelos ONNX se cargan con código sync: fuera del event loop.
            comp = componentes or await run_in_threadpool(crear_componentes, modo_mock=a.modo_mock)
            if comp.base is not None and a.ingesta_al_arrancar:
                reporte = await comp.base.asegurar_indice()  # idempotente: si no cambió nada, no reindexa
                log.info("RAG listo: %d fragmentos (%s)", reporte.fragmentos_totales,
                         "sin cambios" if not reporte.hubo_cambios else f"indexadas: {reporte.indexados}")
            grafo = construir_grafo_api(comp, checkpointer)
            store = JobStore(redis)
            worker = Worker(grafo, store, modo_mock=a.modo_mock, max_concurrentes=a.max_concurrentes,
                            timeout_s=a.timeout_s,
                            guardar_trazas=(not a.modo_mock) if a.guardar_trazas is None else a.guardar_trazas)
            app.state.ajustes, app.state.store, app.state.worker, app.state.grafo = a, store, worker, grafo
            app.state.redis, app.state.tipo_redis, app.state.componentes = redis, tipo_redis, comp
            app.state.checkpointer = type(checkpointer).__name__
            recuperados = await worker.recuperar(checkpoint_persistente=bool(a.redis_url))
            if recuperados:
                log.warning("Jobs recuperados al arrancar: %s", recuperados)
            log.info("API lista · modo %s · %s · %s", "mock" if a.modo_mock else "real", tipo_redis,
                     app.state.checkpointer)
            try:
                yield
            finally:
                await worker.cerrar()
                if comp.base is not None and componentes is None:
                    await comp.base.cerrar()
                await observability.esperar_envio()

    app = FastAPI(
        title="Asesor de viajes multi-agente — API (Entrega final)",
        version="8.0.0",
        description=(
            "Sistema de inteligencia de viajes por Argentina: un **supervisor** (LangGraph) coordina a un "
            "**investigador** (RAG sobre guías de Wikivoyage → memoria de búsquedas → web con aprobación humana) y "
            "a un **analista** (validación y presupuesto). Jobs asíncronos con estado en **Redis**, conversaciones "
            "de varios turnos con checkpoints de LangGraph en Redis (sobreviven a un reinicio), y trazas en "
            "**LangSmith**.\n\nFlujo típico: `POST /tasks` → polling de `GET /tasks/{job_id}` → si queda en "
            "`WAITING_APPROVAL`, `POST /tasks/{job_id}/approve` → `DONE`. Para seguir la conversación, otro "
            "`POST /tasks` con el `thread_id`."
        ),
        lifespan=lifespan,
        openapi_tags=[{"name": "tasks", "description": "Jobs del asesor (encolar, consultar, aprobar)."},
                      {"name": "conversaciones", "description": "Conversaciones de varios turnos (threads)."},
                      {"name": "rag", "description": "La base de conocimiento de guías."},
                      {"name": "operación", "description": "Salud del servicio."}],
    )

    def _store(request: Request) -> JobStore:
        return request.app.state.store

    @app.get("/", include_in_schema=False)
    async def raiz() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskCreada, tags=["tasks"],
              summary="Encolar una pregunta o un seguimiento de una conversación",
              responses={404: {"model": Error, "description": "El thread_id no existe (o se perdió su checkpoint)."},
                         409: {"model": Error, "description": "La conversación tiene un job sin terminar."}})
    async def crear_task(cuerpo: TaskRequest, request: Request) -> TaskCreada:
        """Guarda el job (PENDING) y lo encola. Vuelve enseguida: el grafo corre en segundo plano.

        Con `thread_id`, el job es el turno siguiente de esa conversación: el grafo retoma el estado guardado
        en el checkpointer (pedido, aportes y respuestas anteriores).
        """
        a: Ajustes = request.app.state.ajustes
        store = _store(request)
        worker: Worker = request.app.state.worker
        turno = 1
        if cuerpo.thread_id:
            if (activo := await store.job_activo(cuerpo.thread_id)) is not None:
                raise HTTPException(status.HTTP_409_CONFLICT,
                                    f"La conversación {cuerpo.thread_id} tiene el job {activo} sin terminar: "
                                    "esperá a que termine (o aprobá su pausa) antes de seguir.")
            snap = await worker.estado_thread(cuerpo.thread_id)
            if not snap.values:
                raise HTTPException(status.HTTP_404_NOT_FOUND,
                                    f"No existe la conversación {cuerpo.thread_id} en el checkpointer "
                                    f"({request.app.state.checkpointer}).")
            turno = int(snap.values.get("turno") or 1) + 1
        job_id = await store.crear(cuerpo.pregunta.strip(), cuerpo.auto_aprobar, "mock" if a.modo_mock else "real",
                                   thread_id=cuerpo.thread_id, turno=turno)
        thread_id = cuerpo.thread_id or job_id
        await worker.iniciar(job_id, cuerpo.pregunta, cuerpo.auto_aprobar, turno=turno)
        return TaskCreada(job_id=job_id, thread_id=thread_id, turno=turno, estado=EstadoJob.PENDING,
                          links={"estado": f"/tasks/{job_id}", "aprobar": f"/tasks/{job_id}/approve",
                                 "conversacion": f"/threads/{thread_id}"})

    @app.get("/tasks/{job_id}", response_model=JobInfo, tags=["tasks"], responses=ERRORES_JOB,
             summary="Estado y resultado de un job")
    async def ver_task(job_id: str, request: Request) -> JobInfo:
        job = await _store(request).obtener(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
        return _job_info(job)

    @app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED, response_model=AprobacionAceptada,
              tags=["tasks"], summary="Responder la pausa human-in-the-loop (búsqueda web)",
              responses={**ERRORES_JOB, 409: {"model": Error, "description": "El job no está esperando aprobación."}})
    async def aprobar_task(job_id: str, cuerpo: AprobacionRequest, request: Request) -> AprobacionAceptada:
        """Responde la pausa HITL y reanuda el grafo desde su checkpoint con Command(resume=...)."""
        store = _store(request)
        job = await store.obtener(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
        pendiente = job.get("aprobacion_pendiente")
        aprobaciones = [*(job.get("aprobaciones") or []),
                        {"ronda": (pendiente or {}).get("ronda"), "aprobado": cuerpo.aprobado,
                         "comentario": cuerpo.comentario, "temas": (pendiente or {}).get("temas_sin_memoria", []),
                         "timestamp": ahora()}]
        cambio = await store.transicion(job_id, {EstadoJob.WAITING_APPROVAL}, EstadoJob.RUNNING,
                                        aprobacion_pendiente=None, aprobaciones=aprobaciones)
        if not cambio:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                f"El job {job_id} no está esperando aprobación (estado: {job['estado']}).")
        request.app.state.worker.lanzar(job_id, Command(resume=cuerpo.model_dump()))
        return AprobacionAceptada(job_id=job_id, estado=EstadoJob.RUNNING, aprobado=cuerpo.aprobado)

    @app.get("/tasks", response_model=list[JobInfo], tags=["tasks"], summary="Los últimos jobs")
    async def listar_tasks(request: Request, limite: int = Query(20, ge=1, le=200)) -> list[JobInfo]:
        return [_job_info(j) for j in await _store(request).listar(limite)]

    @app.get("/threads/{thread_id}", response_model=ConversacionInfo, tags=["conversaciones"],
             summary="Historial de una conversación (desde el checkpoint de LangGraph)",
             responses={404: {"model": Error, "description": "No existe la conversación."}})
    async def ver_thread(thread_id: str, request: Request) -> ConversacionInfo:
        snap = await request.app.state.worker.estado_thread(thread_id)
        jobs = await _store(request).jobs_del_thread(thread_id)
        if not snap.values and not jobs:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {thread_id}.")
        turnos = [TurnoInfo(turno=t.turno, pregunta=t.pregunta, respuesta=t.respuesta,
                            pedido=t.pedido.model_dump(mode="json", exclude_none=True) if t.pedido else None,
                            timestamp=t.timestamp) for t in (snap.values or {}).get("turnos", [])]
        return ConversacionInfo(thread_id=thread_id, turnos_completos=len(turnos), turnos=turnos, jobs=jobs,
                                job_activo=await _store(request).job_activo(thread_id),
                                checkpointer=request.app.state.checkpointer)

    @app.get("/rag/buscar", response_model=ResultadoBusqueda, tags=["rag"],
             summary="Consultar la base de conocimiento de guías (sin LLM)",
             responses={503: {"model": Error, "description": "La API corre sin base de conocimiento."}})
    async def buscar_guias(request: Request,
                           q: str = Query(min_length=3, max_length=300, examples=["dónde esquiar"]),
                           destino: str | None = Query(default=None, max_length=80, examples=["Bariloche"]),
                           k: int = Query(default=config.RAG_TOP_K, ge=1, le=20)) -> ResultadoBusqueda:
        """El mismo retriever que usa la herramienta consultar_guias: híbrido (BM25 + vectorial) + RRF +
        re-ranking, con filtro por destino desde los metadatos. Devuelve cada fragmento con sus puntajes."""
        base = request.app.state.componentes.base
        if base is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La API corre sin base de conocimiento.")
        return await base.consultar(q, destino, k)

    @app.get("/health", response_model=HealthInfo, tags=["operación"], summary="Salud del servicio")
    async def health(request: Request) -> HealthInfo:
        s = request.app.state
        try:
            redis_ok = bool(await s.redis.ping())
        except Exception as e:  # noqa: BLE001
            redis_ok = False
            log.warning("health: Redis no responde: %s", e)
        base = s.componentes.base
        return HealthInfo(
            status="ok" if redis_ok else "degradado",
            modo="mock" if s.ajustes.modo_mock else "real",
            modelo=config.ANTHROPIC_MODEL,
            redis=EstadoRedis(ok=redis_ok, tipo=s.tipo_redis),
            checkpointer=s.checkpointer,
            langsmith=s.langsmith,
            rag=await base.estado() if base is not None else {"activo": False},
            jobs_activos=s.worker.activos,
            max_concurrentes=s.ajustes.max_concurrentes,
        )

    return app


app = create_app()
