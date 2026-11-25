"""
api/worker.py — El estado de cada job en Redis y el worker que corre el grafo.

JobStore (Redis, todo async con redis.asyncio): un HASH por job, `job:<job_id>`, con
    pregunta, estado, auto_aprobar, modo, thread_id, turno, timestamps (creado,
    iniciado, actualizado, terminado), nodo_actual, ruta, aprobacion_pendiente,
    aprobaciones, resultado, error, tokens, costo_usd...
más una lista `job:<job_id>:eventos` (lo que hizo cada nodo, para la traza), un
sorted set `jobs` (por fecha, para GET /tasks) y, por conversación, una lista
`thread:<thread_id>:jobs` (un job por turno).

Estados: PENDING → RUNNING → (WAITING_APPROVAL → RUNNING)* → DONE | FAILED.
Las transiciones que pueden competir (dos approve a la vez) usan WATCH/MULTI.

Conversaciones: el grafo corre con config thread_id = el de la conversación (el
primer job usa su propio id como thread_id). Un seguimiento es un job nuevo en el
MISMO thread: entra con `entrada_turno(pregunta, turno=n)` y LangGraph retoma el
estado guardado en el checkpointer (AsyncRedisSaver): el pedido, los aportes y las
respuestas de los turnos anteriores. Por eso sobrevive a un reinicio de la API.

Worker: cada job es una tarea de asyncio (`asyncio.create_task`) cuya referencia
se guarda en un set hasta que termina. Un semáforo limita cuántos grafos corren a
la vez. TODA excepción termina en FAILED con el error (tipo + mensaje): el cliente
nunca queda haciendo polling infinito. También hay un timeout por tramo.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis
from redis.exceptions import WatchError

import config
from agents.llm import explicar_error
from api.schemas import EstadoJob
from graph.state import ahora, entrada_turno
from observability import monitoreo as observability
from observability.traza import Evento, Traza

log = logging.getLogger(__name__)

TERMINALES: frozenset[str] = frozenset({EstadoJob.DONE, EstadoJob.FAILED})
CAMPOS_JSON: frozenset[str] = frozenset({"ruta", "aprobacion_pendiente", "aprobaciones", "resultado", "tokens"})
CAMPOS_NUMERICOS: frozenset[str] = frozenset({"costo_usd", "duracion_s", "tiempo_ejecucion_s"})
CAMPOS_ENTEROS: frozenset[str] = frozenset({"tramos", "total_tokens", "turno"})


def _a_redis(valor: Any) -> str:
    if isinstance(valor, bool):
        return "1" if valor else "0"
    if valor is None:
        return ""
    if isinstance(valor, (dict, list)):
        return json.dumps(valor, ensure_ascii=False, default=str)
    return str(valor)


def _texto(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else v


def _segundos(desde: str, hasta: str) -> float:
    return round((datetime.fromisoformat(hasta) - datetime.fromisoformat(desde)).total_seconds(), 2)


class JobStore:
    """El estado de los jobs en Redis (todas las operaciones son async: no bloquean el event loop)."""

    def __init__(self, redis: Redis, prefijo: str = "job") -> None:
        self.redis = redis
        self.prefijo = prefijo

    def clave(self, job_id: str) -> str:
        return f"{self.prefijo}:{job_id}"

    @staticmethod
    def clave_thread(thread_id: str) -> str:
        return f"thread:{thread_id}:jobs"

    async def crear(self, pregunta: str, auto_aprobar: bool, modo: str, thread_id: str | None = None,
                    turno: int = 1) -> str:
        job_id = uuid.uuid4().hex
        thread_id = thread_id or job_id  # una conversación nueva usa el id de su primer job
        momento = ahora()
        async with self.redis.pipeline(transaction=True) as pipe:  # una sola ida y vuelta a Redis
            pipe.hset(self.clave(job_id), mapping={
                "id": job_id, "thread_id": thread_id, "turno": str(turno), "pregunta": pregunta,
                "estado": EstadoJob.PENDING, "auto_aprobar": _a_redis(auto_aprobar), "modo": modo,
                "creado": momento, "actualizado": momento, "ruta": "[]", "aprobaciones": "[]", "tokens": "{}",
                "tramos": "0",
            })
            pipe.zadd(f"{self.prefijo}s", {job_id: time.time()})
            pipe.rpush(self.clave_thread(thread_id), job_id)
            await pipe.execute()
        return job_id

    async def obtener(self, job_id: str) -> dict[str, Any] | None:
        crudo = await self.redis.hgetall(self.clave(job_id))
        if not crudo:
            return None
        datos: dict[str, Any] = {}
        for k, v in crudo.items():
            k, v = _texto(k), _texto(v)
            if k in CAMPOS_JSON:
                datos[k] = json.loads(v) if v else None
            elif k in CAMPOS_NUMERICOS:
                datos[k] = float(v) if v else None
            elif k in CAMPOS_ENTEROS:
                datos[k] = int(v) if v else 0
            elif k == "auto_aprobar":
                datos[k] = v == "1"
            else:
                datos[k] = v or None
        datos.setdefault("turno", 1)
        return datos

    async def actualizar(self, job_id: str, **campos: Any) -> None:
        campos["actualizado"] = ahora()
        await self.redis.hset(self.clave(job_id), mapping={k: _a_redis(v) for k, v in campos.items()})

    async def transicion(self, job_id: str, desde: set[str], hacia: EstadoJob, **campos: Any) -> bool:
        """Cambia el estado SOLO si el actual está en `desde` (atómico con WATCH/MULTI). Devuelve si cambió."""
        clave = self.clave(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(clave)
                actual = _texto(await pipe.hget(clave, "estado"))
                if actual not in desde:
                    await pipe.unwatch()
                    return False
                pipe.multi()
                campos.update(estado=hacia, actualizado=ahora())
                pipe.hset(clave, mapping={k: _a_redis(v) for k, v in campos.items()})
                await pipe.execute()
                return True
            except WatchError:  # otro request cambió el job en el medio
                return False

    async def agregar_evento(self, job_id: str, evento: dict[str, Any]) -> None:
        clave = self.clave(job_id)
        ruta = json.loads((await self.redis.hget(clave, "ruta")) or "[]")
        ruta.append(evento["nodo"])
        await self.redis.rpush(f"{clave}:eventos", json.dumps(evento, ensure_ascii=False, default=str))
        await self.actualizar(job_id, nodo_actual=evento["nodo"], ruta=ruta)

    async def eventos(self, job_id: str) -> list[dict[str, Any]]:
        return [json.loads(e) for e in await self.redis.lrange(f"{self.clave(job_id)}:eventos", 0, -1)]

    async def listar(self, limite: int = 20) -> list[dict[str, Any]]:
        ids = await self.redis.zrevrange(f"{self.prefijo}s", 0, max(0, limite - 1))
        jobs = [await self.obtener(_texto(i)) for i in ids]
        return [j for j in jobs if j]

    async def en_estado(self, *estados: str) -> list[str]:
        ids = await self.redis.zrange(f"{self.prefijo}s", 0, -1)
        salida = []
        for i in map(_texto, ids):
            if _texto(await self.redis.hget(self.clave(i), "estado")) in estados:
                salida.append(i)
        return salida

    async def jobs_del_thread(self, thread_id: str) -> list[str]:
        return [_texto(i) for i in await self.redis.lrange(self.clave_thread(thread_id), 0, -1)]

    async def job_activo(self, thread_id: str) -> str | None:
        """El job de la conversación que todavía no terminó (a lo sumo uno), o None."""
        for job_id in reversed(await self.jobs_del_thread(thread_id)):
            if _texto(await self.redis.hget(self.clave(job_id), "estado")) not in TERMINALES:
                return job_id
        return None


class Worker:
    """Corre el grafo de cada job en una tarea de asyncio de fondo."""

    def __init__(
        self,
        grafo: CompiledStateGraph,
        store: JobStore,
        *,
        modo_mock: bool,
        max_concurrentes: int = 5,
        timeout_s: float = 900.0,
        guardar_trazas: bool = False,
    ) -> None:
        self.grafo = grafo
        self.store = store
        self.modo_mock = modo_mock
        self.timeout_s = timeout_s
        self.guardar_trazas = guardar_trazas
        self._semaforo = asyncio.Semaphore(max_concurrentes)
        self._tareas: set[asyncio.Task[None]] = set()  # referencias fuertes hasta que cada tarea termina

    @property
    def activos(self) -> int:
        return len(self._tareas)

    def lanzar(self, job_id: str, entrada: dict[str, Any] | Command | None) -> asyncio.Task[None]:
        """Encola el job y vuelve enseguida (lo llama el endpoint; no espera al grafo)."""
        tarea = asyncio.create_task(self._ejecutar(job_id, entrada), name=f"job-{job_id}")
        self._tareas.add(tarea)
        tarea.add_done_callback(self._tareas.discard)
        return tarea

    async def iniciar(self, job_id: str, pregunta: str, auto_aprobar: bool, turno: int = 1) -> asyncio.Task[None]:
        return self.lanzar(job_id, entrada_turno(pregunta, turno=turno, auto_aprobar=auto_aprobar))

    async def estado_thread(self, thread_id: str) -> Any:
        """El snapshot del checkpoint de la conversación (StateSnapshot de LangGraph)."""
        return await self.grafo.aget_state({"configurable": {"thread_id": thread_id}})

    async def esperar(self) -> None:
        """Espera a que terminen las tareas en curso (tests y prueba de carga en proceso)."""
        while self._tareas:
            await asyncio.gather(*list(self._tareas), return_exceptions=True)

    async def cerrar(self) -> None:
        """Al apagar la API: se cancelan las tareas; cada una deja su job en FAILED (no queda colgado)."""
        for t in list(self._tareas):
            t.cancel()
        await asyncio.gather(*list(self._tareas), return_exceptions=True)

    async def recuperar(self, checkpoint_persistente: bool) -> list[str]:
        """Al arrancar: jobs que quedaron RUNNING/PENDING porque la API se cayó sin apagarse bien.

        Con checkpoints en Redis: si el thread tiene un paso pendiente DE ESE TURNO, se reanuda
        desde el último checkpoint (entrada None); si no, el turno se arranca de cero.
        Sin checkpoints persistentes no hay de dónde seguir → FAILED con el motivo.
        Los WAITING_APPROVAL no se tocan: siguen esperando su approve.
        """
        recuperados = []
        for job_id in await self.store.en_estado(EstadoJob.PENDING, EstadoJob.RUNNING):
            job = await self.store.obtener(job_id) or {}
            if not checkpoint_persistente:
                await self._fallar(job_id, "La API se reinició mientras el job corría y no hay checkpoints "
                                           "persistentes (modo sin Redis real).")
                continue
            snap = await self.estado_thread(job.get("thread_id") or job_id)
            turno = int(job.get("turno") or 1)
            if snap.next and int((snap.values or {}).get("turno") or 1) == turno:
                log.warning("[worker] job %s: se reanuda desde su último checkpoint", job_id)
                self.lanzar(job_id, None)
            else:  # nunca llegó a correr este turno: se arranca de cero
                self.lanzar(job_id, entrada_turno(job.get("pregunta") or "?", turno, bool(job.get("auto_aprobar"))))
            recuperados.append(job_id)
        return recuperados

    # ------------------------------------------------------------------
    async def _ejecutar(self, job_id: str, entrada: dict[str, Any] | Command | None) -> None:
        tokens = UsageMetadataCallbackHandler()  # afuera del try: si el tramo falla, sus tokens igual se cuentan
        try:
            async with self._semaforo:
                await asyncio.wait_for(self._correr_tramo(job_id, entrada, tokens), timeout=self.timeout_s)
        except asyncio.CancelledError:
            await asyncio.shield(self._fallar(job_id, "CancelledError: la API se apagó mientras el job corría.", tokens))
            raise
        except TimeoutError:
            await self._fallar(job_id, f"Timeout: el tramo superó los {self.timeout_s:.0f} s.", tokens)
        except Exception as e:  # noqa: BLE001 — toda excepción del grafo termina en FAILED
            log.exception("[worker] job %s falló", job_id)
            await self._fallar(job_id, _explicar(e), tokens)

    async def _fallar(self, job_id: str, error: str, tokens: UsageMetadataCallbackHandler | None = None) -> None:
        try:
            job = await self.store.obtener(job_id) or {}
            momento = ahora()
            extra: dict[str, Any] = {}
            if tokens is not None and tokens.usage_metadata:
                acumulados = observability.sumar_tokens(job.get("tokens") or {}, tokens.usage_metadata)
                extra = {"tokens": acumulados, "costo_usd": observability.costo_usd(acumulados),
                         "total_tokens": observability.total_tokens(acumulados)}
            await self.store.actualizar(job_id, estado=EstadoJob.FAILED, error=error, terminado=momento,
                                        aprobacion_pendiente=None, **extra,
                                        duracion_s=_segundos(job["creado"], momento) if job.get("creado") else None)
        except Exception:  # noqa: BLE001 — si ni Redis responde, al menos queda en el log
            log.exception("[worker] no se pudo marcar FAILED el job %s (error original: %s)", job_id, error)

    async def _correr_tramo(self, job_id: str, entrada: dict[str, Any] | Command | None,
                            tokens: UsageMetadataCallbackHandler) -> None:
        job = await self.store.obtener(job_id)
        if job is None:
            raise KeyError(f"El job {job_id} no existe en Redis.")
        thread_id = job.get("thread_id") or job_id
        turno = int(job.get("turno") or 1)
        tramo = int(job.get("tramos") or 0) + 1
        inicio_tramo = time.perf_counter()
        await self.store.actualizar(job_id, estado=EstadoJob.RUNNING, tramos=tramo, iniciado=job.get("iniciado") or ahora())

        cfg = observability.config_de_job(job_id, thread_id=thread_id, turno=turno, modo_mock=self.modo_mock,
                                          tramo=tramo, callbacks=[tokens])
        auxiliar = Traza(pregunta=job["pregunta"])  # solo para armar el detalle de cada evento igual que en la PE6
        cortado = False
        try:
            async for modo, dato in self.grafo.astream(entrada, cfg, stream_mode=["updates", "values"]):
                if modo != "updates":
                    continue
                for nodo, salida in dato.items():
                    if nodo.startswith("__"):  # "__interrupt__": la pausa se lee del estado más abajo
                        continue
                    auxiliar.registrar(nodo, salida or {})
                    e = auxiliar.eventos[-1]
                    await self.store.agregar_evento(job_id, {"nodo": e.nodo, "timestamp": e.timestamp,
                                                             "detalle": e.detalle, "tramo": tramo})
        except GraphRecursionError:
            cortado = True  # freno de LangGraph: se termina de forma controlada con lo que haya (igual que la PE6)

        job = await self.store.obtener(job_id) or job
        acumulados = observability.sumar_tokens(job.get("tokens") or {}, tokens.usage_metadata)
        ejecucion = round((job.get("tiempo_ejecucion_s") or 0) + time.perf_counter() - inicio_tramo, 2)
        comunes = {"tokens": acumulados, "costo_usd": observability.costo_usd(acumulados),
                   "total_tokens": observability.total_tokens(acumulados), "tiempo_ejecucion_s": ejecucion}

        snapshot = await self.estado_thread(thread_id)
        if snapshot.interrupts and not cortado:
            pendiente = snapshot.interrupts[0].value
            log.info("[worker] job %s: pausa HITL, espera aprobación", job_id)
            await self.store.actualizar(job_id, estado=EstadoJob.WAITING_APPROVAL, nodo_actual="aprobacion_web",
                                        aprobacion_pendiente=pendiente, **comunes)
            return

        await self._terminar(job_id, job, dict(snapshot.values), cortado, comunes)

    async def _terminar(self, job_id: str, job: dict[str, Any], valores: dict[str, Any], cortado: bool,
                        comunes: dict[str, Any]) -> None:
        turno = int(job.get("turno") or 1)
        traza = Traza(pregunta=job["pregunta"], escenario=f"api_{job_id[:8]}_t{turno}",
                      modo="mock" if self.modo_mock else "real", inicio=job["creado"])
        for i, e in enumerate(await self.store.eventos(job_id), 1):
            traza.ruta.append(e["nodo"])
            traza.eventos.append(Evento(n=i, nodo=e["nodo"], timestamp=e["timestamp"], detalle=e["detalle"]))
        aviso = (f"Me detuve de forma controlada: el grafo llegó a su límite de {config.RECURSION_LIMIT} pasos "
                 "(RECURSION_LIMIT) antes de terminar.") if cortado else None
        traza.cerrar("cortado_por_recursion_limit" if cortado else "completo", valores, comunes["tokens"], aviso=aviso)
        if self.guardar_trazas:
            await asyncio.to_thread(traza.guardar, config.TRACES_DIR / "api")  # E/S de disco: fuera del event loop
        resultado = {
            "respuesta_final": traza.respuesta_final,
            "estado_grafo": traza.estado,
            "ruta": traza.texto_ruta(),
            "turno": turno,
            "thread_id": job.get("thread_id") or job_id,
            "permiso_web": valores.get("permiso_web", "libre"),
            "aprobaciones_hitl": [r.model_dump(mode="json") if hasattr(r, "model_dump") else r
                                  for r in valores.get("aprobaciones", []) if getattr(r, "turno", turno) == turno],
            "metricas": traza.metricas,
            "pedido": traza.pedido,
            "traza": traza.archivo or None,
        }
        momento = ahora()
        await self.store.actualizar(job_id, estado=EstadoJob.DONE, resultado=resultado, terminado=momento,
                                    nodo_actual="", aprobacion_pendiente=None,
                                    duracion_s=_segundos(job["creado"], momento), **comunes)
        log.info("[worker] job %s DONE en %.1f s (%s)", job_id, _segundos(job["creado"], momento), traza.texto_ruta())


def _explicar(e: Exception) -> str:
    """Tipo + mensaje, con la traducción de los errores típicos de la API de Claude."""
    texto = explicar_error(e)
    return texto if texto.startswith(type(e).__name__) else f"{type(e).__name__}: {texto}"
