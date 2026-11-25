"""
api/schemas.py — Los esquemas Pydantic de TODAS las entradas y salidas de la API.

Cada endpoint declara su `response_model` con estos modelos, así /docs (OpenAPI)
muestra los campos, sus descripciones y un ejemplo de cada respuesta.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PATRON_ID: str = r"^[0-9a-f]{32}$"

EJEMPLO_PREGUNTA = ("Quiero ir 7 días a Bariloche en enero, somos 2, salimos de Mendoza y tenemos 800.000 pesos. "
                    "¿Me armás el viaje y me decís si nos alcanza?")


class EstadoJob(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


class TaskRequest(BaseModel):
    """POST /tasks: una pregunta nueva, o un seguimiento de una conversación (con su thread_id)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"pregunta": EJEMPLO_PREGUNTA},
        {"pregunta": "¿Y si vamos en julio?", "thread_id": "4f1c2a9e0b7d4c3a8e6f5d2c1b0a9f8e"},
        {"pregunta": "¿Qué se puede hacer en Salta y qué clima hay en julio?", "auto_aprobar": True},
    ]})

    pregunta: str = Field(min_length=5, max_length=2000, description="El pedido de viaje, en castellano.")
    thread_id: str | None = Field(
        default=None, pattern=PATRON_ID,
        description="Para SEGUIR una conversación: el thread_id que devolvió un POST anterior. El grafo retoma el "
                    "estado guardado en el checkpointer (Redis), aunque la API se haya reiniciado.")
    auto_aprobar: bool = Field(default=False, description="true: aprueba la búsqueda web sin pausar (prueba de carga).")


class TaskCreada(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "job_id": "a4be1ccdf4694ac9861baa8a9731eeee", "thread_id": "a4be1ccdf4694ac9861baa8a9731eeee", "turno": 1,
        "estado": "PENDING", "links": {"estado": "/tasks/a4be1ccdf4694ac9861baa8a9731eeee",
                                       "aprobar": "/tasks/a4be1ccdf4694ac9861baa8a9731eeee/approve",
                                       "conversacion": "/threads/a4be1ccdf4694ac9861baa8a9731eeee"}}]})

    job_id: str
    thread_id: str = Field(description="Id de la conversación: mandalo en el próximo POST /tasks para seguirla.")
    turno: int = Field(ge=1, description="Número de turno dentro de la conversación (1 = primera pregunta).")
    estado: EstadoJob
    links: dict[str, str]


class JobInfo(BaseModel):
    """GET /tasks/{job_id}: estado, progreso, pausa HITL, resultado o error de un job."""

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "job_id": "a4be1ccdf4694ac9861baa8a9731eeee", "estado": "WAITING_APPROVAL", "pregunta": EJEMPLO_PREGUNTA,
        "modo": "real", "thread_id": "a4be1ccdf4694ac9861baa8a9731eeee", "turno": 1,
        "creado": "2026-11-25T14:00:00.000-03:00", "nodo_actual": "aprobacion_web", "ruta": ["supervisor"],
        "aprobacion_pendiente": {"tipo": "aprobacion_busqueda_web", "destino": "Bariloche",
                                 "temas_sin_memoria": ["pasajes", "alojamiento", "comida", "cambio"],
                                 "temas_en_guias": ["actividades", "clima"], "temas_en_memoria": []},
        "tramos": 1, "total_tokens": 9120, "costo_usd": 0.0012}]})

    job_id: str
    estado: EstadoJob
    pregunta: str
    auto_aprobar: bool = False
    modo: str
    thread_id: str
    turno: int = 1
    creado: str
    iniciado: str | None = None
    actualizado: str | None = None
    terminado: str | None = None
    nodo_actual: str | None = None
    ruta: list[str] = []
    aprobacion_pendiente: dict[str, Any] | None = Field(
        default=None, description="Con estado WAITING_APPROVAL: qué quiere buscar en la web el investigador.")
    aprobaciones: list[dict[str, Any]] = []
    resultado: dict[str, Any] | None = Field(
        default=None, description="Con estado DONE: respuesta_final, ruta, permiso_web, métricas, pedido y traza.")
    error: str | None = Field(default=None, description="Con estado FAILED: tipo y mensaje del error.")
    tramos: int = Field(default=0, description="Cuántas veces corrió el grafo (2 si hubo pausa HITL y se reanudó).")
    duracion_s: float | None = Field(default=None, description="De la creación al final (incluye la espera humana).")
    tiempo_ejecucion_s: float | None = Field(default=None, description="Solo el tiempo que corrió el grafo.")
    total_tokens: int = 0
    costo_usd: float | None = Field(default=None, description="Costo estimado (tokens × precios de observability/).")
    tokens: dict[str, Any] = {}


class AprobacionRequest(BaseModel):
    """POST /tasks/{job_id}/approve: la respuesta de la persona a la pausa HITL."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"aprobado": True}, {"aprobado": False, "comentario": "Sin gastar búsquedas: usá lo que haya."}]})

    aprobado: bool = Field(description="true: puede buscar en la web; false: sigue con las guías y la memoria.")
    comentario: str = Field(default="", max_length=500, description="Opcional, queda en el historial del job.")


class AprobacionAceptada(BaseModel):
    job_id: str
    estado: EstadoJob
    aprobado: bool


class TurnoInfo(BaseModel):
    turno: int
    pregunta: str
    respuesta: str
    pedido: dict[str, Any] | None = None
    timestamp: str


class ConversacionInfo(BaseModel):
    """GET /threads/{thread_id}: el historial de una conversación, leído del checkpoint de LangGraph."""

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "thread_id": "a4be1ccdf4694ac9861baa8a9731eeee", "turnos_completos": 1,
        "turnos": [{"turno": 1, "pregunta": EJEMPLO_PREGUNTA, "respuesta": "## Itinerario ...",
                    "pedido": {"destino": "Bariloche", "fecha_ida": "2027-01-10"}, "timestamp": "2026-11-25T14:02:00"}],
        "jobs": ["a4be1ccdf4694ac9861baa8a9731eeee"], "job_activo": None, "checkpointer": "AsyncRedisSaver"}]})

    thread_id: str
    turnos_completos: int
    turnos: list[TurnoInfo]
    jobs: list[str] = Field(description="Los jobs de la conversación, en orden (uno por turno).")
    job_activo: str | None = Field(description="Job que todavía no terminó (no se puede seguir hasta que termine).")
    checkpointer: str


class EstadoRedis(BaseModel):
    ok: bool
    tipo: str


class HealthInfo(BaseModel):
    """GET /health: si las dependencias responden y con qué configuración corre la API."""

    status: str = Field(description="'ok' o 'degradado' (Redis no responde).")
    modo: str
    modelo: str
    redis: EstadoRedis
    checkpointer: str
    langsmith: dict[str, Any]
    rag: dict[str, Any]
    jobs_activos: int
    max_concurrentes: int


class Error(BaseModel):
    detail: str
