"""
config.py — Configuración central del asesor de viajes (capa "config").

TODO lo que puede cambiar entre entornos (claves, modelos, índices, URLs,
límites) se lee de variables de entorno (o del .env) y se VALIDA con
pydantic-settings al arrancar: un valor inválido falla enseguida con un
mensaje claro, no a mitad de una corrida. No hay modelos, índices ni URLs
escritos en el código: los valores por defecto viven en `Ajustes` y en
.env.example, que documenta cada variable.

El resto de los módulos usa las constantes de abajo (config.ANTHROPIC_MODEL,
config.RAG_TOP_K, ...), que salen de `AJUSTES`. Los tests pueden cambiarlas con
monkeypatch como cualquier atributo de módulo.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent


class Ajustes(BaseSettings):
    """Variables de entorno del sistema (ver .env.example: cada una está comentada ahí)."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore",
                                      case_sensitive=False)

    # --- LLM (Claude) ---
    anthropic_api_key: str = ""
    anthropic_model: str = Field(default="claude-haiku-5-5", min_length=1)
    max_tokens_respuesta: int = Field(default=8000, ge=256)
    llm_timeout_s: float = Field(default=90, gt=0)
    llm_max_reintentos: int = Field(default=3, ge=0)

    # --- Búsqueda web (Tavily) ---
    tavily_api_key: str = ""
    tavily_max_resultados: int = Field(default=3, ge=1, le=10)

    # --- Modo mock: LLM con políticas + Tavily simulado + embeddings offline (sin claves ni internet) ---
    modo_mock: bool = False

    # --- RAG: base de conocimiento (guías de Wikivoyage) ---
    guias_dir: Path = BASE_DIR / "data" / "guias"
    vector_store: Literal["chroma", "pinecone"] = "chroma"
    chroma_dir: Path = BASE_DIR / "data" / "indice_chroma"
    guias_coleccion: str = Field(default="guias_viaje", min_length=3)
    pinecone_api_key: str = ""
    pinecone_index: str = "guias-viaje"
    pinecone_namespace: str = "guias"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embeddings_modelo: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", min_length=3)
    modelos_cache_dir: Path = BASE_DIR / "data" / "modelos"
    chunk_tokens: int = Field(default=500, ge=50, le=2000)
    chunk_overlap: int = Field(default=60, ge=0)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_candidatos: int = Field(default=20, ge=1, le=100, description="Candidatos de cada buscador antes de fusionar.")
    rag_rrf_k: int = Field(default=60, ge=1)
    reranker: Literal["cross_encoder", "llm", "ninguno"] = "cross_encoder"
    reranker_modelo: str = "jinaai/jina-reranker-v2-base-multilingual"
    reranker_candidatos: int = Field(default=12, ge=1, le=50, description="Cuántos fusionados re-ordena el re-ranker.")
    ingesta_al_arrancar: bool = True

    # --- Memoria de búsquedas web (ChromaDB) ---
    memoria_dir: Path = BASE_DIR / "data" / "memoria_chroma"
    memoria_coleccion: str = "busquedas_viajes"
    memoria_max_dias: int = Field(default=7, ge=0)
    memoria_max_distancia: float = Field(default=0.55, ge=0, le=2)

    # --- Frenos del grafo ---
    max_pasos_supervisor: int = Field(default=6, ge=2)
    recursion_limit: int = Field(default=20, ge=5)
    recursion_limit_especialista: int = Field(default=12, ge=3)

    # --- API, Redis y worker ---
    redis_url: str = ""
    redis_max_conexiones: int = Field(default=20, ge=4)
    max_jobs_concurrentes: int = Field(default=5, ge=1)
    job_timeout_s: float = Field(default=900, gt=0)
    checkpoint_ttl_minutos: int = Field(default=7 * 24 * 60, ge=1)

    # --- Observabilidad (LangSmith) ---
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "entrega-final"
    langsmith_endpoint: str = ""

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_menor_que_chunk(cls, v: int, info) -> int:  # noqa: ANN001
        tokens = info.data.get("chunk_tokens", 500)
        if v >= tokens:
            raise ValueError(f"CHUNK_OVERLAP ({v}) tiene que ser menor que CHUNK_TOKENS ({tokens})")
        return v

    @field_validator("guias_dir", "chroma_dir", "memoria_dir", "modelos_cache_dir")
    @classmethod
    def _rutas_relativas_a_la_raiz(cls, v: Path) -> Path:
        return v if v.is_absolute() else BASE_DIR / v


AJUSTES = Ajustes()

# ---------------------------------------------------------------------------
# Constantes que usa el resto del código (salen de AJUSTES)
# ---------------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
BUSQUEDAS_MOCK: Path = DATA_DIR / "mock_busquedas.json"   # resultados de Tavily prefijados (modo mock)
TRACES_DIR: Path = BASE_DIR / "traces"

ANTHROPIC_API_KEY: str = AJUSTES.anthropic_api_key.strip()
ANTHROPIC_MODEL: str = AJUSTES.anthropic_model.strip()
TAVILY_API_KEY: str = AJUSTES.tavily_api_key.strip()
MODO_MOCK: bool = AJUSTES.modo_mock
MAX_TOKENS_RESPUESTA: int = AJUSTES.max_tokens_respuesta

# RAG
GUIAS_DIR: Path = AJUSTES.guias_dir
VECTOR_STORE: str = AJUSTES.vector_store
INDICE_DIR: Path = AJUSTES.chroma_dir
GUIAS_COLECCION: str = AJUSTES.guias_coleccion
EMBEDDINGS_MODELO: str = AJUSTES.embeddings_modelo
MODELOS_CACHE_DIR: Path = AJUSTES.modelos_cache_dir
CHUNK_TOKENS: int = AJUSTES.chunk_tokens
CHUNK_OVERLAP: int = AJUSTES.chunk_overlap
RAG_TOP_K: int = AJUSTES.rag_top_k
RAG_CANDIDATOS: int = AJUSTES.rag_candidatos
RAG_RRF_K: int = AJUSTES.rag_rrf_k
RERANKER: str = AJUSTES.reranker
RERANKER_MODELO: str = AJUSTES.reranker_modelo
RERANKER_CANDIDATOS: int = AJUSTES.reranker_candidatos

# Memoria de búsquedas
CHROMA_DIR: Path = AJUSTES.memoria_dir
CHROMA_COLECCION: str = AJUSTES.memoria_coleccion
# Un dato guardado sirve si tiene a lo sumo esta antigüedad (precios y clima cambian).
MEMORIA_MAX_DIAS: int = AJUSTES.memoria_max_dias
# Distancia coseno máxima para considerar relevante un resultado de memoria. Calibrado en la PE6 con
# all-MiniLM-L6-v2: la misma consulta da 0.09; parecidas, 0.24-0.44; de otro tema, 0.64-0.72.
MEMORIA_MAX_DISTANCIA: float = AJUSTES.memoria_max_distancia
TAVILY_MAX_RESULTADOS: int = AJUSTES.tavily_max_resultados

# Fecha de "hoy": interpreta "en enero" como el próximo enero y valida que el viaje no sea en el pasado.
FECHA_REFERENCIA: date = date.today()

# Frenos contra el "supervisor infinito"
MAX_PASOS_SUPERVISOR: int = AJUSTES.max_pasos_supervisor      # freno 1: decisiones del supervisor
RECURSION_LIMIT: int = AJUSTES.recursion_limit                # freno 2: nodos del grafo principal
RECURSION_LIMIT_ESPECIALISTA: int = AJUSTES.recursion_limit_especialista  # freno 3: ciclo ReAct interno


class ErrorDeConfiguracion(Exception):
    """Falta una clave del .env: el programa no puede arrancar."""


def validar_entorno(modo_mock: bool = MODO_MOCK) -> None:
    """Verifica que las claves obligatorias estén en el .env. En modo mock no hace falta ninguna."""
    if modo_mock:
        return
    obligatorias = {"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY, "TAVILY_API_KEY": TAVILY_API_KEY}
    if VECTOR_STORE == "pinecone":
        obligatorias["PINECONE_API_KEY"] = AJUSTES.pinecone_api_key
    faltan = [nombre for nombre, valor in obligatorias.items() if not valor.strip()]
    if faltan:
        raise ErrorDeConfiguracion(
            f"Faltan variables en el .env: {', '.join(faltan)}. Completalas en el .env (copiá .env.example si no "
            "existe), o corré en modo mock (MODO_MOCK=1 o --mock)."
        )


def configurar_logs(nivel: int = logging.INFO) -> None:
    """Logs con hora y milisegundos; se silencian las librerías ruidosas."""
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in ("httpx", "httpcore", "urllib3", "anthropic", "chromadb", "onnxruntime", "fastembed",
                    "huggingface_hub", "langsmith"):
        logging.getLogger(ruidoso).setLevel(logging.ERROR)
