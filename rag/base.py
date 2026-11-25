"""
rag/base.py — `BaseConocimiento`: la fachada del RAG que usan los agentes y la API.

Arma las piezas según la configuración (todo por variables de entorno) y expone
tres cosas:
- `consultar(consulta, destino, k)` → ResultadoBusqueda (la usa la herramienta
  consultar_guias del investigador);
- `resolver_destino(texto)` / `tiene_guia(destino)` → para el filtro por metadatos
  y para que el nodo HITL sepa si las guías cubren un destino;
- `asegurar_indice()` → ingesta idempotente (al arrancar la API, o la primera vez
  que se consulta).

Modo real: fastembed (EMBEDDINGS_MODELO) + VECTOR_STORE (chroma | pinecone) +
RERANKER (cross_encoder | llm | ninguno).
Modo mock / tests: embeddings por hashing + Chroma en una carpeta temporal +
re-ranker por solapamiento: sin descargas ni claves. La base mock se reutiliza
dentro del proceso (es de solo lectura una vez indexada).
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import config
from rag.documentos import DocumentoGuia, leer_corpus, normalizar
from rag.embeddings import Embeddings, EmbeddingsFastembed, EmbeddingsHash
from rag.ingesta import ReporteIngesta, ingestar
from rag.reranker import Reranker, RerankerCrossEncoder, RerankerLLM, RerankerSolapamiento, SinReranker
from rag.retriever import ResultadoBusqueda, RetrieverHibrido
from rag.vectorstore import ChromaVectorStore, PineconeVectorStore, VectorStore

log = logging.getLogger(__name__)


class BaseConocimiento:
    def __init__(self, corpus: list[DocumentoGuia], store: VectorStore, embeddings: Embeddings, reranker: Reranker,
                 *, chunk_tokens: int = config.CHUNK_TOKENS, chunk_overlap: int = config.CHUNK_OVERLAP,
                 top_k: int = config.RAG_TOP_K, candidatos: int = config.RAG_CANDIDATOS,
                 rrf_k: int = config.RAG_RRF_K, candidatos_rerank: int = config.RERANKER_CANDIDATOS) -> None:
        self.corpus = corpus
        self.store, self.embeddings, self.reranker = store, embeddings, reranker
        self.chunk_tokens, self.chunk_overlap, self.top_k = chunk_tokens, chunk_overlap, top_k
        self.retriever = RetrieverHibrido(store, embeddings, reranker, candidatos=candidatos, rrf_k=rrf_k,
                                          candidatos_rerank=candidatos_rerank)
        self._nombres: dict[str, str] = {normalizar(n): d.slug for d in corpus for n in d.nombres()}
        self._por_slug = {d.slug: d for d in corpus}
        self._listo = False
        self.ultimo_reporte: ReporteIngesta | None = None

    # ------------------------------------------------------------------
    def destinos(self) -> list[str]:
        return [d.destino for d in self.corpus]

    def documento(self, slug: str) -> DocumentoGuia | None:
        return self._por_slug.get(slug)

    def resolver_destino(self, texto: str | None) -> str | None:
        """'San Carlos de Bariloche', 'bariloche ', 'Cataratas' → slug de la guía (None si no hay guía)."""
        if not texto:
            return None
        t = normalizar(texto)
        if t in self._nombres:
            return self._nombres[t]
        # "Bariloche, Río Negro" contiene un nombre conocido; nombres largos primero ("puerto madryn" antes que "madryn").
        for nombre in sorted(self._nombres, key=len, reverse=True):
            if len(nombre) >= 4 and (nombre in t or (len(t) >= 4 and t in nombre)):
                return self._nombres[nombre]
        return None

    def tiene_guia(self, destino: str | None) -> bool:
        return self.resolver_destino(destino) is not None

    # ------------------------------------------------------------------
    async def asegurar_indice(self, forzar: bool = False) -> ReporteIngesta:
        """Ingesta idempotente. Sin carrera peligrosa: si dos tareas la corren a la vez, los upserts son por ID."""
        reporte = await ingestar(self.corpus, self.store, self.embeddings, self.embeddings.tokenizador,
                                 chunk_tokens=self.chunk_tokens, overlap=self.chunk_overlap, forzar=forzar)
        if reporte.hubo_cambios or not self._listo:
            await self.retriever.recargar_bm25()
        self._listo = True
        self.ultimo_reporte = reporte
        return reporte

    async def consultar(self, consulta: str, destino: str | None, k: int | None = None) -> ResultadoBusqueda:
        if not self._listo:
            await self.asegurar_indice()
        slug = self.resolver_destino(destino)
        sin_guia = bool(destino) and slug is None
        return await self.retriever.consultar(consulta, slug, destino, k or self.top_k, destino_sin_guia=sin_guia)

    async def estado(self) -> dict[str, Any]:
        return {"store": self.store.nombre, "fragmentos": await self.store.contar(),
                "embeddings": self.embeddings.nombre, "reranker": self.reranker.nombre,
                "destinos": self.destinos(), "listo": self._listo}

    async def cerrar(self) -> None:
        await self.store.cerrar()


# ---------------------------------------------------------------------------
# Fábrica (el único lugar que mira VECTOR_STORE, EMBEDDINGS_MODELO y RERANKER)
# ---------------------------------------------------------------------------
_BASE_MOCK: dict[str, BaseConocimiento] = {}


def crear_vector_store(dimension: int, ruta: Path | None = None) -> VectorStore:
    if config.VECTOR_STORE == "pinecone":
        a = config.AJUSTES
        return PineconeVectorStore(api_key=a.pinecone_api_key, indice=a.pinecone_index,
                                   namespace=a.pinecone_namespace, dimension=dimension, cloud=a.pinecone_cloud,
                                   region=a.pinecone_region)
    return ChromaVectorStore(ruta or config.INDICE_DIR, config.GUIAS_COLECCION)


def crear_reranker(llm: Any = None) -> Reranker:
    if config.RERANKER == "cross_encoder":
        return RerankerCrossEncoder(config.RERANKER_MODELO, config.MODELOS_CACHE_DIR, config.RERANKER_MAX_CARACTERES)
    if config.RERANKER == "llm":
        if llm is None:
            from agents.llm import crear_llm_real

            llm = crear_llm_real()
        return RerankerLLM(llm)
    return SinReranker()


def crear_base_conocimiento(modo_mock: bool = config.MODO_MOCK, *, guias_dir: Path | None = None,
                            ruta_indice: Path | None = None, llm_reranker: Any = None) -> BaseConocimiento:
    carpeta = guias_dir or config.GUIAS_DIR
    corpus = leer_corpus(carpeta)
    if modo_mock:
        clave = str(carpeta.resolve())
        if ruta_indice is None and clave in _BASE_MOCK:
            return _BASE_MOCK[clave]
        ruta = ruta_indice or Path(tempfile.mkdtemp(prefix="indice_mock_")) / "chroma"
        base = BaseConocimiento(corpus, ChromaVectorStore(ruta, config.GUIAS_COLECCION), EmbeddingsHash(),
                                RerankerSolapamiento())
        if ruta_indice is None:
            _BASE_MOCK[clave] = base
        return base
    embeddings = EmbeddingsFastembed(config.EMBEDDINGS_MODELO, config.MODELOS_CACHE_DIR)
    return BaseConocimiento(corpus, crear_vector_store(embeddings.dimension, ruta_indice), embeddings,
                            crear_reranker(llm_reranker))
