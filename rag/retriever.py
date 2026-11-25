"""
rag/retriever.py — Retriever HÍBRIDO: vectorial + BM25, fusión RRF y re-ranking.

    consulta ─┬─► embeddings ─► VectorStore.buscar (filtro por destino) ─ top N ─┐
              │                                                               ├─► RRF ─► top M ─► re-ranker ─► top k
              └─► BM25 (mismos fragmentos, filtro por destino)          ─ top N ─┘

- Vectorial: encuentra fragmentos que DICEN lo mismo con otras palabras
  ("dónde esquiar" ↔ "centro de esquí").
- BM25 (rank-bm25): encuentra las palabras EXACTAS, clave para nombres propios
  ("Cerro Otto", "Tren a las Nubes") que los embeddings diluyen.
- RRF (Reciprocal Rank Fusion, como en la PE4): puntaje = Σ 1 / (RAG_RRF_K + puesto).
  Combina rankings sin tener que calibrar escalas distintas (coseno vs. BM25).
- Filtro por destino: sale de los METADATOS (slug), no del texto. Si el destino
  no tiene guía, se devuelve vacío y `destino_sin_guia=True`: el investigador
  sabe que tiene que ir a la memoria o a la web.
- Re-ranking: los mejores RERANKER_CANDIDATOS de la fusión se re-ordenan con
  el re-ranker y quedan los top_k.

El índice BM25 se arma sobre `VectorStore.todos()` (EXACTAMENTE los fragmentos
indexados) y se reconstruye cuando la ingesta cambia algo. rank-bm25 es CPU y
sync: la búsqueda corre en asyncio.to_thread.
"""

import asyncio
import logging
import time

from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi

from rag.documentos import Fragmento, FragmentoRecuperado
from rag.embeddings import Embeddings
from rag.reranker import Reranker, terminos
from rag.vectorstore import VectorStore

log = logging.getLogger(__name__)


class ResultadoBusqueda(BaseModel):
    """Lo que devuelve el retriever (y lo que guarda la traza de cada consulta)."""

    model_config = ConfigDict(extra="forbid")

    consulta: str
    destino: str | None = Field(description="Destino pedido (tal como llegó).")
    slug: str | None = Field(description="Guía elegida por el filtro de metadatos (None = todas).")
    destino_sin_guia: bool = False
    fragmentos: list[FragmentoRecuperado] = Field(default_factory=list)
    candidatos_vectoriales: int = 0
    candidatos_bm25: int = 0
    reranker: str = ""
    milisegundos: int = 0


class IndiceBM25:
    def __init__(self, fragmentos: list[Fragmento]) -> None:
        self.fragmentos = fragmentos
        self._por_slug: dict[str, list[int]] = {}
        for i, f in enumerate(fragmentos):
            self._por_slug.setdefault(f.slug, []).append(i)
        self._bm25 = BM25Okapi([terminos(f.para_embedding()) for f in fragmentos]) if fragmentos else None

    def buscar(self, consulta: str, k: int, slug: str | None = None) -> list[Fragmento]:
        if self._bm25 is None:
            return []
        puntajes = self._bm25.get_scores(terminos(consulta))
        indices = self._por_slug.get(slug, []) if slug else range(len(self.fragmentos))
        ordenados = sorted((i for i in indices if puntajes[i] > 0), key=lambda i: (-puntajes[i], i))
        return [self.fragmentos[i] for i in ordenados[:k]]


def fusion_rrf(rankings: list[list[Fragmento]], k_rrf: int) -> list[tuple[Fragmento, float, list[int | None]]]:
    """[(fragmento, puntaje RRF, [puesto en cada ranking])], de mayor a menor puntaje."""
    puntaje: dict[str, float] = {}
    puestos: dict[str, list[int | None]] = {}
    por_id: dict[str, Fragmento] = {}
    for r, ranking in enumerate(rankings):
        for puesto, f in enumerate(ranking, 1):
            por_id[f.id] = f
            puntaje[f.id] = puntaje.get(f.id, 0.0) + 1.0 / (k_rrf + puesto)
            puestos.setdefault(f.id, [None] * len(rankings))[r] = puesto
    orden = sorted(puntaje, key=lambda i: (-puntaje[i], i))
    return [(por_id[i], round(puntaje[i], 6), puestos[i]) for i in orden]


class RetrieverHibrido:
    def __init__(self, store: VectorStore, embeddings: Embeddings, reranker: Reranker, *, candidatos: int,
                 rrf_k: int, candidatos_rerank: int) -> None:
        self.store, self.embeddings, self.reranker = store, embeddings, reranker
        self.candidatos, self.rrf_k, self.candidatos_rerank = candidatos, rrf_k, candidatos_rerank
        self._bm25: IndiceBM25 | None = None

    async def recargar_bm25(self) -> int:
        fragmentos = await self.store.todos()
        self._bm25 = await asyncio.to_thread(IndiceBM25, fragmentos)
        return len(fragmentos)

    async def buscar(self, consulta: str, *, slug: str | None, k: int, modo: str = "hibrido",
                     usar_reranker: bool = True) -> list[FragmentoRecuperado]:
        """`modo`: "hibrido" | "vectorial" | "bm25" (los dos últimos los usa evaluate.py para comparar)."""
        return (await self._buscar(consulta, slug, k, modo, usar_reranker))[0]

    async def _buscar(self, consulta: str, slug: str | None, k: int, modo: str,
                      usar_reranker: bool) -> tuple[list[FragmentoRecuperado], int, int]:
        """Devuelve también cuántos candidatos trajo cada buscador (sin estado compartido entre jobs)."""
        if self._bm25 is None:
            await self.recargar_bm25()
        assert self._bm25 is not None
        vectoriales: list[Fragmento] = []
        lexicos: list[Fragmento] = []
        if modo in ("hibrido", "vectorial"):
            vector = await self.embeddings.consulta(consulta)
            vectoriales = [f for f, _ in await self.store.buscar(vector, self.candidatos, slug)]
        if modo in ("hibrido", "bm25"):
            lexicos = await asyncio.to_thread(self._bm25.buscar, consulta, self.candidatos, slug)
        fusion = fusion_rrf([vectoriales, lexicos], self.rrf_k)
        recuperados = [FragmentoRecuperado(fragmento=f, puesto_vectorial=p[0], puesto_bm25=p[1], puntaje_rrf=s)
                       for f, s, p in fusion]
        nv, nb = len(vectoriales), len(lexicos)
        if not usar_reranker or not recuperados:
            return recuperados[:k], nv, nb
        a_reordenar = recuperados[:self.candidatos_rerank]
        puntajes = await self.reranker.puntuar(consulta, [r.fragmento for r in a_reordenar])
        reordenados = sorted(zip(a_reordenar, puntajes, strict=True), key=lambda par: -par[1])
        return [r.model_copy(update={"puntaje_rerank": round(p, 4)}) for r, p in reordenados[:k]], nv, nb

    async def consultar(self, consulta: str, slug: str | None, destino: str | None, k: int,
                        destino_sin_guia: bool = False) -> ResultadoBusqueda:
        inicio = time.perf_counter()
        fragmentos, nv, nb = ([], 0, 0) if destino_sin_guia else await self._buscar(consulta, slug, k, "hibrido", True)
        resultado = ResultadoBusqueda(
            consulta=consulta, destino=destino, slug=slug, destino_sin_guia=destino_sin_guia, fragmentos=fragmentos,
            candidatos_vectoriales=nv, candidatos_bm25=nb, reranker=self.reranker.nombre,
            milisegundos=int((time.perf_counter() - inicio) * 1000),
        )
        log.info("RAG: '%s' [%s] → %d fragmentos (%d vect / %d bm25, re-ranker %s, %d ms)", consulta, slug or "todas",
                 len(fragmentos), nv, nb, self.reranker.nombre, resultado.milisegundos)
        return resultado
