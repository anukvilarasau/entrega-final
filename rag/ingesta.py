"""
rag/ingesta.py — Ingesta IDEMPOTENTE de las guías: limpieza → chunking → embeddings → índice.

    python -m rag.ingesta              # indexa lo nuevo o lo que cambió (lo demás no se toca)
    python -m rag.ingesta --forzar     # reindexa todo
    python -m rag.ingesta --mock       # con embeddings offline, en una carpeta temporal (prueba sin descargas)

Idempotencia: cada fragmento guarda `doc_hash` = hash del documento + "huella"
(modelo de embeddings, CHUNK_TOKENS, CHUNK_OVERLAP, versión del chunking). Antes
de indexar se comparan con lo que ya hay en el vector store:
- mismo hash → no se toca (no se recalculan embeddings);
- hash distinto (cambió la guía, el modelo o los parámetros) → se borran los
  fragmentos de ESA guía y se indexa de nuevo;
- guía que ya no está en data/guias/ → se borran sus fragmentos.
Correrla dos veces seguidas no cambia nada (lo verifica tests/test_rag.py).

La API la corre sola al arrancar si INGESTA_AL_ARRANCAR=true (docker compose up).
"""

import argparse
import asyncio
import hashlib
import logging
import sys
import time

from pydantic import BaseModel, ConfigDict, Field

from rag.chunking import Tokenizador, fragmentar
from rag.documentos import DocumentoGuia, Fragmento
from rag.embeddings import Embeddings
from rag.vectorstore import VectorStore

log = logging.getLogger(__name__)

VERSION_CHUNKING: str = "v1"


class ReporteIngesta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store: str
    modelo_embeddings: str
    documentos: int
    fragmentos_totales: int = Field(description="Fragmentos en el índice al terminar.")
    indexados: list[str] = Field(default_factory=list, description="Guías indexadas (nuevas o con cambios).")
    sin_cambios: list[str] = Field(default_factory=list)
    borrados: list[str] = Field(default_factory=list, description="Guías que ya no están en el corpus.")
    fragmentos_nuevos: int = 0
    segundos: float = 0.0

    @property
    def hubo_cambios(self) -> bool:
        return bool(self.indexados or self.borrados)


def huella(modelo: str, chunk_tokens: int, overlap: int) -> str:
    return hashlib.sha256(f"{modelo}|{chunk_tokens}|{overlap}|{VERSION_CHUNKING}".encode()).hexdigest()[:8]


async def ingestar(corpus: list[DocumentoGuia], store: VectorStore, embeddings: Embeddings, tok: Tokenizador, *,
                   chunk_tokens: int, overlap: int, forzar: bool = False) -> ReporteIngesta:
    inicio = time.perf_counter()
    marca = huella(embeddings.nombre, chunk_tokens, overlap)
    actuales = await store.hashes()
    reporte = ReporteIngesta(store=store.nombre, modelo_embeddings=embeddings.nombre, documentos=len(corpus),
                             fragmentos_totales=0)
    for doc in corpus:
        # El tokenizer es CPU: fuera del event loop.
        fragmentos: list[Fragmento] = await asyncio.to_thread(fragmentar, doc, tok, chunk_tokens, overlap, marca)
        if not fragmentos:
            continue
        if not forzar and actuales.get(doc.slug) == fragmentos[0].doc_hash:
            reporte.sin_cambios.append(doc.slug)
            continue
        if doc.slug in actuales:
            await store.borrar_documento(doc.slug)
        vectores = await embeddings.documentos([f.para_embedding() for f in fragmentos])
        await store.agregar(fragmentos, vectores)
        reporte.indexados.append(doc.slug)
        reporte.fragmentos_nuevos += len(fragmentos)
        log.info("Ingesta: %s → %d fragmentos", doc.slug, len(fragmentos))
    for slug in sorted(set(actuales) - {d.slug for d in corpus}):
        await store.borrar_documento(slug)
        reporte.borrados.append(slug)
    reporte.fragmentos_totales = await store.contar()
    reporte.segundos = round(time.perf_counter() - inicio, 2)
    log.info("Ingesta lista: %d indexadas, %d sin cambios, %d borradas, %d fragmentos en el índice (%.1f s)",
             len(reporte.indexados), len(reporte.sin_cambios), len(reporte.borrados), reporte.fragmentos_totales,
             reporte.segundos)
    return reporte


async def _main() -> int:
    import config
    from rag.base import crear_base_conocimiento

    p = argparse.ArgumentParser(description="Ingesta idempotente de las guías (data/guias/) al vector store.")
    p.add_argument("--forzar", action="store_true", help="Reindexa todo aunque no haya cambios.")
    p.add_argument("--mock", action="store_true", help="Embeddings offline en una carpeta temporal.")
    args = p.parse_args()
    config.configurar_logs(logging.INFO)
    base = crear_base_conocimiento(modo_mock=args.mock or config.MODO_MOCK)
    try:
        reporte = await base.asegurar_indice(forzar=args.forzar)
    finally:
        await base.cerrar()
    print(reporte.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
