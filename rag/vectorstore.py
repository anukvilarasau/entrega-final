"""
rag/vectorstore.py — El índice vectorial de las guías, detrás de una interfaz.

`VectorStore` es un protocolo async: el retriever, la ingesta y los agentes solo
conocen estos métodos. Cuál se usa lo decide VECTOR_STORE en el .env:

- "chroma" (por defecto): ChromaDB persistente en INDICE_DIR (data/indice_chroma).
  Corre dentro de Docker sin claves ni servicios externos. Su cliente local es
  SYNC: cada llamada va a asyncio.to_thread para no bloquear el event loop.
- "pinecone": Pinecone serverless con su cliente ASYNC nativo (PineconeAsyncio).
  Índice PINECONE_INDEX, namespace PINECONE_NAMESPACE; se crea solo si no existe.

Cambiar de uno a otro no toca los agentes: `crear_vector_store()` es el único lugar
que mira la variable. Los embeddings los calcula siempre rag/embeddings.py (los
stores reciben vectores ya hechos), así los dos guardan exactamente lo mismo.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

from rag.documentos import Fragmento

log = logging.getLogger(__name__)


class VectorStore(Protocol):
    nombre: str

    async def contar(self) -> int: ...

    async def hashes(self) -> dict[str, str]:
        """{slug: doc_hash} de lo indexado (para la ingesta idempotente)."""
        ...

    async def agregar(self, fragmentos: list[Fragmento], vectores: list[list[float]]) -> None: ...

    async def borrar_documento(self, slug: str) -> None: ...

    async def buscar(self, vector: list[float], k: int, slug: str | None = None) -> list[tuple[Fragmento, float]]:
        """Los k más parecidos (opcionalmente solo de un destino). Devuelve (fragmento, similitud coseno)."""
        ...

    async def todos(self, slug: str | None = None) -> list[Fragmento]:
        """Todos los fragmentos (para armar el índice BM25 sobre EXACTAMENTE lo mismo que está indexado)."""
        ...

    async def cerrar(self) -> None: ...


def _fragmento(id_: str, meta: dict[str, Any], texto: str | None = None) -> Fragmento:
    datos = {k: v for k, v in meta.items() if k in Fragmento.model_fields}
    if texto is not None:
        datos["texto"] = texto
    return Fragmento(id=id_, **datos)


# ---------------------------------------------------------------------------
# ChromaDB (local, persistente)
# ---------------------------------------------------------------------------
class ChromaVectorStore:
    def __init__(self, ruta: Path, coleccion: str) -> None:
        import chromadb

        ruta.mkdir(parents=True, exist_ok=True)
        self.nombre = f"chroma:{ruta.name}/{coleccion}"
        self._cliente = chromadb.PersistentClient(path=str(ruta))
        self._col = self._cliente.get_or_create_collection(
            name=coleccion, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
        )

    async def contar(self) -> int:
        return await asyncio.to_thread(self._col.count)

    def _hashes(self) -> dict[str, str]:
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return {str(m["slug"]): str(m["doc_hash"]) for m in metas}

    async def hashes(self) -> dict[str, str]:
        return await asyncio.to_thread(self._hashes)

    async def agregar(self, fragmentos: list[Fragmento], vectores: list[list[float]]) -> None:
        if not fragmentos:
            return
        await asyncio.to_thread(
            self._col.upsert, ids=[f.id for f in fragmentos], embeddings=vectores,  # type: ignore[arg-type]
            documents=[f.texto for f in fragmentos],
            metadatas=[{k: v for k, v in f.metadatos().items() if k != "texto"} for f in fragmentos],  # type: ignore[misc]
        )

    async def borrar_documento(self, slug: str) -> None:
        await asyncio.to_thread(self._col.delete, where={"slug": slug})

    def _buscar(self, vector: list[float], k: int, slug: str | None) -> list[tuple[Fragmento, float]]:
        if self._col.count() == 0:
            return []
        r = self._col.query(query_embeddings=[vector], n_results=k, where={"slug": slug} if slug else None,
                            include=["metadatas", "documents", "distances"])
        ids, metas, docs, dists = r["ids"][0], r["metadatas"][0], r["documents"][0], r["distances"][0]  # type: ignore[index]
        return [(_fragmento(i, dict(m), d), round(1 - float(dist), 4))
                for i, m, d, dist in zip(ids, metas, docs, dists, strict=True)]

    async def buscar(self, vector: list[float], k: int, slug: str | None = None) -> list[tuple[Fragmento, float]]:
        return await asyncio.to_thread(self._buscar, vector, k, slug)

    def _todos(self, slug: str | None) -> list[Fragmento]:
        r = self._col.get(where={"slug": slug} if slug else None, include=["metadatas", "documents"])
        frags = [_fragmento(i, dict(m), d) for i, m, d in zip(r["ids"], r["metadatas"] or [], r["documents"] or [],
                                                              strict=True)]
        return sorted(frags, key=lambda f: f.id)  # orden estable: BM25 desempata por posición

    async def todos(self, slug: str | None = None) -> list[Fragmento]:
        return await asyncio.to_thread(self._todos, slug)

    async def cerrar(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Pinecone (serverless, cliente async nativo)
# ---------------------------------------------------------------------------
class PineconeVectorStore:
    """Mismo contrato que Chroma, con PineconeAsyncio. `indice` se puede inyectar (tests con un índice falso)."""

    LOTE: int = 100

    def __init__(self, api_key: str, indice: str, namespace: str, dimension: int, cloud: str, region: str,
                 cliente_indice: Any = None) -> None:
        self.nombre = f"pinecone:{indice}/{namespace}"
        self._api_key, self._indice, self._ns = api_key, indice, namespace
        self._dimension, self._cloud, self._region = dimension, cloud, region
        self._idx = cliente_indice
        self._pc: Any = None
        self._candado = asyncio.Lock()

    async def _index(self) -> Any:
        async with self._candado:
            if self._idx is None:
                from pinecone import PineconeAsyncio, ServerlessSpec

                self._pc = PineconeAsyncio(api_key=self._api_key)
                if not await self._pc.has_index(self._indice):
                    log.info("Pinecone: creando el índice %s (%d dims, coseno)", self._indice, self._dimension)
                    await self._pc.create_index(name=self._indice, dimension=self._dimension, metric="cosine",
                                                spec=ServerlessSpec(cloud=self._cloud, region=self._region))
                host = (await self._pc.describe_index(self._indice)).host
                self._idx = self._pc.IndexAsyncio(host=host)
            return self._idx

    async def contar(self) -> int:
        stats = await (await self._index()).describe_index_stats()
        ns = (stats.get("namespaces") or {}).get(self._ns) or {}
        return int(ns.get("vector_count", 0))

    async def _ids(self, prefijo: str = "guia:") -> list[str]:
        idx = await self._index()
        ids: list[str] = []
        async for lote in idx.list(prefix=prefijo, namespace=self._ns):
            ids.extend(lote)
        return ids

    async def _traer(self, ids: list[str]) -> list[Fragmento]:
        idx = await self._index()
        salida: list[Fragmento] = []
        for i in range(0, len(ids), self.LOTE):
            r = await idx.fetch(ids=ids[i:i + self.LOTE], namespace=self._ns)
            for id_, v in r.vectors.items():
                salida.append(_fragmento(id_, dict(v.metadata)))
        return sorted(salida, key=lambda f: f.id)

    async def hashes(self) -> dict[str, str]:
        primeros = sorted({i.split("#")[0] + "#000" for i in await self._ids()})
        return {f.slug: f.doc_hash for f in await self._traer(primeros)}

    async def agregar(self, fragmentos: list[Fragmento], vectores: list[list[float]]) -> None:
        idx = await self._index()
        registros = [{"id": f.id, "values": v, "metadata": f.metadatos()} for f, v in zip(fragmentos, vectores, strict=True)]
        for i in range(0, len(registros), self.LOTE):
            await idx.upsert(vectors=registros[i:i + self.LOTE], namespace=self._ns)

    async def borrar_documento(self, slug: str) -> None:
        ids = await self._ids(f"guia:{slug}#")
        idx = await self._index()
        for i in range(0, len(ids), self.LOTE):
            await idx.delete(ids=ids[i:i + self.LOTE], namespace=self._ns)

    async def buscar(self, vector: list[float], k: int, slug: str | None = None) -> list[tuple[Fragmento, float]]:
        r = await (await self._index()).query(vector=vector, top_k=k, namespace=self._ns, include_metadata=True,
                                              filter={"slug": {"$eq": slug}} if slug else None)
        return [(_fragmento(m.id, dict(m.metadata)), round(float(m.score), 4)) for m in r.matches]

    async def todos(self, slug: str | None = None) -> list[Fragmento]:
        return await self._traer(await self._ids(f"guia:{slug}#" if slug else "guia:"))

    async def cerrar(self) -> None:
        for recurso in (self._idx, self._pc):
            if recurso is not None and hasattr(recurso, "close"):
                await recurso.close()
