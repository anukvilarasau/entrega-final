"""
scripts/descargar_modelos.py — Descarga los modelos ONNX que usa el sistema (lo corre el Dockerfile).

    python scripts/descargar_modelos.py

Descarga (a MODELOS_CACHE_DIR) el modelo de embeddings del RAG (EMBEDDINGS_MODELO),
el re-ranker (RERANKER_MODELO, si RERANKER=cross_encoder) y el embedding por defecto
de ChromaDB que usa la memoria de búsquedas. Es independiente del resto del código
a propósito: en el Dockerfile corre ANTES de copiar la app, así la capa de modelos
queda en caché aunque cambie el código. Los valores por defecto son los de config.py.
"""

import os
from pathlib import Path

POR_DEFECTO = {
    "EMBEDDINGS_MODELO": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "RERANKER_MODELO": "jinaai/jina-reranker-v2-base-multilingual",
    "RERANKER": "cross_encoder",
}


def valor(nombre: str) -> str:
    return (os.getenv(nombre) or "").strip() or POR_DEFECTO[nombre]


def main() -> None:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    cache = Path(os.getenv("MODELOS_CACHE_DIR") or Path.home() / ".cache" / "asesor-viajes" / "modelos")
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Embeddings: {valor('EMBEDDINGS_MODELO')}")
    list(TextEmbedding(valor("EMBEDDINGS_MODELO"), cache_dir=str(cache)).embed(["prueba"]))
    if valor("RERANKER") == "cross_encoder":
        print(f"Re-ranker: {valor('RERANKER_MODELO')}")
        list(TextCrossEncoder(valor("RERANKER_MODELO"), cache_dir=str(cache)).rerank("prueba", ["prueba"]))
    print("Embedding de la memoria: all-MiniLM-L6-v2 (default de ChromaDB)")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    DefaultEmbeddingFunction()(["prueba"])
    print("Modelos listos en", cache)


if __name__ == "__main__":
    main()
