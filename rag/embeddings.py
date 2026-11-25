"""
rag/embeddings.py — Modelos de embeddings del RAG (el modelo se elige con EMBEDDINGS_MODELO).

- `EmbeddingsFastembed`: cualquier modelo de fastembed (ONNX, sin torch). Por
  defecto `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones, multilingüe,
  ~240 MB). El modelo se descarga una vez a MODELOS_CACHE_DIR.
- `EmbeddingsHash`: offline y determinístico (modo mock y tests, sin descargas).

Ventanas: el modelo multilingüe TRUNCA en 128 tokens, pero los fragmentos son de
~500. Para que el vector represente el fragmento entero, cada texto se parte en
ventanas de ≤ 120 tokens (con el mismo tokenizer), se embebe cada una y se
promedian (y se normaliza). Las consultas son cortas: un solo vector.

Async: ONNX Runtime es CPU y sync → cada llamada corre en asyncio.to_thread para
no bloquear el event loop de la API. Las dos clases exponen también el tokenizer
del modelo, que usa el chunking (rag/chunking.py) para contar tokens.
"""

import asyncio
import hashlib
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Protocol

from rag.chunking import Tokenizador, TokenizadorSimple
from rag.documentos import normalizar

log = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows sin modo desarrollador
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class Embeddings(Protocol):
    nombre: str
    dimension: int

    @property
    def tokenizador(self) -> Tokenizador: ...

    async def documentos(self, textos: list[str]) -> list[list[float]]: ...

    async def consulta(self, texto: str) -> list[float]: ...


def _normalizar_vector(v: list[float]) -> list[float]:
    norma = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norma for x in v]


# ---------------------------------------------------------------------------
# Real: fastembed
# ---------------------------------------------------------------------------
class _TokenizadorHF:
    """Adaptador del tokenizer de HuggingFace (tokenizers) al protocolo Tokenizador."""

    def __init__(self, tokenizer) -> None:  # noqa: ANN001 (tokenizers.Tokenizer)
        self._tok = tokenizer

    def offsets(self, texto: str) -> list[tuple[int, int]]:
        enc = self._tok.encode(texto, add_special_tokens=False)
        return [o for o in enc.offsets if o[1] > o[0]]


_MODELOS: dict[tuple[str, str], object] = {}
_CANDADO = threading.Lock()


def cargar_fastembed(clase: str, modelo: str, cache_dir: Path) -> object:
    """Un solo objeto por (clase, modelo) en todo el proceso (cargarlo tarda y ocupa memoria)."""
    with _CANDADO:
        clave = (clase, modelo)
        if clave not in _MODELOS:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if clase == "embedding":
                from fastembed import TextEmbedding

                _MODELOS[clave] = TextEmbedding(modelo, cache_dir=str(cache_dir))
            else:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _MODELOS[clave] = TextCrossEncoder(modelo, cache_dir=str(cache_dir))
            log.info("Modelo %s cargado: %s", clase, modelo)
        return _MODELOS[clave]


class EmbeddingsFastembed:
    TOKENS_POR_VENTANA: int = 120

    def __init__(self, modelo: str, cache_dir: Path) -> None:
        self.nombre = modelo
        self._modelo = cargar_fastembed("embedding", modelo, cache_dir)
        interno = self._modelo.model  # type: ignore[attr-defined]
        interno._ensure_tokenizer() if hasattr(interno, "_ensure_tokenizer") else None
        self._tokenizer = _TokenizadorHF(interno.tokenizer)
        self.dimension = len(next(iter(self._modelo.embed(["dimensión"]))))  # type: ignore[attr-defined]

    @property
    def tokenizador(self) -> Tokenizador:
        return self._tokenizer

    def _ventanas(self, texto: str) -> list[str]:
        spans = self._tokenizer.offsets(texto)
        n = self.TOKENS_POR_VENTANA
        if len(spans) <= n:
            return [texto]
        return [texto[spans[i][0]:spans[min(i + n, len(spans)) - 1][1]] for i in range(0, len(spans), n)]

    def _documentos_sync(self, textos: list[str]) -> list[list[float]]:
        partes: list[str] = []
        dueños: list[int] = []
        for i, t in enumerate(textos):
            for v in self._ventanas(t):
                partes.append(v)
                dueños.append(i)
        vectores = [list(map(float, v)) for v in self._modelo.embed(partes, batch_size=32)]  # type: ignore[attr-defined]
        suma: list[list[float]] = [[0.0] * self.dimension for _ in textos]
        for i, v in zip(dueños, vectores, strict=True):
            suma[i] = [a + b for a, b in zip(suma[i], v, strict=True)]
        return [_normalizar_vector(v) for v in suma]

    async def documentos(self, textos: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._documentos_sync, textos)

    async def consulta(self, texto: str) -> list[float]:
        vectores = await asyncio.to_thread(self._documentos_sync, [texto])
        return vectores[0]


# ---------------------------------------------------------------------------
# Mock: bolsa de palabras con hashing
# ---------------------------------------------------------------------------
class EmbeddingsHash:
    """Embedding "bolsa de palabras con hashing" (mismo criterio que la memoria de la PE6). Offline."""

    def __init__(self, dimension: int = 256) -> None:
        self.nombre = "hash-bolsa-de-palabras"
        self.dimension = dimension
        self._tokenizer = TokenizadorSimple()

    @property
    def tokenizador(self) -> Tokenizador:
        return self._tokenizer

    def _vector(self, texto: str) -> list[float]:
        v = [0.0] * self.dimension
        for palabra in re.findall(r"[a-z0-9]+", normalizar(texto)):
            if len(palabra) < 3:
                continue
            for token in {palabra, palabra[:5]}:
                v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension] += 1.0
        return _normalizar_vector(v)

    async def documentos(self, textos: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in textos]

    async def consulta(self, texto: str) -> list[float]:
        return self._vector(texto)
