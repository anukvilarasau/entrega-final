"""
rag/reranker.py — Re-ranking: el último filtro de relevancia del RAG (RERANKER en el .env).

La búsqueda híbrida trae candidatos "parecidos"; el re-ranker los vuelve a
ordenar mirando la consulta y el fragmento JUNTOS, que es mucho más preciso
que comparar dos vectores por separado. Se aplica solo a los mejores
RERANKER_CANDIDATOS de la fusión (es la etapa más cara).

- "cross_encoder" (por defecto): cross-encoder multilingüe en ONNX vía fastembed
  (RERANKER_MODELO, por defecto jinaai/jina-reranker-v2-base-multilingual,
  ~1,1 GB). Gratis, determinístico, sin red una vez descargado.
- "llm": Claude puntúa cada fragmento de 0 a 10 con salida estructurada Pydantic.
  Sin dependencias, aparece en LangSmith, pero gasta tokens.
- "ninguno": se queda el orden de la fusión RRF.
- mock (modo mock y tests): solapamiento de palabras, offline y determinístico.

Async: el cross-encoder es CPU sync → asyncio.to_thread; el LLM usa ainvoke.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from rag.documentos import Fragmento, normalizar

PALABRAS_VACIAS: frozenset[str] = frozenset(
    "a al algo algun alguna ante con como cual cuales cuando de del desde donde el en entre era es esa ese eso esta "
    "este esto estan hay la las le les lo los mas me mi muy no nos o otra otro para pero por que quien se si sin "
    "sobre son su sus tambien te tiene un una uno unos unas y ya hace hacer qué cómo cuál dónde".split()
)


def terminos(texto: str) -> list[str]:
    """Tokenización léxica (BM25 y re-ranker mock): sin tildes, sin palabras vacías, plural → singular simple."""
    salida = []
    for p in re.findall(r"[a-z0-9]+", normalizar(texto)):
        if len(p) < 2 or p in PALABRAS_VACIAS:
            continue
        if len(p) > 4 and p.endswith("es"):
            p = p[:-2]
        elif len(p) > 3 and p.endswith("s"):
            p = p[:-1]
        salida.append(p)
    return salida


class Reranker(Protocol):
    nombre: str

    async def puntuar(self, consulta: str, fragmentos: list[Fragmento]) -> list[float]:
        """Un puntaje por fragmento (más alto = más relevante), en el mismo orden."""
        ...


class SinReranker:
    nombre = "ninguno"

    async def puntuar(self, consulta: str, fragmentos: list[Fragmento]) -> list[float]:
        return [-float(i) for i in range(len(fragmentos))]  # conserva el orden de la fusión


class RerankerSolapamiento:
    """Mock: proporción de términos de la consulta que aparecen en el fragmento (+ la sección)."""

    nombre = "solapamiento-lexico"

    async def puntuar(self, consulta: str, fragmentos: list[Fragmento]) -> list[float]:
        q = set(terminos(consulta))
        if not q:
            return [0.0] * len(fragmentos)
        salida = []
        for f in fragmentos:
            texto, seccion = set(terminos(f.texto)), set(terminos(f.seccion))
            salida.append(round(len(q & texto) / len(q) + 0.5 * len(q & seccion) / len(q), 4))
        return salida


class RerankerCrossEncoder:
    """`max_caracteres`: cuánto de cada fragmento ve el modelo (la atención es cuadrática: es lo que más pesa en CPU)."""

    def __init__(self, modelo: str, cache_dir: Path, max_caracteres: int = 1200) -> None:
        from rag.embeddings import cargar_fastembed

        self.nombre = modelo
        self.max_caracteres = max_caracteres
        self._modelo: Any = cargar_fastembed("reranker", modelo, cache_dir)

    def _puntuar(self, consulta: str, textos: list[str]) -> list[float]:
        return [float(s) for s in self._modelo.rerank(consulta, textos, batch_size=8)]

    async def puntuar(self, consulta: str, fragmentos: list[Fragmento]) -> list[float]:
        if not fragmentos:
            return []
        return await asyncio.to_thread(self._puntuar, consulta,
                                       [f.para_embedding()[:self.max_caracteres] for f in fragmentos])


class PuntajeFragmento(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=1, description="Número del fragmento en la lista.")
    relevancia: int = Field(ge=0, le=10, description="0 = no sirve para la consulta, 10 = la responde de lleno.")


class PuntajesRelevancia(BaseModel):
    model_config = ConfigDict(extra="forbid")

    puntajes: list[PuntajeFragmento]


class RerankerLLM:
    """Claude como re-ranker (listwise): un solo llamado puntúa todos los candidatos."""

    LARGO: int = 700

    def __init__(self, llm: Any) -> None:
        self.nombre = "llm"
        self._llm = llm.with_structured_output(PuntajesRelevancia)

    async def puntuar(self, consulta: str, fragmentos: list[Fragmento]) -> list[float]:
        if not fragmentos:
            return []
        lista = "\n\n".join(f"[{i}] ({f.destino} — {f.seccion}) {f.texto[:self.LARGO]}"
                            for i, f in enumerate(fragmentos, 1))
        prompt = (f"Consulta: {consulta}\n\nPuntuá de 0 a 10 qué tanto sirve cada fragmento de guía de viaje para "
                  f"responder la consulta. Devolvé un puntaje por cada fragmento.\n\n{lista}")
        r = await self._llm.ainvoke([HumanMessage(content=prompt)])
        r = r if isinstance(r, PuntajesRelevancia) else PuntajesRelevancia.model_validate(r)
        por_n = {p.n: float(p.relevancia) for p in r.puntajes}
        return [por_n.get(i, 0.0) - i * 1e-3 for i in range(1, len(fragmentos) + 1)]  # empate → orden previo
