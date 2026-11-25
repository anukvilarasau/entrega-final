"""
rag/evaluate.py — Evaluación del retriever con un golden set: Recall@k, Precision@k y MRR.

    python -m rag.evaluate                       # configuración del .env; guarda docs/evaluacion_rag.{md,json}
    python -m rag.evaluate --mock                # embeddings offline (sirve para probar el script, no para medir)
    python -m rag.evaluate --reranker ninguno --salida sin_reranker
    python -m rag.evaluate --reranker-modelo BAAI/bge-reranker-v2-m3-int8 --salida bge

Golden set: data/golden_set.json (15 preguntas). Cada una trae su documento
esperado (la guía) y qué fragmentos cuentan como relevantes (por sección o por
palabras clave). Se miden DOS niveles, con k = 1 y k = RAG_TOP_K:

1. Documento (SIN filtro de destino, sobre las 9 guías):
   - Recall@k    = ¿aparece la guía esperada entre los k primeros? (hay una sola guía relevante)
   - Precision@k = fracción de los k fragmentos que son de la guía esperada
2. Fragmento (CON el filtro de destino que usa el agente):
   - Recall@k    = relevantes recuperados / relevantes que existen en el índice (topeado a k)
   - Precision@k = relevantes entre los k recuperados / k
   - MRR         = 1 / puesto del primer fragmento relevante

Y cuatro modos, para ver qué aporta cada etapa: vectorial, BM25, híbrido (RRF)
e híbrido + re-ranker. También se mide la latencia media por consulta.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

import config
from rag.base import BaseConocimiento, crear_base_conocimiento
from rag.documentos import Fragmento, normalizar
from rag.reranker import RerankerCrossEncoder, RerankerLLM, SinReranker

GOLDEN_SET: Path = config.DATA_DIR / "golden_set.json"
SALIDA_DIR: Path = config.BASE_DIR / "docs"


class PreguntaGolden(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pregunta: str
    destino: str
    slug: str
    secciones: list[str] = Field(default_factory=list)
    palabras: list[str] = Field(default_factory=list)

    def es_relevante(self, f: Fragmento) -> bool:
        if f.slug != self.slug:
            return False
        texto = normalizar(f.texto)
        return (any(s.lower() in f.seccion.lower() for s in self.secciones)
                or any(normalizar(p) in texto for p in self.palabras))


class Metricas(BaseModel):
    modo: str
    recall_doc_1: float
    precision_doc_1: float
    recall_doc_k: float
    precision_doc_k: float
    recall_frag_1: float
    precision_frag_1: float
    recall_frag_k: float
    precision_frag_k: float
    mrr: float
    ms_por_consulta: float


MODOS: dict[str, tuple[str, bool]] = {
    "vectorial": ("vectorial", False),
    "bm25": ("bm25", False),
    "hibrido_rrf": ("hibrido", False),
    "hibrido_rrf_rerank": ("hibrido", True),
}


async def evaluar_modo(base: BaseConocimiento, preguntas: list[PreguntaGolden], nombre: str, k: int,
                       todos: list[Fragmento]) -> tuple[Metricas, list[dict]]:
    modo, rerank = MODOS[nombre]
    r = base.retriever
    filas: list[dict] = []
    suma = {c: 0.0 for c in Metricas.model_fields if c not in ("modo", "ms_por_consulta")}
    tiempo = 0.0
    for q in preguntas:
        inicio = time.perf_counter()
        sin_filtro = [x.fragmento for x in await r.buscar(q.pregunta, slug=None, k=k, modo=modo, usar_reranker=rerank)]
        con_filtro = [x.fragmento for x in await r.buscar(q.pregunta, slug=q.slug, k=k, modo=modo,
                                                           usar_reranker=rerank)]
        tiempo += time.perf_counter() - inicio
        relevantes_total = sum(1 for f in todos if q.es_relevante(f))
        rel = [q.es_relevante(f) for f in con_filtro]
        de_la_guia = [f.slug == q.slug for f in sin_filtro]
        primer = next((i for i, ok in enumerate(rel, 1) if ok), None)
        valores = {
            "recall_doc_1": float(any(de_la_guia[:1])), "precision_doc_1": float(sum(de_la_guia[:1])),
            "recall_doc_k": float(any(de_la_guia)), "precision_doc_k": sum(de_la_guia) / k,
            "recall_frag_1": (sum(rel[:1]) / min(1, relevantes_total)) if relevantes_total else 0.0,
            "precision_frag_1": float(sum(rel[:1])),
            "recall_frag_k": (sum(rel) / min(k, relevantes_total)) if relevantes_total else 0.0,
            "precision_frag_k": sum(rel) / k,
            "mrr": 1 / primer if primer else 0.0,
        }
        for c, v in valores.items():
            suma[c] += v
        filas.append({"id": q.id, "relevantes_en_indice": relevantes_total, **{c: round(v, 3) for c, v in valores.items()},
                      "top_con_filtro": [f"{f.id} ({f.seccion})" for f in con_filtro]})
    n = len(preguntas)
    return Metricas(modo=nombre, ms_por_consulta=round(tiempo / (2 * n) * 1000, 1),
                    **{c: round(v / n, 3) for c, v in suma.items()}), filas


def tabla_md(metricas: list[Metricas], k: int) -> str:
    filas = [f"| Modo | Recall@1 doc | Recall@{k} doc | Precision@{k} doc | Recall@1 frag | Recall@{k} frag | "
             f"Precision@{k} frag | MRR | ms/consulta |", "|---|---|---|---|---|---|---|---|---|"]
    for m in metricas:
        filas.append(f"| {m.modo} | {m.recall_doc_1:.2f} | {m.recall_doc_k:.2f} | {m.precision_doc_k:.2f} | "
                     f"{m.recall_frag_1:.2f} | {m.recall_frag_k:.2f} | {m.precision_frag_k:.2f} | {m.mrr:.2f} | "
                     f"{m.ms_por_consulta:.0f} |")
    return "\n".join(filas)


async def _main() -> int:
    p = argparse.ArgumentParser(description="Evaluación del RAG (Recall@k / Precision@k / MRR).")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--reranker", choices=["cross_encoder", "llm", "ninguno"], default=config.RERANKER)
    p.add_argument("--reranker-modelo", default=config.RERANKER_MODELO)
    p.add_argument("--candidatos-rerank", type=int, default=config.RERANKER_CANDIDATOS)
    p.add_argument("--max-caracteres", type=int, default=config.RERANKER_MAX_CARACTERES)
    p.add_argument("-k", type=int, default=config.RAG_TOP_K)
    p.add_argument("--salida", default="evaluacion_rag", help="Nombre base de los archivos en docs/.")
    args = p.parse_args()
    config.configurar_logs(logging.WARNING)

    preguntas = [PreguntaGolden.model_validate(x) for x in json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["preguntas"]]
    base = crear_base_conocimiento(modo_mock=args.mock)
    await base.asegurar_indice()
    if not args.mock:
        if args.reranker == "cross_encoder":
            base.retriever.reranker = RerankerCrossEncoder(args.reranker_modelo, config.MODELOS_CACHE_DIR,
                                                           args.max_caracteres)
        elif args.reranker == "llm":
            from agents.llm import crear_llm_real

            base.retriever.reranker = RerankerLLM(crear_llm_real())
        else:
            base.retriever.reranker = SinReranker()
    base.retriever.candidatos_rerank = args.candidatos_rerank
    todos = await base.store.todos()

    metricas, detalle = [], {}
    for nombre in MODOS:
        m, filas = await evaluar_modo(base, preguntas, nombre, args.k, todos)
        metricas.append(m)
        detalle[nombre] = filas
        print(f"{nombre:20s} R@{args.k} frag {m.recall_frag_k:.2f} · P@{args.k} frag {m.precision_frag_k:.2f} · "
              f"MRR {m.mrr:.2f} · R@{args.k} doc {m.recall_doc_k:.2f} · {m.ms_por_consulta:.0f} ms")
    await base.cerrar()

    configuracion = {
        "modo": "mock" if args.mock else "real", "embeddings": base.embeddings.nombre, "store": base.store.nombre,
        "reranker": base.retriever.reranker.nombre, "candidatos_rerank": args.candidatos_rerank,
        "max_caracteres_rerank": args.max_caracteres, "k": args.k, "candidatos_por_buscador": config.RAG_CANDIDATOS,
        "rrf_k": config.RAG_RRF_K, "chunk_tokens": config.CHUNK_TOKENS, "chunk_overlap": config.CHUNK_OVERLAP,
        "fragmentos_en_indice": len(todos), "preguntas": len(preguntas), "fecha": time.strftime("%Y-%m-%d %H:%M"),
    }
    SALIDA_DIR.mkdir(exist_ok=True)
    (SALIDA_DIR / f"{args.salida}.json").write_text(json.dumps(
        {"configuracion": configuracion, "metricas": [m.model_dump() for m in metricas], "detalle": detalle},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (SALIDA_DIR / f"{args.salida}.md").write_text(
        f"# Evaluación del RAG ({configuracion['fecha']})\n\n"
        + "\n".join(f"- **{c}**: `{v}`" for c, v in configuracion.items()) + "\n\n" + tabla_md(metricas, args.k) + "\n",
        encoding="utf-8")
    print(f"\nGuardado en docs/{args.salida}.md y .json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
