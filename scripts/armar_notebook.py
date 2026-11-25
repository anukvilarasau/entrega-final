"""
scripts/armar_notebook.py — Genera demo.ipynb (la demo de la API, para ejecutar con nbconvert).

    python scripts/armar_notebook.py
    python -m jupyter nbconvert --to notebook --execute --inplace demo.ipynb --ExecutePreprocessor.timeout=900

El notebook habla con la API levantada con `docker compose up` (http://localhost:8000), como cualquier
cliente: POST /tasks, polling, approve, seguimiento con thread_id y GET /threads. Queda guardado CON sus
salidas en el repo.
"""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


CELDAS = [
    md("""
# Demo — Asesor de viajes multi-agente (entrega final)

Este notebook usa la **API real** levantada con `docker compose up` (app + Redis 8), como lo haría cualquier
cliente. Muestra, en este orden:

1. `GET /health`: Redis, checkpointer, RAG y LangSmith.
2. `GET /rag/buscar`: el retriever híbrido solo (BM25 + vectorial + RRF + re-ranking), sin LLM.
3. Una pregunta **estable** que se resuelve solo con las guías (sin pausa ni web).
4. Un pedido con presupuesto: **pausa human-in-the-loop** → approve → respuesta con fuentes.
5. Un **seguimiento** en el mismo `thread_id` ("¿y si somos 4?") y el historial con `GET /threads/{id}`.

Cada job queda trazado en LangSmith (proyecto `entrega-final`, metadata `job_id` / `thread_id` / `turno`).
"""),
    code("""
import json, time
import httpx
from IPython.display import Markdown, display

API = "http://localhost:8000"
c = httpx.Client(base_url=API, timeout=30)

def esperar(job_id, estados=("DONE", "FAILED", "WAITING_APPROVAL"), limite=600):
    fin = time.time() + limite
    while time.time() < fin:
        job = c.get(f"/tasks/{job_id}").json()
        if job["estado"] in estados:
            return job
        time.sleep(2)
    raise TimeoutError(job_id)

def resumen(job):
    r = job.get("resultado") or {}
    m = r.get("metricas") or {}
    return {"estado": job["estado"], "turno": job["turno"], "tramos": job["tramos"], "ruta": r.get("ruta"),
            "permiso_web": r.get("permiso_web"),
            "fuentes (guía/memoria/web)": f"{m.get('fuentes_guia')}/{m.get('fuentes_memoria')}/{m.get('fuentes_web')}",
            "búsquedas web": m.get("busquedas_web"), "tiempo de grafo (s)": job.get("tiempo_ejecucion_s"),
            "tokens": job.get("total_tokens"), "costo (US$)": job.get("costo_usd")}
"""),
    md("## 1. Salud del servicio"),
    code("""
h = c.get("/health").json()
print(json.dumps({k: h[k] for k in ("status", "modo", "modelo", "redis", "checkpointer", "langsmith")}, indent=2, ensure_ascii=False))
print(json.dumps({k: h["rag"][k] for k in ("store", "fragmentos", "embeddings", "reranker", "destinos")}, indent=2, ensure_ascii=False))
"""),
    md("## 2. El RAG solo: `GET /rag/buscar` (filtro por destino desde los metadatos)"),
    code("""
r = c.get("/rag/buscar", params={"q": "qué hacer en invierno, esquí", "destino": "Ushuaia", "k": 3}).json()
print("guía:", r["slug"], "| candidatos vectoriales:", r["candidatos_vectoriales"], "| BM25:", r["candidatos_bm25"],
      "| re-ranker:", r["reranker"], "|", r["milisegundos"], "ms")
for x in r["fragmentos"]:
    f = x["fragmento"]
    print(f"\\n[{f['id']}] § {f['seccion']}  (puesto vectorial {x['puesto_vectorial']}, BM25 {x['puesto_bm25']}, "
          f"RRF {x['puntaje_rrf']:.4f}, re-rank {x['puntaje_rerank']})")
    print("   ", f["texto"][:220].replace("\\n", " "), "...")
"""),
    md("## 3. Una pregunta estable: solo guías, sin pausa ni búsquedas web"),
    code("""
t = c.post("/tasks", json={"pregunta": "¿Cómo llego del aeropuerto al centro de Mendoza y qué bodegas puedo visitar?"}).json()
job = esperar(t["job_id"])
print(json.dumps(resumen(job), indent=2, ensure_ascii=False))
display(Markdown(job["resultado"]["respuesta_final"]))
"""),
    md("## 4. Pedido con presupuesto: pausa human-in-the-loop antes de gastar en la web"),
    code("""
t = c.post("/tasks", json={"pregunta": "Quiero ir 4 días a Puerto Madryn en septiembre para ver ballenas, somos 2, "
                                        "salimos de Buenos Aires y tenemos 1.200.000 pesos. ¿Nos alcanza?"}).json()
pausa = esperar(t["job_id"])
print("estado:", pausa["estado"])
print(json.dumps(pausa.get("aprobacion_pendiente"), indent=2, ensure_ascii=False))
"""),
    code("""
if pausa["estado"] == "WAITING_APPROVAL":
    print(c.post(f"/tasks/{t['job_id']}/approve", json={"aprobado": True, "comentario": "demo del notebook"}).json())
job = esperar(t["job_id"], ("DONE", "FAILED"))
print(json.dumps(resumen(job), indent=2, ensure_ascii=False))
display(Markdown(job["resultado"]["respuesta_final"]))
thread_id = job["thread_id"]
"""),
    md("## 5. Seguimiento en la misma conversación (`thread_id`) e historial"),
    code("""
t2 = c.post("/tasks", json={"pregunta": "¿Y si en vez de 2 somos 4?", "thread_id": thread_id}).json()
print("turno:", t2["turno"], "| thread:", t2["thread_id"] == thread_id)
job2 = esperar(t2["job_id"])
if job2["estado"] == "WAITING_APPROVAL":
    c.post(f"/tasks/{t2['job_id']}/approve", json={"aprobado": True})
    job2 = esperar(t2["job_id"], ("DONE", "FAILED"))
print(json.dumps(resumen(job2), indent=2, ensure_ascii=False))
print("pedido del turno 2:", json.dumps(job2["resultado"]["pedido"], ensure_ascii=False))
display(Markdown(job2["resultado"]["respuesta_final"]))
"""),
    code("""
conv = c.get(f"/threads/{thread_id}").json()
print(conv["turnos_completos"], "turnos ·", conv["checkpointer"], "· jobs:", conv["jobs"])
for t in conv["turnos"]:
    print(f"  turno {t['turno']}: {t['pregunta']}  →  {t['pedido'].get('personas')} personas, {t['pedido'].get('destino')}")
"""),
]

nb = {"cells": CELDAS, "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                                                   "name": "python3"},
                                    "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(RAIZ / "demo.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("demo.ipynb listo")
