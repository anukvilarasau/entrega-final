"""
scripts/pruebas_finales.py — Las 5 pruebas de punta a punta contra la API REAL (docker compose up).

    docker compose up -d                       # API + Redis, modo real (claves en el .env)
    python scripts/pruebas_finales.py          # corre las 5 en orden
    python scripts/pruebas_finales.py --solo 1 # una sola
    python scripts/pruebas_finales.py --tabla  # no corre nada: rearma la tabla con lo guardado

Escenarios (cada uno deja su job completo en traces/pruebas_finales/<n>_<nombre>.json):
  1. solo_guias        — pregunta estable: se resuelve con el RAG, sin pausa HITL ni web ni analista.
  2. flujo_completo    — pedido con presupuesto: pausa HITL por los precios → approve → investigador
                         (guías + web) → analista → síntesis.
  3. seguimiento       — `docker compose restart app` y "¿y si vamos en julio?" en el thread de la 2:
                         el checkpoint en Redis recuerda el pedido; la memoria (volumen) tiene los precios.
  4. no_alcanza        — presupuesto insuficiente: el analista lo detecta y calcula una alternativa.
  5. error_controlado  — fechas imposibles + la persona RECHAZA la web: validar_viaje da INVÁLIDO, el grafo
                         termina igual (DONE) y pide corregir. Además, errores HTTP controlados (404/409/422).

El cliente es httpx async (como un cliente real: POST, polling, approve). Al final pide a LangSmith el
run raíz de cada job (por metadata job_id) para guardar su id, latencia y costo según LangSmith.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import config  # noqa: E402

CARPETA = RAIZ / "traces" / "pruebas_finales"
BASE = "http://localhost:8000"
DOCKER = "docker"

ESCENARIOS: dict[int, dict[str, Any]] = {
    1: {"nombre": "solo_guias", "pregunta": "¿Qué se puede hacer en Salta en julio y qué clima hace? Pensamos ir 4 días.",
        "aprobar": None, "demuestra": "RAG sin pausa: lo estable sale de las guías (0 búsquedas web, sin analista)"},
    2: {"nombre": "flujo_completo",
        "pregunta": ("Quiero ir 5 días a Ushuaia en marzo, somos 2, salimos de Buenos Aires y tenemos 2.000.000 de "
                     "pesos. ¿Me armás el viaje y me decís si nos alcanza?"),
        "aprobar": True, "demuestra": "Flujo completo: guías + pausa HITL + web + analista + síntesis"},
    3: {"nombre": "seguimiento_tras_reinicio", "pregunta": "¿Y si vamos en julio, en temporada de esquí?",
        "aprobar": True, "seguimiento_de": 2, "reiniciar": True,
        "demuestra": "Follow-up después de `docker compose restart app`: recuerda el pedido (checkpoint en Redis)"},
    4: {"nombre": "no_alcanza",
        "pregunta": ("Queremos ir 7 días a El Calafate en febrero, somos 2, salimos desde Córdoba y tenemos "
                     "350.000 pesos. ¿Nos alcanza?"),
        "aprobar": True, "demuestra": "Presupuesto insuficiente: el analista lo detecta y propone una alternativa"},
    5: {"nombre": "error_controlado",
        "pregunta": ("Quiero ir a Mendoza del 20 al 12 de marzo, somos 3, salimos de Rosario y tenemos 900.000 "
                     "pesos. ¿Nos alcanza?"),
        "aprobar": False, "demuestra": "Error controlado: fechas imposibles + web rechazada → DONE pidiendo corregir"},
}


async def esperar(c: httpx.AsyncClient, job_id: str, estados: set[str], limite_s: float = 600) -> dict[str, Any]:
    fin = time.monotonic() + limite_s
    while time.monotonic() < fin:
        job = (await c.get(f"/tasks/{job_id}")).json()
        if job["estado"] in estados:
            return job
        await asyncio.sleep(1.5)
    raise TimeoutError(f"el job {job_id} no llegó a {estados}")


async def reiniciar_app() -> float:
    """docker compose restart app y espera a /health. Devuelve los segundos que tardó."""
    inicio = time.monotonic()
    proc = await asyncio.create_subprocess_exec(DOCKER, "compose", "restart", "app", cwd=RAIZ,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    async with httpx.AsyncClient(base_url=BASE, timeout=5) as c:
        for _ in range(120):
            try:
                if (await c.get("/health")).status_code == 200:
                    return round(time.monotonic() - inicio, 1)
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError("la API no volvió después del reinicio")


async def errores_http(c: httpx.AsyncClient) -> dict[str, int]:
    """Errores controlados de la API (sin costo): cada uno devuelve el código esperado y un mensaje claro."""
    return {
        "GET job inexistente (404)": (await c.get("/tasks/" + "0" * 32)).status_code,
        "seguimiento de thread inexistente (404)": (await c.post("/tasks", json={"pregunta": "¿Y en julio?",
                                                                                 "thread_id": "0" * 32})).status_code,
        "pregunta demasiado corta (422)": (await c.post("/tasks", json={"pregunta": "hola"})).status_code,
        "campo desconocido (422)": (await c.post("/tasks", json={"pregunta": "¿Qué hacer en Salta?", "x": 1})).status_code,
    }


async def correr(n: int) -> dict[str, Any]:
    e = ESCENARIOS[n]
    salida: dict[str, Any] = {"n": n, **e}
    if e.get("reiniciar"):
        salida["reinicio_s"] = await reiniciar_app()
        print(f"  reinicio de la app: {salida['reinicio_s']} s")
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as c:
        cuerpo: dict[str, Any] = {"pregunta": e["pregunta"]}
        if (previo := e.get("seguimiento_de")) is not None:
            anterior = json.loads(next(CARPETA.glob(f"{previo}_*.json")).read_text(encoding="utf-8"))
            cuerpo["thread_id"] = anterior["job"]["thread_id"]
        creado = (await c.post("/tasks", json=cuerpo)).json()
        print(f"  job {creado['job_id'][:8]} · thread {creado['thread_id'][:8]} · turno {creado['turno']}")
        job = await esperar(c, creado["job_id"], {"WAITING_APPROVAL", "DONE", "FAILED"})
        if job["estado"] == "WAITING_APPROVAL":
            salida["pausa"] = job["aprobacion_pendiente"]
            print(f"  pausa HITL: web para {job['aprobacion_pendiente']['temas_sin_memoria']} "
                  f"(guías: {job['aprobacion_pendiente'].get('temas_en_guias')}, "
                  f"memoria: {job['aprobacion_pendiente']['temas_en_memoria']}) → aprobado={e['aprobar']}")
            await c.post(f"/tasks/{creado['job_id']}/approve",
                         json={"aprobado": bool(e["aprobar"]),
                               "comentario": "aprobado en la prueba final" if e["aprobar"] else "sin gastar en la web"})
            job = await esperar(c, creado["job_id"], {"DONE", "FAILED"})
        if n == 5:
            salida["errores_http"] = await errores_http(c)
        salida["conversacion"] = (await c.get(f"/threads/{job['thread_id']}")).json() if e.get("seguimiento_de") else None
    salida["job"] = job
    return salida


def fila(s: dict[str, Any]) -> dict[str, Any]:
    j = s["job"]
    r = j.get("resultado") or {}
    m = r.get("metricas") or {}
    ls = s.get("langsmith") or {}
    return {
        "n": s["n"], "escenario": s["nombre"], "estado": j["estado"], "turno": j.get("turno"),
        "pausa_hitl": "sí" if s.get("pausa") else "no", "permiso_web": r.get("permiso_web"),
        "ruta": r.get("ruta"), "fuentes": f"{m.get('fuentes_guia', 0)} / {m.get('fuentes_memoria', 0)} / {m.get('fuentes_web', 0)}",
        "busquedas_web": m.get("busquedas_web"), "presupuesto_entra": m.get("presupuesto_entra"),
        "validacion": m.get("validacion"), "tiempo_ejecucion_s": j.get("tiempo_ejecucion_s"),
        "duracion_s": j.get("duracion_s"), "tokens": j.get("total_tokens"), "costo_usd": j.get("costo_usd"),
        "langsmith_costo_usd": ls.get("costo_usd"), "langsmith_latencia_s": ls.get("latencia_s"),
        "langsmith_run": ls.get("run_id"),
    }


def tabla(salidas: list[dict[str, Any]]) -> str:
    filas = ["| # | Escenario | Estado | Turno | Pausa HITL | Permiso web | Fuentes guía / memoria / web | Búsquedas web | "
             "Entra en el presupuesto | Validación | Tiempo de grafo (s) | Tokens | Costo (US$) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in salidas:
        f = fila(s)
        entra = {True: "sí", False: "**no**", None: "—"}[f["presupuesto_entra"]]
        filas.append(f"| {f['n']} | [{f['escenario']}](traces/pruebas_finales/{s['archivo']}) | {f['estado']} | "
                     f"{f['turno']} | {f['pausa_hitl']} | {f['permiso_web']} | {f['fuentes']} | {f['busquedas_web']} | "
                     f"{entra} | {f['validacion']} | {f['tiempo_ejecucion_s']} | {f['tokens']} | {f['costo_usd']} |")
    return "\n".join(filas)


async def langsmith_de(job_id: str) -> dict[str, Any]:
    """Los runs raíz del job en LangSmith (por metadata job_id): 1 tramo, o 2 si hubo pausa HITL.

    Usa la API nueva del SDK (`client.runs.query_v2`, async): `list_runs()` está deprecado y deja de
    funcionar el 31/01/2027 (es el aviso que mostraba LangSmith con el SDK viejo).
    """
    from langsmith import Client

    from observability.monitoreo import configurar_langsmith

    configurar_langsmith(False)
    cliente = Client()
    proyecto = await asyncio.to_thread(cliente.read_project, project_name=config.AJUSTES.langsmith_project)
    runs = [r async for r in cliente.runs.query_v2(
        project_ids=[str(proyecto.id)], is_root=True, page_size=10,
        filter=f'and(eq(metadata_key, "job_id"), eq(metadata_value, "{job_id}"))',
        selects=["ID", "TRACE_ID", "START_TIME", "LATENCY_SECONDS", "TOTAL_TOKENS", "TOTAL_COST", "STATUS", "TAGS"])]
    if not runs:
        return {}
    runs.sort(key=lambda r: r.start_time)
    return {
        "run_ids": [str(r.id) for r in runs], "run_id": str(runs[0].id), "tramos": len(runs),
        "latencia_s": round(sum(float(r.latency_seconds or 0) for r in runs), 2),
        "costo_usd": round(sum(float(r.total_cost or 0) for r in runs), 6),
        "tokens": sum(int(r.total_tokens or 0) for r in runs),
        "estados": [r.status for r in runs],
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--solo", type=int, nargs="+", choices=list(ESCENARIOS))
    p.add_argument("--tabla", action="store_true")
    args = p.parse_args()
    CARPETA.mkdir(parents=True, exist_ok=True)
    if not args.tabla:
        for n in args.solo or list(ESCENARIOS):
            print(f"[{n}] {ESCENARIOS[n]['nombre']}: {ESCENARIOS[n]['demuestra']}")
            s = await correr(n)
            for viejo in CARPETA.glob(f"{n}_*.json"):
                viejo.unlink()
            s["archivo"] = f"{n}_{s['nombre']}.json"
            (CARPETA / s["archivo"]).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
            print("  →", json.dumps({k: v for k, v in fila(s).items() if k in ("estado", "ruta", "fuentes",
                                                                                "busquedas_web", "costo_usd")},
                                    ensure_ascii=False))
        await asyncio.sleep(5)  # que LangSmith termine de recibir los runs
    salidas = []
    for archivo in sorted(CARPETA.glob("[0-9]_*.json")):
        s = json.loads(archivo.read_text(encoding="utf-8"))
        if not s.get("langsmith"):
            try:
                s["langsmith"] = await langsmith_de(s["job"]["job_id"])
                archivo.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:  # noqa: BLE001 — sin LangSmith la tabla sale igual
                print(f"  (LangSmith no disponible para {archivo.name}: {e})")
        salidas.append(s)
    md = ["# Pruebas finales de punta a punta (API real en Docker)", "", tabla(salidas), "",
          "Fuentes: cantidad de fuentes de cada origen que citó el investigador. Tiempo de grafo: solo ejecución "
          "(sin la espera humana). Costo: tokens × precios de observability/monitoreo.py.", ""]
    for s in salidas:
        ls = s.get("langsmith") or {}
        md.append(f"- **{s['n']}. {s['nombre']}** — {s['demuestra']}. LangSmith: run `{ls.get('run_id')}`, "
                  f"{ls.get('tramos')} tramo(s), latencia {ls.get('latencia_s')} s, costo US$ {ls.get('costo_usd')}."
                  + (f" Reinicio de la app: {s['reinicio_s']} s." if s.get("reinicio_s") else "")
                  + (f" Errores HTTP: {s['errores_http']}." if s.get("errores_http") else ""))
    (CARPETA / "resultados.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + tabla(salidas))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
