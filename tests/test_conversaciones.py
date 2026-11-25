"""Tests de conversaciones de varios turnos (follow-ups) y de persistencia ante un reinicio de la API.

- POST /tasks con el thread_id de una conversación → turno 2 sobre el MISMO checkpoint;
- el pedido del turno anterior se arrastra ("¿y si vamos en julio?" mantiene destino, personas y presupuesto);
- los aportes de un turno no se mezclan con los del siguiente (filtro por turno);
- un "reinicio" (worker, store y grafo nuevos sobre el mismo Redis + checkpointer) recuerda la conversación;
- 404 / 409 / 422 y GET /threads/{id};
- con REDIS_URL_TESTS (Redis 8): reinicio REAL con AsyncRedisSaver (conexión nueva entre turnos).
"""

import os
import time
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import fakeredis
import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

import config
from agents.supervisor import combinar_pedidos
from api.main import Ajustes, create_app
from api.worker import EstadoJob, JobStore, Worker
from graph.builder import construir_grafo_api, crear_componentes
from graph.checkpointer import abrir_checkpointer
from graph.state import PedidoViaje, entrada_turno
from tests.test_api import DEMO, esperar

JULIO = "¿Y si vamos en julio?"


@pytest.fixture(autouse=True)
def hoy_fijo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "FECHA_REFERENCIA", date(2026, 11, 5))
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[SimpleNamespace]:
    comp = crear_componentes(modo_mock=True, ruta_memoria=tmp_path / "chroma")
    app = create_app(Ajustes(modo_mock=True, redis_url="", timeout_s=30), componentes=comp)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield SimpleNamespace(app=app, c=c, comp=comp)


async def primer_turno(c: httpx.AsyncClient) -> dict:
    creado = (await c.post("/tasks", json={"pregunta": DEMO})).json()
    assert creado["turno"] == 1 and creado["thread_id"] == creado["job_id"]
    await esperar(c, creado["job_id"], {"WAITING_APPROVAL"})
    await c.post(f"/tasks/{creado['job_id']}/approve", json={"aprobado": True})
    final = await esperar(c, creado["job_id"], {"DONE", "FAILED"})
    assert final["estado"] == "DONE", final["error"]
    return final


# ---------------------------------------------------------------------------
# Unidad: pedido del turno anterior y entrada de un turno
# ---------------------------------------------------------------------------
def test_combinar_pedidos_mantiene_lo_que_no_cambio() -> None:
    anterior = PedidoViaje(origen="Mendoza", destino="Bariloche", fecha_ida=date(2027, 1, 10),
                           fecha_vuelta=date(2027, 1, 16), dias=7, personas=2, presupuesto=800_000, moneda="ARS",
                           supuestos=["enero sin día exacto: se asume del 10/01/2027 al 16/01/2027"])
    nuevo = PedidoViaje(fecha_ida=date(2027, 7, 10), supuestos=["julio sin día exacto: se asume del 10/07/2027"])
    p = combinar_pedidos(anterior, nuevo)
    assert (p.origen, p.destino, p.personas, p.presupuesto, p.moneda, p.dias) == ("Mendoza", "Bariloche", 2, 800_000,
                                                                                  "ARS", 7)
    assert p.fecha_ida == date(2027, 7, 10) and p.fecha_vuelta == date(2027, 7, 16)  # la vuelta se recalcula
    assert p.supuestos == ["julio sin día exacto: se asume del 10/07/2027"]
    # si el cambio es solo de personas, las fechas quedan como estaban
    q = combinar_pedidos(anterior, PedidoViaje(personas=4))
    assert q.personas == 4 and q.fecha_vuelta == date(2027, 1, 16)


def test_la_entrada_de_un_seguimiento_no_borra_el_historial() -> None:
    uno, dos = entrada_turno("hola mundo", turno=1), entrada_turno("y en julio", turno=2)
    assert uno["aportes"] == [] and uno["turnos"] == [] and uno["turno"] == 1
    assert "aportes" not in dos and "turnos" not in dos and "decisiones" not in dos  # los reductores suman
    assert dos["turno"] == 2 and dos["pedido"] is None and dos["pasos_supervisor"] == 0


# ---------------------------------------------------------------------------
# API: seguimiento en el mismo thread
# ---------------------------------------------------------------------------
async def test_seguimiento_en_el_mismo_thread_recuerda_el_pedido(api: SimpleNamespace) -> None:
    t1 = await primer_turno(api.c)
    thread = t1["thread_id"]
    assert t1["resultado"]["pedido"]["fecha_ida"] == "2027-01-10"

    r = await api.c.post("/tasks", json={"pregunta": JULIO, "thread_id": thread})
    assert r.status_code == 202 and r.json()["turno"] == 2 and r.json()["thread_id"] == thread
    t2 = await esperar(api.c, r.json()["job_id"], {"DONE", "FAILED", "WAITING_APPROVAL"})
    # los precios del turno 1 quedaron en la memoria y lo estable sale de las guías: no hace falta pausar
    assert t2["estado"] == "DONE" and t2["tramos"] == 1, t2.get("error")
    pedido = t2["resultado"]["pedido"]
    assert (pedido["destino"], pedido["origen"], pedido["personas"], pedido["presupuesto"]) == \
        ("Bariloche", "Mendoza", 2, 800000.0)
    assert pedido["fecha_ida"] == "2027-07-10" and pedido["fecha_vuelta"] == "2027-07-16"
    assert t2["resultado"]["turno"] == 2 and t2["resultado"]["permiso_web"] == "sin_pedir"
    assert t2["resultado"]["metricas"]["busquedas_web"] == 0 and t2["resultado"]["metricas"]["turno"] == 2
    assert "## Itinerario" in t2["resultado"]["respuesta_final"]

    conv = (await api.c.get(f"/threads/{thread}")).json()
    assert conv["turnos_completos"] == 2 and [t["turno"] for t in conv["turnos"]] == [1, 2]
    assert conv["turnos"][1]["pregunta"] == JULIO and conv["jobs"] == [t1["job_id"], t2["job_id"]]
    assert conv["job_activo"] is None

    # en el checkpoint conviven los dos turnos, cada aporte etiquetado con el suyo
    snap = await api.app.state.worker.estado_thread(thread)
    assert {a.turno for a in snap.values["aportes"]} == {1, 2} and snap.values["turno"] == 2


async def test_errores_de_conversacion(api: SimpleNamespace) -> None:
    otro = "0" * 32
    r = await api.c.post("/tasks", json={"pregunta": JULIO, "thread_id": otro})
    assert r.status_code == 404 and "No existe la conversación" in r.json()["detail"]
    assert (await api.c.post("/tasks", json={"pregunta": JULIO, "thread_id": "no-es-un-id"})).status_code == 422
    assert (await api.c.get(f"/threads/{otro}")).status_code == 404

    creado = (await api.c.post("/tasks", json={"pregunta": DEMO})).json()
    await esperar(api.c, creado["job_id"], {"WAITING_APPROVAL"})
    r = await api.c.post("/tasks", json={"pregunta": JULIO, "thread_id": creado["thread_id"]})
    assert r.status_code == 409 and creado["job_id"] in r.json()["detail"]  # el turno 1 sigue pausado
    assert (await api.c.get(f"/threads/{creado['thread_id']}")).json()["job_activo"] == creado["job_id"]


async def test_rag_buscar_y_health(api: SimpleNamespace) -> None:
    r = (await api.c.get("/rag/buscar", params={"q": "dónde esquiar", "destino": "Bariloche", "k": 3})).json()
    assert r["slug"] == "bariloche" and len(r["fragmentos"]) == 3
    assert r["fragmentos"][0]["fragmento"]["url"] == "https://es.wikivoyage.org/wiki/San_Carlos_de_Bariloche"
    h = (await api.c.get("/health")).json()
    assert h["rag"]["fragmentos"] > 100 and h["rag"]["listo"] and len(h["rag"]["destinos"]) == 9
    openapi = (await api.c.get("/openapi.json")).json()
    for ruta in ("/tasks", "/tasks/{job_id}", "/tasks/{job_id}/approve", "/threads/{thread_id}", "/rag/buscar",
                 "/health"):
        for metodo, op in openapi["paths"][ruta].items():
            assert "200" in op["responses"] or "202" in op["responses"], (ruta, metodo)
            assert "$ref" in str(op["responses"]), (ruta, metodo)  # respuesta con esquema Pydantic


# ---------------------------------------------------------------------------
# "Reinicio": todo lo que hace falta vive en Redis + checkpointer, no en el proceso
# ---------------------------------------------------------------------------
async def test_un_worker_nuevo_continua_la_conversacion_despues_de_un_reinicio(tmp_path: Path) -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    checkpointer = InMemorySaver()
    comp = crear_componentes(modo_mock=True, ruta_memoria=tmp_path / "chroma")

    store1 = JobStore(redis)
    worker1 = Worker(construir_grafo_api(comp, checkpointer), store1, modo_mock=True)
    job1 = await store1.crear(DEMO, True, "mock")
    await worker1.iniciar(job1, DEMO, auto_aprobar=True)
    await worker1.esperar()
    assert (await store1.obtener(job1))["estado"] == EstadoJob.DONE  # type: ignore[index]
    del worker1, store1  # "se apaga la API"

    store2 = JobStore(redis)
    worker2 = Worker(construir_grafo_api(comp, checkpointer), store2, modo_mock=True)
    snap = await worker2.estado_thread(job1)
    turno = int(snap.values["turno"]) + 1
    job2 = await store2.crear(JULIO, False, "mock", thread_id=job1, turno=turno)
    await worker2.iniciar(job2, JULIO, auto_aprobar=False, turno=turno)
    await worker2.esperar()
    final = await store2.obtener(job2)
    assert final is not None and final["estado"] == EstadoJob.DONE and final["turno"] == 2
    assert final["resultado"]["pedido"]["destino"] == "Bariloche"
    assert final["resultado"]["pedido"]["fecha_ida"] == "2027-07-10"
    assert await store2.jobs_del_thread(job1) == [job1, job2]


@pytest.mark.skipif(not os.getenv("REDIS_URL_TESTS"), reason="sin Redis 8 / Redis Stack (definí REDIS_URL_TESTS)")
async def test_seguimiento_con_async_redis_saver_real_entre_dos_conexiones(tmp_path: Path) -> None:
    comp = crear_componentes(modo_mock=True, ruta_memoria=tmp_path / "chroma")
    thread = f"test-conv-{time.time_ns()}"
    cfg = {"configurable": {"thread_id": thread}}
    async with abrir_checkpointer(os.environ["REDIS_URL_TESTS"]) as saver:
        uno = await construir_grafo_api(comp, saver).ainvoke(entrada_turno(DEMO, 1, auto_aprobar=True), cfg)
    assert uno["respuesta_final"] and uno["turno"] == 1
    async with abrir_checkpointer(os.environ["REDIS_URL_TESTS"]) as saver:  # conexión nueva: "reinicio"
        grafo = construir_grafo_api(comp, saver)
        previo = await grafo.aget_state(cfg)
        assert previo.values["turnos"][0].pedido.destino == "Bariloche"  # vuelve como modelo, no como dict
        dos = await grafo.ainvoke(entrada_turno(JULIO, 2), cfg)
    assert dos["turno"] == 2 and dos["pedido"].fecha_ida == date(2027, 7, 10) and len(dos["turnos"]) == 2
