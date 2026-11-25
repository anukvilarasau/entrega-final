"""Auditoría automática de los requisitos transversales de la entrega final.

1. Async: ningún módulo de la app importa `requests` (ni urllib) y ninguna función async llama a `time.sleep`;
   todas las herramientas de los agentes que hacen E/S son corutinas.
2. Pydantic: toda herramienta tiene args_schema Pydantic; todo endpoint declara response_model.
3. Configuración: ningún modelo, índice ni URL de servicio está escrito fuera de config.py; .env.example
   documenta TODAS las variables de config.Ajustes.
"""

import ast
import re
from pathlib import Path

from fastapi.routing import APIRoute
from pydantic import BaseModel

import config
from agents.analyst_tools import crear_herramientas_analista
from agents.research_tools import BuscadorMock, crear_herramientas_investigador
from api.main import create_app
from rag.base import crear_base_conocimiento
from rag.memoria import EmbeddingHash, MemoriaBusquedas

RAIZ = config.BASE_DIR
PAQUETES = ["agents", "api", "graph", "rag", "observability"]
ARCHIVOS_APP = [p for d in PAQUETES for p in (RAIZ / d).rglob("*.py")] + [RAIZ / "main.py", RAIZ / "config.py"]


def _arbol(ruta: Path) -> ast.Module:
    return ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))


def test_ningun_modulo_de_la_app_usa_clientes_http_sync() -> None:
    prohibidos = {"requests", "urllib", "urllib3", "http.client"}
    hallados = []
    for ruta in ARCHIVOS_APP:
        for nodo in ast.walk(_arbol(ruta)):
            nombres = ([a.name for a in nodo.names] if isinstance(nodo, ast.Import)
                       else [nodo.module or ""] if isinstance(nodo, ast.ImportFrom) else [])
            hallados += [f"{ruta.relative_to(RAIZ)}: {n}" for n in nombres if n.split(".")[0] in prohibidos
                         or n in prohibidos]
    assert hallados == []


def test_ninguna_funcion_async_bloquea_con_time_sleep() -> None:
    hallados = []
    for ruta in ARCHIVOS_APP:
        for f in ast.walk(_arbol(ruta)):
            if isinstance(f, ast.AsyncFunctionDef):
                for n in ast.walk(f):
                    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sleep"
                            and isinstance(n.func.value, ast.Name) and n.func.value.id == "time"):
                        hallados.append(f"{ruta.relative_to(RAIZ)}:{n.lineno}")
    assert hallados == []


def test_herramientas_async_con_args_schema_pydantic(tmp_path: Path) -> None:
    memoria = MemoriaBusquedas(ruta=tmp_path / "m", embedding=EmbeddingHash())
    investigador = crear_herramientas_investigador(memoria, BuscadorMock(), crear_base_conocimiento(modo_mock=True))
    for t in investigador:  # hacen E/S (índice, memoria, Tavily): corutinas
        assert t.coroutine is not None, t.name
    for t in [*investigador, *crear_herramientas_analista()]:
        assert isinstance(t.args_schema, type) and issubclass(t.args_schema, BaseModel), t.name
        assert t.response_format == "content_and_artifact", t.name


def test_todos_los_endpoints_declaran_response_model() -> None:
    app = create_app()
    rutas = [r for r in app.routes if isinstance(r, APIRoute) and r.include_in_schema]
    assert {r.path for r in rutas} >= {"/tasks", "/tasks/{job_id}", "/tasks/{job_id}/approve", "/threads/{thread_id}",
                                       "/rag/buscar", "/health"}
    for r in rutas:
        assert r.response_model is not None, r.path
        for _, cuerpo in ([(None, r.body_field)] if r.body_field else []):
            assert issubclass(cuerpo.field_info.annotation, BaseModel), r.path  # entrada validada


def test_sin_modelos_ni_urls_hardcodeados_fuera_de_config() -> None:
    # Un id de modelo de Claude, de fastembed/HF, o una URL de servicio que tiene que venir del entorno.
    patron = re.compile(r"claude-(haiku|sonnet|opus|fable)|sentence-transformers/|jinaai/|BAAI/|redis://|"
                        r"api\.smith\.langchain|pinecone\.io")
    permitidos = {RAIZ / "config.py",
                  RAIZ / "observability" / "monitoreo.py"}  # tabla de PRECIOS por modelo (no elige el modelo)
    hallados = []
    for ruta in ARCHIVOS_APP:
        if ruta in permitidos:
            continue
        arbol = _arbol(ruta)
        docstrings = {id(n.body[0].value) for n in ast.walk(arbol)  # la documentación puede nombrar modelos
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(arbol):
            if id(n) in docstrings:
                continue
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and patron.search(n.value):
                hallados.append(f"{ruta.relative_to(RAIZ)}:{n.lineno}: {n.value[:60]}")
    assert hallados == []


def test_env_example_documenta_todas_las_variables() -> None:
    ejemplo = (RAIZ / ".env.example").read_text(encoding="utf-8")
    definidas = set(re.findall(r"(?m)^#?\s*([A-Z][A-Z0-9_]+)=", ejemplo))
    faltan = {c.upper() for c in config.Ajustes.model_fields} - definidas
    assert faltan == set(), f"Faltan en .env.example: {sorted(faltan)}"
