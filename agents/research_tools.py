"""
research_tools.py — Las 3 herramientas del agente investigador, en el orden en que se usan.

1. consultar_guias   → RAG sobre la base de conocimiento propia (guías de
                       Wikivoyage): lo ESTABLE (qué hacer, cómo llegar, barrios,
                       clima típico, consejos). Gratis, local, con fuente exacta.
2. buscar_en_memoria → busca en ChromaDB lo que ya se trajo de la web antes
                       (precios y cotizaciones de los últimos días).
3. buscar_en_web     → busca en la web con Tavily (solo lo que CAMBIA: precios,
                       tipo de cambio) y GUARDA lo encontrado en la memoria. Si
                       el tema es estable y hay guía del destino, no va a la web.

Reglas de diseño (las mismas de la Pre-entrega 5):
- Esquema de entrada Pydantic (args_schema): el LLM ve qué argumentos existen
  y sus límites.
- Docstring muy descriptivo: es lo que el LLM lee para decidir CUÁNDO usarla.
- Los errores vuelven como TEXTO al LLM (para que razone); nunca rompen el programa.

Cada herramienta devuelve DOS cosas (response_format="content_and_artifact"),
validadas con el modelo Pydantic `SalidaHerramienta` antes de volver al agente:
- content: texto legible para el LLM;
- artifact: la lista de `Fuente` (Pydantic) de cada dato, con su origen
  ("guia", "memoria" o "web"). El nodo investigador arma el Aporte con estos
  artifacts, NO con lo que el LLM diga: así las fuentes no se pueden inventar.
  En las guías, la fuente sale de los METADATOS del fragmento indexado
  (URL del artículo, sección, licencia).

El buscador web se "inyecta": en modo real es Tavily; en modo mock y en los
tests es `BuscadorMock`, que lee resultados prefijados de un JSON.

PE7 (human-in-the-loop): `PERMISO_WEB` dice si en ESTA corrida se puede gastar
una búsqueda en Tavily. Lo fija el nodo investigador con lo que decidió el nodo
aprobacion_web. Si no hay permiso, buscar_en_web NO llama a Tavily: devuelve lo
que haya en memoria (con el filtro de parecido aflojado) o un aviso en texto para
el LLM, como cualquier otro error de herramienta. Es un ContextVar: cada job de
la API corre en su propia tarea de asyncio, así que los permisos no se mezclan.
"""

import asyncio
import contextvars
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

import config
from agents.common import SalidaHerramienta, salida  # noqa: F401 (SalidaHerramienta se re-exporta)
from graph.state import PATRON_PRECIO, TEMAS_ESTABLES, Fuente, PermisoWeb, Tema, cantidad
from rag.base import BaseConocimiento
from rag.memoria import MemoriaBusquedas, ResultadoBusquedaMemoria, ResultadoWeb, id_memoria, normalizar
from rag.retriever import ResultadoBusqueda

log = logging.getLogger(__name__)


# "libre" = comportamiento de la PE6 (sin control humano). Ver el docstring del módulo.
PERMISO_WEB: contextvars.ContextVar[PermisoWeb] = contextvars.ContextVar("permiso_web", default="libre")
WEB_PERMITIDA: frozenset[str] = frozenset({"libre", "aprobada"})

LARGO_EXTRACTO: int = 700  # caracteres de cada resultado que ve el LLM


# ---------------------------------------------------------------------------
# Buscadores web (el real y el simulado)
# ---------------------------------------------------------------------------
class BuscadorWeb(Protocol):
    """Cualquier cosa que sepa buscar en la web y devolver ResultadoWeb."""

    async def buscar(self, consulta: str) -> list[ResultadoWeb]: ...


class BuscadorTavily:
    """Búsqueda web real con Tavily (paquete langchain-tavily).

    `TavilySearch` reemplaza a `TavilySearchResults` (langchain-community),
    que es el que nombra la consigna y hoy está deprecado.
    """

    def __init__(self, api_key: str = config.TAVILY_API_KEY, max_resultados: int = config.TAVILY_MAX_RESULTADOS) -> None:
        from langchain_tavily import TavilySearch

        self._tool = TavilySearch(max_results=max_resultados, tavily_api_key=api_key, search_depth="basic")

    async def buscar(self, consulta: str) -> list[ResultadoWeb]:
        try:
            crudo: dict[str, Any] = await self._tool.ainvoke({"query": consulta})
        except ToolException as e:  # langchain-tavily lanza esto cuando no hay resultados
            if "No search results" in str(e):
                return []
            raise
        if isinstance(crudo, str):  # algunos errores vuelven como texto
            raise RuntimeError(crudo)
        return [
            ResultadoWeb(url=r["url"], titulo=str(r.get("title", "")), contenido=str(r.get("content", "")))
            for r in crudo.get("results", [])
            if r.get("url")
        ]


class BuscadorMock:
    """Tavily simulado: resultados prefijados en data/mock_busquedas.json.

    Elige los resultados cuyas palabras clave aparecen en la consulta. Lleva la
    cuenta de las llamadas (para demostrar que la memoria evita ir a la web) y
    se puede configurar para fallar (escenario "Tavily falla").
    """

    def __init__(self, ruta: Path = config.BUSQUEDAS_MOCK, fallar: bool = False) -> None:
        self._entradas: list[dict[str, Any]] = json.loads(ruta.read_text(encoding="utf-8"))["resultados"]
        self.fallar = fallar
        self.llamadas: list[str] = []

    async def buscar(self, consulta: str) -> list[ResultadoWeb]:
        self.llamadas.append(consulta)
        if self.fallar:
            raise ConnectionError("Tavily no respondió (simulado): 503 Service Unavailable")
        texto = normalizar(consulta)
        puntajes: list[tuple[int, dict[str, Any]]] = []
        for e in self._entradas:
            if normalizar(e["destino"]) not in texto:
                continue
            puntaje = sum(1 for p in e["palabras"] if normalizar(p) in texto)
            if puntaje:
                puntajes.append((puntaje, e))
        puntajes.sort(key=lambda par: -par[0])
        return [
            ResultadoWeb(url=e["url"], titulo=e["titulo"], contenido=e["contenido"])
            for _, e in puntajes[: config.TAVILY_MAX_RESULTADOS]
        ]


# ---------------------------------------------------------------------------
# Esquemas de entrada (lo que ve el LLM)
# ---------------------------------------------------------------------------
_DESC_TEMA = (
    "Tema de la búsqueda: 'pasajes' (vuelos o micros), 'alojamiento', 'comida', 'cambio' (cotización del dólar), "
    "'clima', 'actividades', 'como_llegar', 'barrios', 'consejos' u 'otro'."
)
LARGO_FRAGMENTO: int = 1100  # caracteres de cada fragmento de guía que ve el LLM


class ConsultarGuiasInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(min_length=3, max_length=300,
                          description="Qué querés saber, en castellano: 'qué hacer en invierno', 'cómo llegar desde el aeropuerto'.")
    destino: str = Field(min_length=2, max_length=80, description="Destino del viaje, por ejemplo 'Bariloche'.")
    tema: Tema = Field(description=_DESC_TEMA)


class BuscarEnMemoriaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(min_length=3, max_length=300, description="Qué buscás, en castellano y concreto.")
    destino: str = Field(min_length=2, max_length=80, description="Ciudad de destino, por ejemplo 'Bariloche'.")
    tema: Tema = Field(description=_DESC_TEMA)


class BuscarEnWebInput(BuscarEnMemoriaInput):
    forzar: bool = Field(
        default=False,
        description=(
            "Dejalo en false. Ponelo en true SOLO si en la memoria hay datos de este tema pero NO "
            "sirven (por ejemplo, no traen precios) y necesitás buscar de nuevo en la web."
        ),
    )


# ---------------------------------------------------------------------------
# Formateo de resultados
# ---------------------------------------------------------------------------
def _formatear_memoria(r: ResultadoBusquedaMemoria, consulta: str, ahora: datetime) -> str:
    lineas = [f"MEMORIA — {cantidad(len(r.utiles), 'dato vigente', 'datos vigentes')} para '{consulta}':"]
    for reg in r.utiles:
        extracto = reg.texto[:LARGO_EXTRACTO] + ("..." if len(reg.texto) > LARGO_EXTRACTO else "")
        lineas.append(
            f"\n[{reg.id}] origen=memoria | tema={reg.tema} | buscado hace {cantidad(reg.antiguedad_dias(ahora), 'día', 'días')} "
            f"({reg.fecha.date().isoformat()}) | url={reg.url}\n{extracto}"
        )
    return "\n".join(lineas)


def _fuentes_de_memoria(r: ResultadoBusquedaMemoria) -> list[Fuente]:
    return [
        Fuente(ref=reg.id, origen="memoria", tema=reg.tema, url=reg.url, fecha_busqueda=reg.fecha.date())  # type: ignore[arg-type]
        for reg in r.utiles
    ]


def formatear_guias(r: ResultadoBusqueda, tema: str) -> str:
    lineas = [f"GUÍAS (Wikivoyage, {cantidad(len(r.fragmentos), 'fragmento')}) para '{r.consulta}' "
              f"[destino={r.destino}, tema={tema}]. Información estable; los precios que mencionen las guías "
              "pueden estar desactualizados: NO los uses como precio actual."]
    for x in r.fragmentos:
        f = x.fragmento
        texto = f.texto.replace("\n\n", "\n")
        extracto = texto[:LARGO_FRAGMENTO] + ("..." if len(texto) > LARGO_FRAGMENTO else "")
        lineas.append(f"\n[{f.id}] origen=guia | {f.titulo} § {f.seccion} | url={f.url}\n{extracto}")
    return "\n".join(lineas)


def fuentes_de_guias(r: ResultadoBusqueda, tema: Tema) -> list[Fuente]:
    """Las fuentes salen de los METADATOS de cada fragmento indexado (no del texto ni del LLM)."""
    return [Fuente(ref=x.fragmento.id, origen="guia", tema=tema, url=x.fragmento.url, titulo=x.fragmento.titulo,
                   seccion=x.fragmento.seccion, licencia=x.fragmento.licencia) for x in r.fragmentos]


# ---------------------------------------------------------------------------
# Fábrica de herramientas
# ---------------------------------------------------------------------------
def crear_herramientas_investigador(memoria: MemoriaBusquedas, buscador: BuscadorWeb,
                                    base: BaseConocimiento | None = None) -> list[BaseTool]:
    """Crea las herramientas del investigador, conectadas a ESTA base de guías, ESTA memoria y ESTE buscador.

    Sin `base` (tests de la PE6) quedan solo las 2 herramientas de memoria y web.
    """

    async def consultar_guias(consulta: str, destino: str, tema: Tema) -> tuple[str, list[Fuente]]:
        assert base is not None
        try:
            r = await base.consultar(consulta, destino)
        except Exception as e:  # índice dañado, modelo que no carga...: se informa, no rompe el grafo
            log.warning("consultar_guias falló: %s", e)
            return salida(f"ERROR al consultar las guías: {type(e).__name__}: {e}. Probá con buscar_en_memoria.")
        if r.destino_sin_guia:
            return salida(f"SIN GUÍA para '{destino}': la base de conocimiento tiene guías de "
                          f"{', '.join(base.destinos())}. Para este destino usá buscar_en_memoria y, si no hay "
                          "nada, buscar_en_web.")
        if not r.fragmentos:
            return salida(f"SIN RESULTADOS EN LAS GUÍAS para '{consulta}' [destino={destino}]. Probá con otras "
                          "palabras o usá buscar_en_memoria.")
        return salida(formatear_guias(r, tema), fuentes_de_guias(r, tema))

    async def buscar_en_memoria(consulta: str, destino: str, tema: Tema) -> tuple[str, list[Fuente]]:
        try:
            ahora = datetime.now().astimezone()
            r = await asyncio.to_thread(memoria.buscar, consulta, destino, tema, 4, ahora)
            if r.hay_utiles:
                return _formatear_memoria(r, consulta, ahora), _fuentes_de_memoria(r)
            motivos: list[str] = []
            if r.descartados_por_antiguedad:
                motivos.append(f"{cantidad(r.descartados_por_antiguedad, 'dato vencido', 'datos vencidos')} "
                               f"(más de {memoria.max_dias} días)")
            if r.descartados_por_relevancia:
                motivos.append(f"{cantidad(r.descartados_por_relevancia, 'dato reciente', 'datos recientes')} "
                               "pero poco parecido a la consulta")
            detalle = "; ".join(motivos) or "no hay nada guardado sobre este destino y tema"
            return (
                f"SIN RESULTADOS EN MEMORIA para '{consulta}' [destino={destino}, tema={tema}]: {detalle}. "
                "Buscá en la web con buscar_en_web."
            ), []
        except Exception as e:  # la base local dañada, disco lleno, etc.
            log.warning("buscar_en_memoria falló: %s", e)
            return f"ERROR al leer la memoria: {type(e).__name__}: {e}. Buscá en la web con buscar_en_web.", []

    async def buscar_en_web(consulta: str, destino: str, tema: Tema, forzar: bool = False) -> tuple[str, list[Fuente]]:
        ahora = datetime.now().astimezone()
        # Guarda (entrega final): lo estable lo responden las guías. Si hay guía del destino, no se gasta Tavily.
        if base is not None and tema in TEMAS_ESTABLES and base.tiene_guia(destino):
            log.info("buscar_en_web: evitada, el tema '%s' es estable y hay guía de %s", tema, destino)
            return (f"NO SE BUSCÓ EN LA WEB: '{tema}' es un tema estable y la base de conocimiento tiene la guía de "
                    f"{destino}. Usá consultar_guias (la web es solo para precios y tipo de cambio)."), []
        # Guarda: si la memoria ya tiene datos vigentes, no se gasta una búsqueda web.
        # forzar=true solo pasa si lo guardado NO trae precios: si los trae, se bloquea POR CÓDIGO
        # (en la corrida real del escenario 5 el LLM forzaba la web teniendo precios en memoria).
        try:
            previo = await asyncio.to_thread(memoria.buscar, consulta, destino, tema, 4, ahora)
            con_precios = any(PATRON_PRECIO.search(reg.texto) for reg in previo.utiles)
            if previo.hay_utiles and (not forzar or con_precios):
                log.info("buscar_en_web: evitada (forzar=%s), datos vigentes en memoria: %d", forzar, len(previo.utiles))
                motivo = ("la memoria ya tiene datos vigentes CON PRECIOS sobre esto: forzar=true no aplica, usá estos"
                          if forzar else
                          "la memoria ya tiene datos vigentes sobre esto (si no traen precios, repetí con forzar=true)")
                return (f"NO SE BUSCÓ EN LA WEB: {motivo}.\n\n" + _formatear_memoria(previo, consulta, ahora)), \
                    _fuentes_de_memoria(previo)
        except Exception as e:  # si la memoria falla, se sigue a la web igual
            log.warning("No se pudo consultar la memoria antes de la web: %s", e)

        permiso = PERMISO_WEB.get()
        if permiso not in WEB_PERMITIDA:
            return await _sin_permiso_web(consulta, destino, tema, permiso, ahora)

        try:
            resultados = await buscador.buscar(f"{consulta} {destino}" if normalizar(destino) not in normalizar(consulta) else consulta)
        except Exception as e:  # Tavily caído, clave inválida, sin internet, cuota agotada...
            log.warning("buscar_en_web falló: %s", e)
            return (
                f"ERROR al buscar en la web (Tavily): {type(e).__name__}: {e}. "
                "No inventes datos: informá que la búsqueda web no está disponible."
            ), []

        if not resultados:
            return f"SIN RESULTADOS EN LA WEB para '{consulta}'. Probá con otras palabras.", []

        try:
            ids = await asyncio.to_thread(memoria.guardar, resultados, destino, tema, consulta, ahora)
        except Exception as e:  # que falle el guardado no invalida los resultados
            log.warning("No se pudo guardar en memoria: %s", e)
            ids = [id_memoria(r.url, destino, tema) for r in resultados]

        lineas = [f"WEB (Tavily) — {cantidad(len(resultados), 'resultado')} para '{consulta}' (guardados en memoria):"]
        for r, id_ in zip(resultados, ids, strict=True):
            extracto = r.contenido[:LARGO_EXTRACTO] + ("..." if len(r.contenido) > LARGO_EXTRACTO else "")
            lineas.append(f"\n[{r.url}] origen=web | tema={tema} | guardado como {id_}\n{r.titulo}\n{extracto}")
        fuentes = [Fuente(ref=r.url, origen="web", tema=tema, url=r.url, fecha_busqueda=ahora.date()) for r in resultados]
        return "\n".join(lineas), fuentes

    async def _sin_permiso_web(consulta: str, destino: str, tema: Tema, permiso: str,
                               ahora: datetime) -> tuple[str, list[Fuente]]:
        """La web no está autorizada: lo mejor que hay en memoria, o un aviso claro (nunca Tavily)."""
        motivo = ("la persona que revisa el pedido RECHAZÓ la búsqueda web" if permiso == "rechazada"
                  else "la búsqueda web no fue autorizada (la memoria cubría los temas pedidos)")
        log.info("buscar_en_web: bloqueada (permiso=%s) para '%s'", permiso, consulta)
        try:
            flojo = await asyncio.to_thread(memoria.buscar, consulta, destino, tema, 4, ahora, 2.0)
        except Exception as e:  # noqa: BLE001 — si la memoria falla, se avisa igual
            log.warning("No se pudo consultar la memoria sin permiso web: %s", e)
            flojo = None
        if flojo is not None and flojo.hay_utiles:
            return (f"NO SE BUSCÓ EN LA WEB: {motivo}. Estos son los datos vigentes en memoria sobre "
                    f"{destino} / {tema} (pueden ser menos precisos):\n\n" + _formatear_memoria(flojo, consulta, ahora)), \
                _fuentes_de_memoria(flojo)
        return (f"WEB NO AUTORIZADA: {motivo}, y la memoria no tiene datos de {destino} / {tema}. "
                "No inventes datos: informalo en 'Datos que faltan'."), []

    herramienta_guias = StructuredTool.from_function(
        coroutine=_validada(consultar_guias),
        name="consultar_guias",
        args_schema=ConsultarGuiasInput,
        response_format="content_and_artifact",
        description=(
            "Consulta la BASE DE CONOCIMIENTO de guías de viaje (Wikivoyage en español, búsqueda híbrida con "
            "re-ranking) sobre lo ESTABLE de un destino: qué ver y qué hacer, cómo llegar y moverse, barrios y "
            "zonas, clima típico por mes, dónde comer o dormir (tipos y zonas, no precios) y consejos de seguridad. "
            "Usala PRIMERO para esos temas: es gratis e instantánea. Cada fragmento trae su ID (guia:...), la "
            "sección y la URL del artículo: citalo. NO sirve para precios actuales ni tipo de cambio. Si responde "
            "'SIN GUÍA', el destino no está en la base: usá buscar_en_memoria."
        ),
    )
    herramienta_memoria = StructuredTool.from_function(
        coroutine=_validada(buscar_en_memoria),
        name="buscar_en_memoria",
        args_schema=BuscarEnMemoriaInput,
        response_format="content_and_artifact",
        description=(
            "Busca en la MEMORIA LOCAL (ChromaDB) datos que ya se trajeron de la web antes (precios de pasajes, "
            "alojamiento y comida, tipo de cambio). Usala SIEMPRE antes de buscar_en_web: es gratis e instantánea. "
            f"Solo devuelve datos del mismo destino y tema, guardados hace {config.MEMORIA_MAX_DIAS} días o menos "
            "y parecidos a tu consulta. Cada dato trae su ID (mem-...), la URL original y la fecha de búsqueda. "
            "Si responde 'SIN RESULTADOS EN MEMORIA', usá buscar_en_web."
        ),
    )
    herramienta_web = StructuredTool.from_function(
        coroutine=_validada(buscar_en_web),
        name="buscar_en_web",
        args_schema=BuscarEnWebInput,
        response_format="content_and_artifact",
        description=(
            "Busca en la WEB con Tavily lo que CAMBIA: precios actuales de pasajes (avión o micro), alojamiento, "
            "comida o excursiones, y la cotización del dólar. Usala SOLO si buscar_en_memoria no trajo nada útil "
            "(y para temas estables solo si el destino no tiene guía). Todo lo que encuentra queda guardado en la "
            "memoria. Cada resultado trae su URL: citala siempre. Si responde con 'ERROR', no inventes datos."
        ),
    )
    if base is None:
        return [herramienta_memoria, herramienta_web]
    return [herramienta_guias, herramienta_memoria, herramienta_web]


def _validada(fn: Any) -> Any:
    """Toda salida de herramienta pasa por SalidaHerramienta (Pydantic) antes de volver al agente."""
    import functools

    @functools.wraps(fn)
    async def envoltura(*args: Any, **kwargs: Any) -> tuple[str, list[Fuente]]:
        texto, fuentes = await fn(*args, **kwargs)
        return salida(texto, fuentes)

    return envoltura
