"""
research_agent.py — El agente INVESTIGADOR.

Es un agente ReAct armado con `create_react_agent` (LangGraph) con TRES
herramientas, en este orden: consultar_guias (RAG sobre las guías: lo estable),
buscar_en_memoria (búsquedas web de los últimos días) y buscar_en_web (Tavily,
solo lo que cambia: precios y tipo de cambio, con la pausa HITL antes). Consigue
datos con fuente. No arma itinerarios ni hace cuentas: eso es trabajo de otros.

Cómo se conecta con el grafo principal (nodo "investigador"):
1. Recibe SOLO `instruccion_actual` (lo que le pidió el supervisor), la lista
   estructurada TEMAS A BUSCAR (`temas_pedidos`) y el bloque "DATOS DEL PEDIDO" (origen, destino, fechas, personas,
   presupuesto), armado por código desde `state["pedido"]`. No recibe la
   pregunta original, ni el historial, ni los aportes de otros: así se evita
   la "contaminación de contexto" y no depende de que el supervisor se
   acuerde de copiar todos los datos en la instrucción.
2. Corre su propio ciclo ReAct (pensar → herramienta → observar → ...), con
   su propio techo de pasos (RECURSION_LIMIT_ESPECIALISTA).
3. Arma un `Aporte` con: su respuesta, las fuentes que DEVOLVIERON LAS
   HERRAMIENTAS (no las que diga el LLM) y cada llamada que hizo.
4. Un chequeo por código (`chequear_investigacion`) anota problemas
   objetivos en `observaciones` (sin fuentes, sin precios, web caída...).
   El supervisor los lee para decidir si pide rehacer el trabajo.
5. Escribe SOLO su parte del estado (validar_actualizacion lo garantiza).
"""

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

import config
from agents.common import NodoAsync, correr_react, extraer_llamadas, respuesta_final, resultados
from agents.research_tools import PERMISO_WEB
from graph.state import (
    PATRON_PRECIO,
    Aporte,
    CoberturaTema,
    Fuente,
    OrquestadorState,
    cantidad,
    contexto_especialista,
    turno_actual,
    validar_actualizacion,
)

log = logging.getLogger(__name__)

PROMPT_INVESTIGADOR: str = f"""Sos el agente INVESTIGADOR de un asesor de viajes. Tu ÚNICO trabajo es conseguir datos con fuente. No armás itinerarios ni hacés cuentas: eso lo hacen otros agentes.

Herramientas, EN ESTE ORDEN:
1. consultar_guias: la base de conocimiento de guías (Wikivoyage). Para lo ESTABLE: actividades (qué ver y hacer), como_llegar, barrios, clima (clima típico del mes), consejos. Usala primero para cada uno de esos temas (podés pedir varias en paralelo). Si la guía responde, NO busques ese tema en la web.
2. buscar_en_memoria: datos ya buscados en la web antes ({config.MEMORIA_MAX_DIAS} días o menos). Para lo que CAMBIA: pasajes, alojamiento, comida (precios) y cambio (cotización del dólar). También si consultar_guias dijo "SIN GUÍA" (destino sin guía).
3. buscar_en_web: Tavily. SOLO para los temas en los que la memoria respondió "SIN RESULTADOS EN MEMORIA". Usá forzar=true únicamente si lo que hay en memoria no sirve (por ejemplo, no trae precios).

Presupuesto de búsquedas: como máximo UNA búsqueda web por tema (dos solo si la primera no trajo ningún precio). Los precios de la web son de referencia: rangos, valores "desde" o de fechas cercanas sirven; no busques precios exactos para las fechas del viaje. Los precios que aparezcan en las guías pueden ser viejos: NO los informes como precio actual.

Reglas:
1. No inventes nada. Cada dato va con su fuente entre corchetes, tal como la devolvió la herramienta: [guia:...] (guía), [https://...] (web) o [mem-...] (memoria), y aclarando (guía), (web) o (memoria).
2. Precios: copiá el monto con su moneda y SU UNIDAD tal como dice la fuente:
   - pasajes: por persona, ida y vuelta o por tramo; avión o micro;
   - alojamiento: "por noche por habitación (para N personas)" o "por persona por noche";
   - si la fuente no dice la unidad, escribí "unidad no especificada".
3. Tipo de cambio: buscá el dólar oficial, tarjeta y MEP (no el blue); informá cada valor y su fuente.
4. Clima: usá la tabla de clima típico de la guía (máximas, mínimas y lluvias del mes del viaje) si está.
5. Si una búsqueda da ERROR o no encuentra nada, decilo en "Datos que faltan". No lo tapes.

Formato de tu respuesta final (en castellano, breve):
## Hallazgos
### <Tema>
- <dato (con monto, moneda y unidad si es un precio)> — [fuente] (guía|web|memoria)
## Datos que faltan
- <lo que no se encontró, o "ninguno">"""

TEMAS_CON_PRECIO: tuple[str, ...] = ("pasajes", "alojamiento")


def crear_agente_investigador(llm: BaseChatModel, herramientas: list[BaseTool]) -> CompiledStateGraph:
    """El agente ReAct del investigador (subgrafo con su propio ciclo)."""
    return create_react_agent(llm, herramientas, prompt=PROMPT_INVESTIGADOR, name="investigador")


def calcular_cobertura(mensajes: list[BaseMessage]) -> list[CoberturaTema]:
    """Resumen por tema armado POR CÓDIGO: fuentes, de dónde salieron y si hay precios.

    El supervisor lo ve junto al texto del investigador, así no depende de leer un
    informe largo (en la corrida real del escenario 5 el texto le llegaba cortado y
    volvió a pedir temas que ya estaban cubiertos).
    """
    _, _, argumentos_de = extraer_llamadas(mensajes)
    por_tema: dict[str, dict[str, Any]] = {}
    for m in resultados(mensajes):
        tema = str(argumentos_de.get(m.tool_call_id, {}).get("tema", "otro"))
        datos = por_tema.setdefault(tema, {"refs": {}, "precios": False})
        texto = str(m.content)
        # Los precios de las guías pueden ser viejos: solo cuentan los de la memoria o la web.
        if m.name != "consultar_guias" and not texto.startswith("ERROR") and PATRON_PRECIO.search(texto):
            datos["precios"] = True
        for f in (m.artifact if isinstance(m.artifact, list) else []):
            if isinstance(f, Fuente):
                datos["refs"].setdefault(f.url or f.ref, f.origen)
    return [
        CoberturaTema(
            tema=tema,  # type: ignore[arg-type]
            fuentes=len(d["refs"]),
            memoria=sum(1 for o in d["refs"].values() if o == "memoria"),
            web=sum(1 for o in d["refs"].values() if o == "web"),
            guia=sum(1 for o in d["refs"].values() if o == "guia"),
            con_precios=d["precios"],
        )
        for tema, d in por_tema.items()
    ]


def chequear_investigacion(
    temas_pedidos: list[str], contenido: str, mensajes: list[BaseMessage], fuentes: list[Fuente]
) -> list[str]:
    """Chequeo POR CÓDIGO (no depende del LLM). Devuelve la lista de problemas encontrados.

    Los temas a controlar salen del campo ESTRUCTURADO `temas_pedidos` que escribió el
    supervisor (no de buscar palabras en la instrucción: "no repitas pasajes" no es pedir pasajes).
    """
    observaciones: list[str] = []
    hechos = resultados(mensajes)
    if not hechos:
        observaciones.append("El investigador no usó ninguna herramienta: sus datos no tienen respaldo.")
    if hechos and not fuentes:
        observaciones.append("Las búsquedas no devolvieron ninguna fuente utilizable.")

    errores = [str(m.content) for m in hechos if str(m.content).startswith("ERROR")]
    for e in errores:
        observaciones.append(f"Falló una búsqueda: {e[:160]}")

    # ¿Se buscó cada tema pedido? ¿Hay precios en pasajes y alojamiento?
    _, _, argumentos_de = extraer_llamadas(mensajes)
    for tema in dict.fromkeys(temas_pedidos):
        del_tema = [m for m in hechos
                    if argumentos_de.get(m.tool_call_id, {}).get("tema") == tema and not str(m.content).startswith("ERROR")]
        if not del_tema:
            if not errores:
                observaciones.append(f"Se pidió el tema '{tema}' pero no se hizo ninguna búsqueda sobre él.")
        elif tema in TEMAS_CON_PRECIO and not any(PATRON_PRECIO.search(str(m.content)) for m in del_tema
                                                   if m.name != "consultar_guias"):
            observaciones.append(f"Se pidieron precios de {tema} pero los resultados encontrados no traen ningún precio.")

    # ¿La respuesta cita al menos una de las fuentes reales?
    if fuentes and not any(f.ref in contenido or (f.url and f.url in contenido) for f in fuentes):
        observaciones.append("La respuesta del investigador no cita ninguna de las fuentes que encontró.")
    return observaciones


def crear_nodo_investigador(agente: CompiledStateGraph) -> NodoAsync:
    """Envuelve al agente ReAct como nodo del grafo principal."""

    async def investigador(state: OrquestadorState) -> dict[str, Any]:
        instruccion = state.get("instruccion_actual", "").strip() or state["pregunta"]
        ronda = max(1, state.get("pasos_supervisor", 1))
        log.info("[investigador] ronda %d — instrucción: %s", ronda, instruccion[:150].replace("\n", " "))

        # Contexto acotado: la instrucción del supervisor + los datos estructurados del
        # pedido. Nada del historial general ni la pregunta original.
        temas = list(state.get("temas_pedidos", []))
        extra = f"TEMAS A BUSCAR: {', '.join(temas)}" if temas else ""
        # PE7: el permiso de usar la web que decidió el nodo aprobacion_web (o "libre" sin HITL).
        token = PERMISO_WEB.set(state.get("permiso_web", "libre"))
        try:
            mensajes, cortado = await correr_react(agente, contexto_especialista(instruccion, state.get("pedido"), extra))
        finally:
            PERMISO_WEB.reset(token)

        llamadas, fuentes, _ = extraer_llamadas(mensajes)
        contenido = respuesta_final(mensajes, "(el investigador no llegó a dar una respuesta final)")
        observaciones = chequear_investigacion(temas, contenido, mensajes, fuentes)
        if cortado:
            observaciones.insert(0, f"El ciclo ReAct se cortó por el límite de {config.RECURSION_LIMIT_ESPECIALISTA} pasos.")

        aporte = Aporte(
            agente="investigador",
            ronda=ronda,
            turno=turno_actual(state),
            instruccion=instruccion,
            contenido=contenido,
            fuentes=fuentes,
            herramientas=list(dict.fromkeys(ll.herramienta for ll in llamadas)),
            llamadas=llamadas,
            cobertura=calcular_cobertura(mensajes),
            observaciones=observaciones,
        )
        origenes = {o: sum(1 for f in aporte.fuentes if f.origen == o) for o in ("guia", "memoria", "web")}
        log.info(
            "[investigador] llamadas: %d, fuentes: %d de guías / %d de memoria / %d de la web, observaciones: %d",
            len(llamadas), origenes["guia"], origenes["memoria"], origenes["web"], len(observaciones),
        )
        resumen = AIMessage(
            content=f"[investigador] {cantidad(len(aporte.fuentes), 'fuente')} ({origenes['guia']} de guías, "
                    f"{origenes['memoria']} de memoria, {origenes['web']} de la web). "
                    f"Observaciones: {'; '.join(observaciones) or 'ninguna'}",
            name="investigador",
        )
        return validar_actualizacion("investigador", {"aportes": [aporte], "messages": [resumen]})

    return investigador
