"""
graph/hitl.py — Nodo human-in-the-loop: pedir permiso antes de gastar búsquedas web.

Dónde está el punto crítico: el investigador es el único agente que gasta
plata y cuota de terceros (cada búsqueda en Tavily). Por eso el grafo de la API
intercala el nodo "aprobacion_web" en el camino supervisor → investigador:

    supervisor ──(investigador)──► aprobacion_web ──► investigador ──► supervisor

Qué hace el nodo, por código (sin LLM):
1. Mira los temas que pidió el supervisor (`temas_pedidos`) y, para cada uno,
   quién lo puede responder sin gastar:
   - las GUÍAS (RAG), si el tema es estable (actividades, cómo llegar, barrios,
     clima típico, consejos) y el destino tiene guía en la base de conocimiento;
   - la MEMORIA, si ChromaDB ya tiene datos vigentes (MemoriaBusquedas.hay_vigentes).
2. Si guías + memoria responden todos los temas → NO hay pausa (permiso
   "sin_pedir": la web queda cerrada y el investigador trabaja sin gastar).
3. Si el request vino con auto_aprobar=true (prueba de carga) → "aprobada" sin pausa.
4. Si una persona ya decidió en este job (ronda anterior) → se respeta esa decisión.
5. Si no: `interrupt()` con lo que se quiere buscar. El grafo se detiene, el
   checkpoint queda en Redis y el job pasa a WAITING_APPROVAL. Cuando llega
   POST /tasks/{id}/approve, el worker reanuda con Command(resume={...}) y el
   nodo vuelve a correr desde el principio: los pasos 1-4 son determinísticos y
   baratos, e `interrupt()` esta vez devuelve la respuesta de la persona.

aprobado=false no corta el viaje: el investigador sigue SIN web (buscar_en_web
devuelve lo que haya en memoria o un aviso) y la síntesis aclara que faltan datos.
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from agents.common import NodoAsync
from graph.state import (
    TEMAS_ESTABLES,
    OrquestadorState,
    PermisoWeb,
    RegistroAprobacion,
    Tema,
    aprobaciones_del_turno,
    turno_actual,
    validar_actualizacion,
)
from rag.base import BaseConocimiento
from rag.memoria import MemoriaBusquedas

log = logging.getLogger(__name__)

TIPO_SOLICITUD: str = "aprobacion_busqueda_web"


class SolicitudAprobacion(BaseModel):
    """Lo que ve la persona (GET /tasks/{id} → "aprobacion_pendiente") mientras el grafo está pausado."""

    model_config = ConfigDict(extra="forbid")

    tipo: str = TIPO_SOLICITUD
    ronda: int
    destino: str | None
    temas_sin_memoria: list[Tema] = Field(description="Temas que el investigador va a buscar en la web (Tavily).")
    temas_en_memoria: list[Tema] = Field(description="Temas que ya responde la memoria local (no gastan búsquedas).")
    temas_en_guias: list[Tema] = Field(default_factory=list,
                                       description="Temas estables que responden las guías (RAG, no gastan).")
    consultas_previstas: list[str] = Field(description="Búsquedas web aproximadas que se harían (una por tema).")
    instruccion_del_supervisor: str
    como_responder: str = 'POST /tasks/{job_id}/approve con {"aprobado": true} o {"aprobado": false}'


class RespuestaAprobacion(BaseModel):
    """Lo que manda la persona en POST /tasks/{id}/approve (y lo que recibe interrupt() al reanudar)."""

    model_config = ConfigDict(extra="forbid")

    aprobado: bool = Field(description="true: puede buscar en la web; false: sigue solo con la memoria.")
    comentario: str = Field(default="", max_length=500, description="Opcional, queda en el historial del job.")


def interpretar_respuesta(valor: Any) -> RespuestaAprobacion:
    """El valor de Command(resume=...). Cualquier cosa que no sea un sí explícito cuenta como NO (no gasta)."""
    if isinstance(valor, RespuestaAprobacion):
        return valor
    if isinstance(valor, bool):
        return RespuestaAprobacion(aprobado=valor)
    if isinstance(valor, dict):
        try:
            return RespuestaAprobacion.model_validate(valor)
        except ValueError:
            pass
    log.warning("[aprobacion_web] respuesta no reconocida (%r): se toma como rechazo", valor)
    return RespuestaAprobacion(aprobado=False, comentario="respuesta no reconocida: se tomó como rechazo")


async def temas_por_origen(memoria: MemoriaBusquedas, destino: str | None, temas: list[Tema],
                           base: BaseConocimiento | None = None) -> tuple[list[Tema], list[Tema], list[Tema]]:
    """(temas que irían a la web, temas que responde la memoria, temas que responden las guías)."""
    if not destino:  # sin destino no se puede consultar nada local: todo iría a la web
        return list(temas), [], []
    sin: list[Tema] = []
    con: list[Tema] = []
    guias: list[Tema] = []
    hay_guia = base is not None and base.tiene_guia(destino)  # por metadatos: sin embeddings ni LLM
    for tema in dict.fromkeys(temas):
        if hay_guia and tema in TEMAS_ESTABLES:
            guias.append(tema)
            continue
        hay = await asyncio.to_thread(memoria.hay_vigentes, destino, tema)  # ChromaDB es sync: fuera del event loop
        (con if hay else sin).append(tema)
    return sin, con, guias


CONSULTAS_POR_TEMA: dict[str, str] = {
    "pasajes": "precio de pasajes {origen} → {destino}",
    "alojamiento": "precio de alojamiento por noche en {destino}",
    "comida": "cuánto cuesta comer en {destino}",
    "clima": "clima en {destino} en {mes}",
    "actividades": "actividades y excursiones en {destino}",
    "como_llegar": "cómo llegar de {origen} a {destino}",
    "barrios": "barrios y zonas para alojarse en {destino}",
    "consejos": "consejos y seguridad para viajar a {destino}",
    "cambio": "cotización del dólar oficial, tarjeta y MEP",
    "otro": "información de viaje sobre {destino}",
}


def consulta_prevista(tema: str, pedido: Any) -> str:
    """Una descripción legible de la búsqueda web que haría el investigador para ese tema."""
    from graph.state import MESES

    datos = {"origen": "el origen", "destino": "el destino", "mes": "las fechas del viaje"}
    if pedido is not None:
        datos.update({k: v for k, v in {"origen": pedido.origen, "destino": pedido.destino}.items() if v})
        if pedido.fecha_ida:
            datos["mes"] = MESES[pedido.fecha_ida.month - 1]
    return CONSULTAS_POR_TEMA.get(tema, CONSULTAS_POR_TEMA["otro"]).format(**datos)


def crear_nodo_aprobacion(memoria: MemoriaBusquedas, base: BaseConocimiento | None = None) -> NodoAsync:
    """Crea el nodo human-in-the-loop conectado a ESTA memoria y ESTA base de guías (las mismas del investigador)."""

    async def aprobacion_web(state: OrquestadorState) -> dict[str, Any]:
        ronda = max(1, state.get("pasos_supervisor", 1))
        pedido = state.get("pedido")
        destino = pedido.destino if pedido is not None else None
        if not destino and base is not None:  # pedido sin destino: el que se mencione en la pregunta (con guía)
            destino = base.destino_mencionado(state.get("pregunta"))
        temas: list[Tema] = list(state.get("temas_pedidos") or [])
        sin_memoria, en_memoria, en_guias = await temas_por_origen(memoria, destino, temas, base)
        previo: PermisoWeb = state.get("permiso_web", "libre")
        turno = turno_actual(state)

        pausa = False
        comentario = ""
        decidido = [r for r in aprobaciones_del_turno(state) if r.origen == "persona"]
        if previo in ("aprobada", "rechazada") and decidido:
            resultado, origen = previo, "decision_previa"  # una persona ya decidió en este turno: no se pregunta de nuevo
        elif not sin_memoria:
            resultado, origen = "sin_pedir", "memoria"
        elif state.get("auto_aprobar", False):
            resultado, origen = "aprobada", "auto_aprobar"
        else:
            solicitud = SolicitudAprobacion(
                ronda=ronda, destino=destino, temas_sin_memoria=sin_memoria, temas_en_memoria=en_memoria,
                temas_en_guias=en_guias,
                consultas_previstas=[consulta_prevista(t, pedido) for t in sin_memoria],
                instruccion_del_supervisor=state.get("instruccion_actual", ""),
            )
            log.info("[aprobacion_web] pausa: se pide permiso para buscar en la web %s", sin_memoria)
            # La primera vez lanza GraphInterrupt (el grafo se detiene). Al reanudar devuelve el resume.
            respuesta = interpretar_respuesta(interrupt(solicitud.model_dump(mode="json")))
            pausa = True
            resultado, origen = ("aprobada" if respuesta.aprobado else "rechazada"), "persona"
            comentario = respuesta.comentario

        registro = RegistroAprobacion(ronda=ronda, turno=turno, temas_sin_memoria=sin_memoria,
                                      temas_en_memoria=en_memoria, temas_en_guias=en_guias, pausa=pausa,
                                      resultado=resultado, origen=origen, comentario=comentario)
        log.info("[aprobacion_web] ronda %d → %s (%s)", ronda, resultado, origen)
        texto = {
            "aprobada": "la búsqueda web está autorizada",
            "rechazada": "la búsqueda web NO está autorizada: el investigador trabaja solo con la memoria",
            "sin_pedir": "las guías y la memoria cubren los temas pedidos: no hace falta buscar en la web",
        }[resultado]
        return validar_actualizacion("aprobacion_web", {
            "permiso_web": resultado,
            "aprobaciones": [registro],
            "messages": [AIMessage(content=f"[aprobacion_web] ronda {ronda}: {texto} ({origen}).", name="aprobacion_web")],
        })

    return aprobacion_web
