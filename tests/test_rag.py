"""Tests del RAG (rag/): conversión del corpus, chunking por tokens, ingesta idempotente, retriever híbrido,
filtro por destino, re-rankers, Pinecone (con un índice falso), la herramienta consultar_guias y el HITL con guías.
Todo offline: embeddings por hashing, Chroma en carpetas temporales, re-ranker léxico."""

import builtins
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import config
from agents.common import SalidaHerramienta
from agents.llm import LLMMock
from agents.research_tools import BuscadorMock, crear_herramientas_investigador
from graph.hitl import temas_por_origen
from rag.base import BaseConocimiento, crear_base_conocimiento
from rag.chunking import TokenizadorSimple, fragmentar, secciones, ventanas
from rag.documentos import Fragmento, leer_corpus, leer_guia
from rag.embeddings import EmbeddingsHash
from rag.evaluate import GOLDEN_SET, MODOS, PreguntaGolden, evaluar_modo
from rag.ingesta import ingestar
from rag.memoria import EmbeddingHash, MemoriaBusquedas
from rag.reranker import PuntajesRelevancia, RerankerLLM, RerankerSolapamiento, SinReranker
from rag.retriever import fusion_rrf
from rag.vectorstore import ChromaVectorStore, PineconeVectorStore
from rag.wikitexto import wikitexto_a_markdown

TOK = TokenizadorSimple()


# ---------------------------------------------------------------------------
# Corpus: wikitexto → markdown y lectura con metadatos
# ---------------------------------------------------------------------------
def test_wikitexto_conserva_fichas_clima_y_montos() -> None:
    wiki = """{{pagebanner|Banner.jpg}}
== Hacer ==
[[Archivo:Foto.jpg|thumb|Una [[foto]] linda]]
* {{hacer
| name=Cerro Catedral | alt=Esquí | url=http://x.example | address=Ruta 82
| price={{ARS|1680}} por día | content=El centro de [[esquí]] más grande de [[Sudamérica|América del Sur]].
}}
Más info en [http://otro.example la web oficial].<ref>nota</ref>

== Comprar ==

== Clima ==
{{Clima|enemax=21.5|enemin=6.4|eneprecip=22.2|julmax=6|julmin=-1}}
'''Importante''': tarifa de {{USD|50}}.
"""
    md = wikitexto_a_markdown(wiki)
    assert "## Hacer" in md and "Banner" not in md and "Foto.jpg" not in md and "thumb" not in md
    assert ("- **Cerro Catedral** (Esquí): El centro de esquí más grande de América del Sur. "
            "Dirección: Ruta 82. Precio: AR$ 1680 por día.") in md
    assert "la web oficial" in md and "http://otro" not in md and "nota" not in md
    assert "## Comprar" not in md  # sección vacía
    assert "| Enero | 21.5 | 6.4 | 22.2 |" in md and "| Julio | 6 | -1 | - |" in md and "|---|---|---|---|" in md
    assert "**Importante**: tarifa de US$ 50." in md


def test_el_corpus_tiene_las_9_guias_con_url_y_licencia() -> None:
    corpus = leer_corpus(config.GUIAS_DIR)
    assert {d.slug for d in corpus} == {"bariloche", "mendoza", "salta", "ushuaia", "puerto-madryn", "mar-del-plata",
                                        "cordoba", "iguazu", "el-calafate"}
    for d in corpus:
        assert d.url.startswith("https://es.wikivoyage.org/wiki/") and d.licencia == "CC BY-SA 4.0"
        assert "Wikivoyage" in d.atribucion and d.revision and len(d.texto) > 5000
    assert (config.GUIAS_DIR / "_ATRIBUCION.md").exists()  # documentación, no se indexa


# ---------------------------------------------------------------------------
# Chunking por tokens
# ---------------------------------------------------------------------------
def test_chunking_respeta_tope_overlap_y_secciones() -> None:
    doc = leer_guia(config.GUIAS_DIR / "cordoba.md")
    frags = fragmentar(doc, TOK, max_tokens=500, overlap=60, huella="t")
    assert len(frags) > 10
    assert all(f.tokens <= 500 for f in frags) and max(f.tokens for f in frags) > 300
    assert [f.orden for f in frags] == list(range(len(frags)))
    assert all(f.id == f"guia:cordoba#{f.orden:03d}" for f in frags)
    # ninguna sección se mezcla y el texto sale del ORIGINAL (no de tokens decodificados)
    rutas = dict(secciones(doc.texto))
    assert all(f.texto in rutas[f.seccion] for f in frags)
    assert all(f.url == doc.url and f.licencia == doc.licencia and f.destino == "Córdoba" for f in frags)


def test_ventanas_con_solapamiento() -> None:
    texto = " ".join(f"palabra{i}." for i in range(1200))
    partes = ventanas(texto, TOK, max_tokens=500, overlap=60)
    assert len(partes) >= 3 and all(n <= 500 for _, n in partes)
    fin_primera = partes[0][0].split()[-3:]
    assert all(p in partes[1][0] for p in fin_primera)  # el final de una ventana se repite al principio de la otra


# ---------------------------------------------------------------------------
# Ingesta idempotente
# ---------------------------------------------------------------------------
@pytest.fixture
def base_temporal(tmp_path: Path) -> BaseConocimiento:
    carpeta = tmp_path / "guias"
    carpeta.mkdir()
    for slug in ("bariloche", "salta", "mendoza"):
        shutil.copy(config.GUIAS_DIR / f"{slug}.md", carpeta / f"{slug}.md")
    return crear_base_conocimiento(modo_mock=True, guias_dir=carpeta, ruta_indice=tmp_path / "indice")


async def test_ingesta_idempotente_reindexa_solo_lo_que_cambio(base_temporal: BaseConocimiento, tmp_path: Path) -> None:
    b = base_temporal
    r1 = await b.asegurar_indice()
    assert sorted(r1.indexados) == ["bariloche", "mendoza", "salta"] and r1.fragmentos_totales > 30
    r2 = await b.asegurar_indice()
    assert r2.indexados == [] and sorted(r2.sin_cambios) == ["bariloche", "mendoza", "salta"]
    assert r2.fragmentos_totales == r1.fragmentos_totales and not r2.hubo_cambios

    # cambia una guía y se borra otra: solo se toca lo que cambió
    salta = tmp_path / "guias" / "salta.md"
    salta.write_text(salta.read_text(encoding="utf-8") + "\n## Nueva sección\nTexto agregado para la prueba de ingesta.\n",
                     encoding="utf-8")
    (tmp_path / "guias" / "mendoza.md").unlink()
    corpus = leer_corpus(tmp_path / "guias")
    r3 = await ingestar(corpus, b.store, b.embeddings, b.embeddings.tokenizador, chunk_tokens=b.chunk_tokens,
                        overlap=b.chunk_overlap)
    assert r3.indexados == ["salta"] and r3.sin_cambios == ["bariloche"] and r3.borrados == ["mendoza"]
    assert set((await b.store.hashes())) == {"bariloche", "salta"}


async def test_cambiar_el_tamano_de_chunk_reindexa_todo(base_temporal: BaseConocimiento) -> None:
    b = base_temporal
    await b.asegurar_indice()
    b.chunk_tokens = 300
    r = await b.asegurar_indice()
    assert sorted(r.indexados) == ["bariloche", "mendoza", "salta"]  # la huella cambió: no quedan chunks viejos
    assert all(f.tokens <= 300 for f in await b.store.todos())


# ---------------------------------------------------------------------------
# Retriever híbrido
# ---------------------------------------------------------------------------
@pytest.fixture
async def base_mock() -> BaseConocimiento:
    b = crear_base_conocimiento(modo_mock=True)
    await b.asegurar_indice()
    return b


async def test_filtro_por_destino_sale_de_los_metadatos(base_mock: BaseConocimiento) -> None:
    r = await base_mock.consultar("qué hacer y dónde comer", "San Carlos de Bariloche", k=6)
    assert r.slug == "bariloche" and len(r.fragmentos) == 6
    assert all(x.fragmento.slug == "bariloche" for x in r.fragmentos)
    assert r.candidatos_bm25 > 0 and r.candidatos_vectoriales > 0 and r.reranker == "solapamiento-lexico"


async def test_destino_sin_guia(base_mock: BaseConocimiento) -> None:
    r = await base_mock.consultar("qué hacer", "Tilcara")
    assert r.destino_sin_guia and r.fragmentos == []


def test_resolver_destino_por_alias(base_mock: BaseConocimiento) -> None:
    b = base_mock
    assert b.resolver_destino("Cataratas") == "iguazu" and b.resolver_destino("puerto iguazú") == "iguazu"
    assert b.resolver_destino("CÓRDOBA") == "cordoba" and b.resolver_destino("Bariloche, Río Negro") == "bariloche"
    assert b.resolver_destino("Puerto Madryn") == "puerto-madryn" and b.resolver_destino("Rosario") is None
    assert not b.tiene_guia(None)


async def test_bm25_encuentra_nombres_propios(base_mock: BaseConocimiento) -> None:
    hits = await base_mock.retriever.buscar("Garganta del Diablo", slug="iguazu", k=3, modo="bm25", usar_reranker=False)
    assert hits and "Garganta del Diablo" in hits[0].fragmento.texto and hits[0].puesto_bm25 == 1


def _frag(i: str) -> Fragmento:
    return Fragmento(id=f"guia:x#{i}", slug="x", destino="X", seccion="S", texto=i, url="https://x.example",
                     titulo="X", licencia="L", orden=0, tokens=1, doc_hash="h")


def test_fusion_rrf() -> None:
    a, b, c = _frag("001"), _frag("002"), _frag("003")
    fusion = fusion_rrf([[a, b], [b, c]], k_rrf=60)
    assert [f.id for f, _, _ in fusion] == [b.id, a.id, c.id]  # b está en los dos rankings
    assert fusion[0][2] == [2, 1] and fusion[2][2] == [None, 2]
    assert fusion[0][1] == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)


async def test_rerankers() -> None:
    frags = [_frag("001").model_copy(update={"texto": "playas y mar"}),
             _frag("002").model_copy(update={"texto": "esquí en el cerro con nieve"})]
    assert await SinReranker().puntuar("esquí", frags) == [0.0, -1.0]
    lex = await RerankerSolapamiento().puntuar("esquí nieve", frags)
    assert lex[1] > lex[0]

    def politica(_m: Any, _s: Any) -> PuntajesRelevancia:
        return PuntajesRelevancia.model_validate({"puntajes": [{"n": 1, "relevancia": 2}, {"n": 2, "relevancia": 9}]})

    llm = RerankerLLM(LLMMock(politica_estructurada=politica))
    puntajes = await llm.puntuar("esquí", frags)
    assert puntajes[1] > puntajes[0]


async def test_evaluate_en_mock_calcula_metricas(base_mock: BaseConocimiento) -> None:
    import json

    preguntas = [PreguntaGolden.model_validate(x) for x in json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["preguntas"]]
    assert len(preguntas) >= 10
    todos = await base_mock.store.todos()
    assert all(any(q.es_relevante(f) for f in todos) for q in preguntas)  # toda pregunta tiene relevantes
    m, filas = await evaluar_modo(base_mock, preguntas, "hibrido_rrf_rerank", 4, todos)
    assert set(MODOS) == {"vectorial", "bm25", "hibrido_rrf", "hibrido_rrf_rerank"}
    assert 0 <= m.precision_frag_k <= 1 and m.recall_doc_k >= 0.8 and len(filas) == len(preguntas)


# ---------------------------------------------------------------------------
# Pinecone detrás de la misma interfaz (índice falso, sin red)
# ---------------------------------------------------------------------------
class IndiceFalso:
    def __init__(self) -> None:
        self.datos: dict[str, dict[str, Any]] = {}

    async def upsert(self, vectors: list[dict[str, Any]], namespace: str) -> None:
        for v in vectors:
            self.datos[v["id"]] = v

    async def list(self, prefix: str, namespace: str):  # noqa: A003, ANN201
        yield [i for i in self.datos if i.startswith(prefix)]

    async def fetch(self, ids: builtins.list[str], namespace: str) -> Any:
        return SimpleNamespace(vectors={i: SimpleNamespace(metadata=self.datos[i]["metadata"]) for i in ids
                                        if i in self.datos})

    async def delete(self, ids: builtins.list[str], namespace: str) -> None:
        for i in ids:
            self.datos.pop(i, None)

    async def query(self, vector: builtins.list[float], top_k: int, namespace: str, include_metadata: bool,
                    filter: dict | None) -> Any:  # noqa: A002
        slug = (filter or {}).get("slug", {}).get("$eq")
        hits = [v for v in self.datos.values() if slug is None or v["metadata"]["slug"] == slug]
        hits.sort(key=lambda v: -sum(a * b for a, b in zip(v["values"], vector, strict=True)))
        return SimpleNamespace(matches=[SimpleNamespace(id=v["id"], score=0.5, metadata=v["metadata"])
                                        for v in hits[:top_k]])

    async def describe_index_stats(self) -> dict[str, Any]:
        return {"namespaces": {"guias": {"vector_count": len(self.datos)}}}


async def test_pinecone_cumple_la_misma_interfaz(tmp_path: Path) -> None:
    store = PineconeVectorStore(api_key="x", indice="i", namespace="guias", dimension=256, cloud="aws",
                                region="us-east-1", cliente_indice=IndiceFalso())
    carpeta = tmp_path / "g"
    carpeta.mkdir()
    shutil.copy(config.GUIAS_DIR / "ushuaia.md", carpeta / "ushuaia.md")
    base = BaseConocimiento(leer_corpus(carpeta), store, EmbeddingsHash(), SinReranker())
    r1 = await base.asegurar_indice()
    assert r1.indexados == ["ushuaia"] and r1.fragmentos_totales == await store.contar() > 5
    assert (await base.asegurar_indice()).indexados == []  # idempotente también en Pinecone
    r = await base.consultar("qué hacer en invierno", "Ushuaia")
    assert r.fragmentos and all(x.fragmento.slug == "ushuaia" for x in r.fragmentos)


def test_la_fabrica_elige_el_store_por_variable_de_entorno(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rag.base import crear_vector_store

    assert isinstance(crear_vector_store(256, tmp_path / "c"), ChromaVectorStore)
    monkeypatch.setattr(config, "VECTOR_STORE", "pinecone")
    assert isinstance(crear_vector_store(256), PineconeVectorStore)


# ---------------------------------------------------------------------------
# La herramienta consultar_guias del investigador
# ---------------------------------------------------------------------------
@pytest.fixture
def herramientas(base_mock: BaseConocimiento, tmp_path: Path) -> dict[str, Any]:
    memoria = MemoriaBusquedas(ruta=tmp_path / "mem", embedding=EmbeddingHash())
    return {t.name: t for t in crear_herramientas_investigador(memoria, BuscadorMock(), base_mock)}


def _llamar(tool: Any, **args: Any) -> dict[str, Any]:
    return {"type": "tool_call", "id": "c1", "name": tool.name, "args": args}


async def test_consultar_guias_devuelve_fuentes_desde_los_metadatos(herramientas: dict[str, Any]) -> None:
    assert list(herramientas) == ["consultar_guias", "buscar_en_memoria", "buscar_en_web"]
    msg = await herramientas["consultar_guias"].ainvoke(
        _llamar(herramientas["consultar_guias"], consulta="ver ballenas", destino="Puerto Madryn", tema="actividades"))
    assert str(msg.content).startswith("GUÍAS (Wikivoyage") and "[guia:puerto-madryn#" in str(msg.content)
    assert msg.artifact and all(f.origen == "guia" and f.url == "https://es.wikivoyage.org/wiki/Puerto_Madryn"
                                and f.licencia == "CC BY-SA 4.0" and f.seccion and f.tema == "actividades"
                                for f in msg.artifact)
    SalidaHerramienta(texto=str(msg.content), fuentes=msg.artifact)  # la salida es un modelo válido


async def test_consultar_guias_sin_guia_y_args_invalidos(herramientas: dict[str, Any]) -> None:
    t = herramientas["consultar_guias"]
    msg = await t.ainvoke(_llamar(t, consulta="qué hacer", destino="Tilcara", tema="actividades"))
    assert str(msg.content).startswith("SIN GUÍA") and msg.artifact == []
    with pytest.raises(Exception):  # args_schema Pydantic: tema fuera del Literal
        await t.ainvoke(_llamar(t, consulta="qué hacer", destino="Salta", tema="inventado"))


async def test_la_web_no_se_usa_para_temas_estables_con_guia(herramientas: dict[str, Any]) -> None:
    t = herramientas["buscar_en_web"]
    msg = await t.ainvoke(_llamar(t, consulta="qué hacer en Salta", destino="Salta", tema="actividades"))
    assert str(msg.content).startswith("NO SE BUSCÓ EN LA WEB") and "consultar_guias" in str(msg.content)
    msg = await t.ainvoke(_llamar(t, consulta="precio pasajes Mendoza Bariloche", destino="Bariloche", tema="pasajes"))
    assert str(msg.content).startswith("WEB (Tavily)")  # lo volátil sí va a la web


async def test_el_hitl_no_pausa_por_temas_que_responden_las_guias(base_mock: BaseConocimiento, tmp_path: Path) -> None:
    memoria = MemoriaBusquedas(ruta=tmp_path / "mem", embedding=EmbeddingHash())
    sin, con, guias = await temas_por_origen(memoria, "Salta", ["actividades", "clima", "pasajes"], base_mock)
    assert guias == ["actividades", "clima"] and sin == ["pasajes"] and con == []
    sin, _, guias = await temas_por_origen(memoria, "Tilcara", ["actividades"], base_mock)
    assert sin == ["actividades"] and guias == []  # sin guía del destino, lo estable también iría a la web


def test_el_reranker_llm_no_se_usa_en_mock() -> None:
    b = crear_base_conocimiento(modo_mock=True)
    assert isinstance(b.reranker, RerankerSolapamiento) and isinstance(b.embeddings, EmbeddingsHash)

