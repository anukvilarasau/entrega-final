"""
rag/documentos.py — Los modelos del RAG y la lectura del corpus (data/guias/*.md).

- `DocumentoGuia`: una guía completa, con los metadatos del encabezado YAML
  (destino, URL, revisión, licencia, atribución) y un hash de su contenido.
- `Fragmento`: un chunk indexado. Sus metadatos (destino, sección, URL, licencia)
  son lo que después se cita como fuente: las fuentes salen de acá, nunca del LLM.
- `FragmentoRecuperado`: un fragmento devuelto por el retriever, con el puntaje de
  cada etapa (vectorial, BM25, RRF, re-ranking) para la traza y la evaluación.

Todos son modelos Pydantic: lo que entra al índice y lo que sale del retriever
está validado.
"""

import hashlib
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

PATRON_ID_FRAGMENTO: re.Pattern[str] = re.compile(r"^guia:[a-z0-9-]+#\d{3}$")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin espacios de más ('Córdoba ' → 'cordoba')."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", sin_tildes.lower()).strip()


class DocumentoGuia(BaseModel):
    """Una guía de viaje del corpus (un archivo data/guias/<slug>.md)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str = Field(pattern=r"^[a-z0-9-]+$")
    destino: str = Field(min_length=2)
    titulo: str
    alias: list[str] = Field(default_factory=list)
    url: str = Field(pattern=r"^https://")
    revision: int | None = None
    licencia: str
    licencia_url: str = ""
    atribucion: str
    texto: str = Field(min_length=50, description="El markdown sin el encabezado YAML.")

    @property
    def hash(self) -> str:
        """Cambia si cambia el texto o un metadato que va al índice."""
        base = "|".join([self.slug, self.destino, self.url, str(self.revision), self.licencia, self.texto])
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

    def nombres(self) -> list[str]:
        """Todas las formas de nombrar el destino (para resolver lo que escribe el usuario o el LLM)."""
        return [self.destino, self.titulo, self.slug.replace("-", " "), *self.alias]


class Fragmento(BaseModel):
    """Un chunk del índice con sus metadatos (lo que se cita como fuente)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=PATRON_ID_FRAGMENTO.pattern, description="'guia:<slug>#<nnn>'")
    slug: str
    destino: str
    seccion: str = Field(description="Ruta de la sección: 'Hacer > Senderismo'.")
    texto: str = Field(min_length=1)
    url: str
    titulo: str
    licencia: str
    orden: int = Field(ge=0)
    tokens: int = Field(ge=1)
    doc_hash: str = Field(description="Hash del documento (+ parámetros de chunking y modelo) que lo generó.")

    def para_embedding(self) -> str:
        """Lo que se embebe y se indexa en BM25: el contexto (destino y sección) + el texto."""
        return f"{self.destino} — {self.seccion}\n{self.texto}"

    def metadatos(self) -> dict[str, str | int]:
        """Metadatos planos para el vector store (Chroma y Pinecone solo aceptan escalares)."""
        return self.model_dump(exclude={"id"})


class FragmentoRecuperado(BaseModel):
    """Un fragmento devuelto por el retriever, con la traza de cómo llegó al top-k."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fragmento: Fragmento
    puesto_vectorial: int | None = Field(default=None, description="Posición en la búsqueda vectorial (1 = mejor).")
    puesto_bm25: int | None = Field(default=None, description="Posición en BM25 (1 = mejor).")
    puntaje_rrf: float = 0.0
    puntaje_rerank: float | None = None


def leer_frontmatter(texto: str) -> tuple[dict[str, object], str]:
    """Encabezado YAML simple ('clave: valor' y listas '[a, b]') → (dict, cuerpo). Sin dependencias."""
    if not texto.startswith("---\n"):
        raise ValueError("La guía no tiene encabezado YAML (---)")
    cabecera, _, cuerpo = texto[4:].partition("\n---\n")
    datos: dict[str, object] = {}
    for linea in cabecera.splitlines():
        clave, _, valor = linea.partition(":")
        valor = valor.strip()
        if valor.startswith("[") and valor.endswith("]"):
            datos[clave.strip()] = [v.strip().strip('"') for v in valor[1:-1].split(",") if v.strip().strip('"')]
        else:
            datos[clave.strip()] = valor.strip('"')
    return datos, cuerpo.strip()


def leer_guia(ruta: Path) -> DocumentoGuia:
    datos, cuerpo = leer_frontmatter(ruta.read_text(encoding="utf-8"))
    revision = datos.get("revision")
    return DocumentoGuia(
        slug=str(datos["slug"]), destino=str(datos["destino"]), titulo=str(datos.get("titulo", datos["destino"])),
        alias=list(datos.get("alias") or []), url=str(datos["url"]),  # type: ignore[arg-type]
        revision=int(revision) if revision else None, licencia=str(datos["licencia"]),
        licencia_url=str(datos.get("licencia_url", "")), atribucion=str(datos["atribucion"]), texto=cuerpo,
    )


def leer_corpus(carpeta: Path) -> list[DocumentoGuia]:
    """Todas las guías de la carpeta (los archivos que empiezan con '_' son documentación, no guías)."""
    guias = [leer_guia(r) for r in sorted(carpeta.glob("*.md")) if not r.name.startswith("_")]
    if not guias:
        raise FileNotFoundError(f"No hay guías en {carpeta} (correr scripts/descargar_guias.py)")
    return guias
