"""
scripts/descargar_guias.py — Descarga las guías de viaje de Wikivoyage en español a data/guias/.

Se corre UNA vez (o cuando se quiera actualizar el corpus); el resultado queda
versionado en el repo, así la ingesta (rag/ingesta.py) es reproducible SIN internet.

    python scripts/descargar_guias.py            # descarga las que faltan
    python scripts/descargar_guias.py --todas    # vuelve a descargar todas

Qué hace con cada artículo:
1. Pide a la API de MediaWiki el WIKITEXTO de la revisión actual (con su número).
2. Lo convierte a markdown con rag/wikitexto.py: conserva las fichas de lugares
   ({{ver}}, {{comer}}, {{dormir}}...) y la tabla {{Clima}}, que el "texto plano"
   de la API descarta; saca imágenes, mapas y plantillas de formato. No cambia el
   contenido.
3. Lo guarda como data/guias/<slug>.md con un encabezado YAML: destino, URL,
   revisión, fecha, licencia (CC BY-SA 4.0) y la atribución completa.

Licencia: el texto de Wikivoyage es CC BY-SA 4.0. Por eso cada archivo lleva su
URL, el enlace al historial (los autores) y la licencia, y data/guias/_ATRIBUCION.md
las lista todas. Las fuentes que cita el asesor salen de esos metadatos.

Async: cada pedido HTTP corre en un hilo con asyncio.to_thread (urllib de la
biblioteca estándar). No se usa httpx porque el filtro anti-bots de Wikimedia le
responde 403 aun con el mismo User-Agent con el que urllib y curl reciben 200
(probado el 24/11/2026). Es un script de descarga única, fuera del camino de la API.
Wikivoyage limita la frecuencia de pedidos (429): se espera entre artículos y se
reintenta con backoff.
"""

import argparse
import asyncio
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from rag.wikitexto import wikitexto_a_markdown  # noqa: E402

API = "https://es.wikivoyage.org/w/api.php"
# Cliente genérico, sin datos personales (política de User-Agent de Wikimedia).
USER_AGENT = "asesor-viajes-ingesta/1.0 (proyecto educativo)"
LICENCIA = "CC BY-SA 4.0"
LICENCIA_URL = "https://creativecommons.org/licenses/by-sa/4.0/deed.es"
PAUSA_S = 4.0


class FuenteGuia(BaseModel):
    """Un artículo de Wikivoyage que forma parte del corpus."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(pattern=r"^[a-z0-9-]+$")
    destino: str
    titulo: str = Field(description="Título exacto del artículo en es.wikivoyage.org")
    alias: list[str] = Field(default_factory=list, description="Otros nombres con los que se pregunta por el destino.")


CORPUS: list[FuenteGuia] = [
    FuenteGuia(slug="bariloche", destino="Bariloche", titulo="San Carlos de Bariloche",
               alias=["San Carlos de Bariloche"]),
    FuenteGuia(slug="mendoza", destino="Mendoza", titulo="Mendoza"),
    FuenteGuia(slug="salta", destino="Salta", titulo="Salta"),
    FuenteGuia(slug="ushuaia", destino="Ushuaia", titulo="Ushuaia"),
    FuenteGuia(slug="puerto-madryn", destino="Puerto Madryn", titulo="Puerto Madryn", alias=["Madryn"]),
    FuenteGuia(slug="mar-del-plata", destino="Mar del Plata", titulo="Mar del Plata", alias=["Mardel"]),
    FuenteGuia(slug="cordoba", destino="Córdoba", titulo="Córdoba (Argentina)", alias=["Córdoba capital"]),
    FuenteGuia(slug="iguazu", destino="Iguazú", titulo="Cataratas del Iguazú",
               alias=["Puerto Iguazú", "Cataratas del Iguazú", "Cataratas"]),
    FuenteGuia(slug="el-calafate", destino="El Calafate", titulo="El Calafate", alias=["Calafate"]),
]


def url_articulo(titulo: str) -> str:
    return "https://es.wikivoyage.org/wiki/" + quote(titulo.replace(" ", "_"))




def con_encabezado(f: FuenteGuia, cuerpo: str, revision: int, fecha_revision: str) -> str:
    url = url_articulo(f.titulo)
    atribucion = (f"Texto de Wikivoyage en español, artículo «{f.titulo}» ({url}), revisión {revision}. "
                  f"Autores: colaboradores de Wikivoyage ({url}?action=history). Licencia {LICENCIA} "
                  f"({LICENCIA_URL}). Convertido a markdown y limpiado de formato; sin cambios de contenido.")
    alias = ", ".join(f'"{a}"' for a in f.alias)
    return (
        "---\n"
        f"destino: {f.destino}\n"
        f"slug: {f.slug}\n"
        f"titulo: {f.titulo}\n"
        f"alias: [{alias}]\n"
        f"url: {url}\n"
        f"revision: {revision}\n"
        f"fecha_revision: {fecha_revision}\n"
        f"fecha_descarga: {datetime.now(UTC).date().isoformat()}\n"
        f"licencia: {LICENCIA}\n"
        f"licencia_url: {LICENCIA_URL}\n"
        f'atribucion: "{atribucion}"\n'
        "---\n\n"
        f"# {f.titulo}\n\n{cuerpo}"
    )


def _get_json(params: dict[str, str | int]) -> dict:
    pedido = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(pedido, timeout=30) as r:
        return json.load(r)


async def pedir(params: dict[str, str | int]) -> dict:
    for intento in range(6):
        try:
            return await asyncio.to_thread(_get_json, params)  # urllib es sync: fuera del event loop
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            espera = 10 * (intento + 1)
            print(f"  429 (demasiados pedidos): espero {espera} s")
            await asyncio.sleep(espera)
    raise RuntimeError("Wikivoyage sigue respondiendo 429: probá más tarde.")


async def descargar(f: FuenteGuia, carpeta: Path) -> Path:
    datos = await pedir({
        "action": "query", "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main",
        "titles": f.titulo, "redirects": 1, "format": "json", "formatversion": 2,
    })
    pagina = datos["query"]["pages"][0]
    if pagina.get("missing"):
        raise RuntimeError(f"No existe el artículo «{f.titulo}» en es.wikivoyage.org")
    rev = pagina["revisions"][0]
    destino = carpeta / f"{f.slug}.md"
    destino.write_text(con_encabezado(f, wikitexto_a_markdown(rev["slots"]["main"]["content"]), rev["revid"],
                                      rev["timestamp"][:10]),
                       encoding="utf-8", newline="\n")
    return destino


def escribir_atribucion(carpeta: Path) -> None:
    filas = ["# Atribución del corpus de guías", "",
             f"Todas las guías de esta carpeta son texto de **Wikivoyage en español** bajo licencia "
             f"**{LICENCIA}** ({LICENCIA_URL}). Se convirtieron a markdown y se limpiaron de formato, sin cambios "
             "de contenido. La revisión exacta de cada artículo está en su encabezado YAML. Los autores son los "
             "colaboradores de Wikivoyage que figuran en el historial de cada artículo.", "",
             "| Archivo | Artículo | Revisión | Historial (autores) |", "|---|---|---|---|"]
    for f in CORPUS:
        archivo = carpeta / f"{f.slug}.md"
        if not archivo.exists():
            continue
        rev = re.search(r"^revision: (\d+)$", archivo.read_text(encoding="utf-8"), re.M)
        url = url_articulo(f.titulo)
        filas.append(f"| `{f.slug}.md` | [{f.titulo}]({url}) | {rev.group(1) if rev else '?'} | "
                     f"[historial]({url}?action=history) |")
    (carpeta / "_ATRIBUCION.md").write_text("\n".join(filas) + "\n", encoding="utf-8", newline="\n")


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--todas", action="store_true", help="Vuelve a descargar aunque el archivo exista.")
    args = p.parse_args()
    carpeta = config.GUIAS_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(CORPUS):
        if (carpeta / f"{f.slug}.md").exists() and not args.todas:
            print(f"= {f.slug}.md ya existe")
            continue
        if i:
            await asyncio.sleep(PAUSA_S)
        ruta = await descargar(f, carpeta)
        print(f"+ {ruta.name}: {len(ruta.read_text(encoding='utf-8')):,} caracteres")
    escribir_atribucion(carpeta)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
