"""
rag/wikitexto.py — Wikitext de Wikivoyage → markdown limpio (para el corpus de guías).

Por qué no alcanza con el "texto plano" de la API (TextExtracts): descarta las
FICHAS de lugares ({{ver|name=Cerro Catedral|content=...}}, {{comer|...}},
{{dormir|...}}), que son justamente lo más útil de una guía, y la tabla
{{Clima}} con los promedios mensuales. Este conversor las conserva:

- fichas (ver/see, hacer/do, comer/eat, beber/drink, dormir/sleep, comprar/buy,
  ir/go, listing/listado) → "- **Nombre** (alt): descripción. Dirección: ... Precio: ...";
- {{Clima}} → tabla markdown "Mes | Máx | Mín | Lluvia";
- {{ARS|1680}} → "AR$ 1680", {{USD|50}} → "US$ 50", {{km|80}} → "80 km", {{IATA|BRC}} → "(IATA: BRC)";
- marcadores → su nombre; el resto de las plantillas (banners, mapas, estado del artículo) se descarta;
- enlaces internos → su texto; enlaces externos → su texto; imágenes, categorías y referencias → fuera;
- títulos "== X ==" → "## X"; listas "*" → "- ".

Es puro (sin red ni archivos): se prueba con tests/test_rag.py.
"""

import re

FICHAS: frozenset[str] = frozenset({
    "ver", "see", "hacer", "do", "comer", "eat", "beber", "drink", "dormir", "sleep", "comprar", "buy",
    "ir", "go", "listing", "listado",
})
MESES_CLIMA: tuple[tuple[str, str], ...] = (
    ("ene", "Enero"), ("feb", "Febrero"), ("mar", "Marzo"), ("abr", "Abril"), ("may", "Mayo"), ("jun", "Junio"),
    ("jul", "Julio"), ("ago", "Agosto"), ("sep", "Septiembre"), ("oct", "Octubre"), ("nov", "Noviembre"),
    ("dic", "Diciembre"),
)
PREFIJOS_A_DESCARTAR: tuple[str, ...] = ("archivo:", "file:", "imagen:", "image:", "categoría:", "category:")


# ---------------------------------------------------------------------------
# Plantillas {{...}} (con anidamiento)
# ---------------------------------------------------------------------------
def _cierre(texto: str, inicio: int, abre: str, cierra: str) -> int:
    """Índice justo después del `cierra` que balancea el `abre` en `inicio` (-1 si no cierra)."""
    profundidad = 0
    i = inicio
    while i < len(texto):
        if texto.startswith(abre, i):
            profundidad += 1
            i += len(abre)
        elif texto.startswith(cierra, i):
            profundidad -= 1
            i += len(cierra)
            if profundidad == 0:
                return i
        else:
            i += 1
    return -1


def _partir_parametros(interior: str) -> list[str]:
    """Separa por "|" de primer nivel (no dentro de {{...}} ni [[...]])."""
    partes: list[str] = []
    actual: list[str] = []
    llaves = corchetes = 0
    i = 0
    while i < len(interior):
        dos = interior[i:i + 2]
        if dos == "{{":
            llaves += 1
        elif dos == "}}":
            llaves -= 1
        elif dos == "[[":
            corchetes += 1
        elif dos == "]]":
            corchetes -= 1
        if dos in ("{{", "}}", "[[", "]]"):
            actual.append(dos)
            i += 2
            continue
        if interior[i] == "|" and llaves == 0 and corchetes == 0:
            partes.append("".join(actual))
            actual = []
        else:
            actual.append(interior[i])
        i += 1
    partes.append("".join(actual))
    return partes


def _parametros(partes: list[str]) -> tuple[dict[str, str], list[str]]:
    nombrados: dict[str, str] = {}
    posicionales: list[str] = []
    for p in partes:
        clave, igual, valor = p.partition("=")
        if igual and re.fullmatch(r"\s*[\w áéíóúñ-]+\s*", clave):
            nombrados[clave.strip().lower()] = valor.strip()
        else:
            posicionales.append(p.strip())
    return nombrados, posicionales


def _primero(p: dict[str, str], *claves: str) -> str:
    for c in claves:
        if p.get(c, "").strip():
            return p[c].strip()
    return ""


def _ficha(p: dict[str, str]) -> str:
    nombre = _primero(p, "name", "nombre")
    if not nombre:
        return ""
    alt = _primero(p, "alt")
    partes = [f"**{nombre}**" + (f" ({alt})" if alt else "")]
    descripcion = _primero(p, "content", "contenido", "descripción", "descripcion")
    detalles = []
    for etiqueta, claves in (("Dirección", ("address", "dirección", "direccion")),
                             ("Cómo llegar", ("directions", "indicaciones")),
                             ("Horario", ("hours", "horario")), ("Precio", ("price", "precio"))):
        if valor := _primero(p, *claves):
            detalles.append(f"{etiqueta}: {valor.rstrip('.')}.")
    texto = partes[0] + (f": {descripcion.strip()}" if descripcion else "")
    if detalles:
        texto += " " + " ".join(detalles)
    return "\n- " + texto + "\n"


def _clima(p: dict[str, str]) -> str:
    filas = []
    for clave, mes in MESES_CLIMA:
        maxima, minima, lluvia = p.get(f"{clave}max", ""), p.get(f"{clave}min", ""), p.get(f"{clave}precip", "")
        if maxima or minima or lluvia:
            filas.append(f"| {mes} | {maxima or '-'} | {minima or '-'} | {lluvia or '-'} |")
    if not filas:
        return ""
    return ("\n\nClima típico (promedios mensuales):\n\n| Mes | Máxima (°C) | Mínima (°C) | Lluvia (mm) |\n"
            "|---|---|---|---|\n" + "\n".join(filas) + "\n\n")


def _plantilla(interior: str) -> str:
    partes = _partir_parametros(interior)
    nombre = partes[0].strip().lower()
    nombrados, posicionales = _parametros(partes[1:])
    nombrados = {k: expandir_plantillas(v) for k, v in nombrados.items()}
    posicionales = [expandir_plantillas(v) for v in posicionales]
    primero = posicionales[0] if posicionales else ""
    if nombre in FICHAS:
        return _ficha(nombrados)
    if nombre == "clima":
        return _clima(nombrados)
    if nombre in ("marker", "marcador"):
        return _primero(nombrados, "name", "nombre")
    if nombre == "ars":
        return f"AR$ {primero}" if primero else ""
    if nombre == "usd":
        return f"US$ {primero}" if primero else ""
    if nombre == "km":
        return f"{primero} km" if primero else ""
    if nombre == "iata":
        return f"(IATA: {primero})" if primero else ""
    return ""  # banners, mapas, cajas de navegación, "estado del artículo"...


def expandir_plantillas(texto: str) -> str:
    salida: list[str] = []
    i = 0
    while (j := texto.find("{{", i)) != -1:
        fin = _cierre(texto, j, "{{", "}}")
        if fin == -1:
            break
        salida.append(texto[i:j])
        salida.append(_plantilla(texto[j + 2:fin - 2]))
        i = fin
    salida.append(texto[i:])
    return "".join(salida)


# ---------------------------------------------------------------------------
# Enlaces, formato y estructura
# ---------------------------------------------------------------------------
def _enlaces_internos(texto: str) -> str:
    salida: list[str] = []
    i = 0
    while (j := texto.find("[[", i)) != -1:
        fin = _cierre(texto, j, "[[", "]]")
        if fin == -1:
            break
        salida.append(texto[i:j])
        interior = texto[j + 2:fin - 2]
        destino, _, etiqueta = interior.partition("|")
        d = destino.strip().lower()
        if d.startswith(PREFIJOS_A_DESCARTAR) or re.match(r"^[a-z]{2,3}:", d):  # imágenes, categorías, interwikis
            pass
        else:
            salida.append(_enlaces_internos(etiqueta or destino).strip())
        i = fin
    salida.append(texto[i:])
    return "".join(salida)


def wikitexto_a_markdown(wikitexto: str) -> str:
    t = wikitexto.replace("\r", "")
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"__[A-Z]+__", "", t)
    t = expandir_plantillas(t)
    t = _enlaces_internos(t)
    t = re.sub(r"\[(?:https?:)?//[^\s\]]+\s+([^\]]+)\]", r"\1", t)   # [url texto] → texto
    t = re.sub(r"\[(?:https?:)?//[^\s\]]+\]", "", t)                 # [url] → nada
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</?[a-zA-Z][^>]*>", "", t)                           # resto de las etiquetas HTML
    t = t.replace("'''", "**").replace("''", "")
    t = t.replace("&nbsp;", " ").replace("&ndash;", "–").replace("&mdash;", "—")

    lineas: list[str] = []
    for linea in t.split("\n"):
        linea = re.sub(r"[ \t ​]+", " ", linea).strip()
        if m := re.fullmatch(r"(={2,5})\s*(.+?)\s*\1", linea):
            linea = "#" * len(m.group(1)) + " " + m.group(2).strip("*").strip()
        elif m := re.match(r"^([*#]+)\s*(.*)$", linea):
            nivel = len(m.group(1)) - 1
            linea = ("  " * nivel + "- " + m.group(2)) if m.group(2) else ""
        elif linea.startswith(":"):
            linea = linea.lstrip(":").strip()
        elif re.match(r"^(\{\||\|\}|\|-(?!-)|!)", linea):  # restos de tablas wiki (no "|---|" de markdown)
            linea = ""
        lineas.append(linea)
    md = "\n".join(lineas)
    md = re.sub(r"(?m)^- \s*$", "", md)
    md = re.sub(r"(?m)^(- .*)\n\n(?=- )", r"\1\n", md)                 # las fichas seguidas forman una lista
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return quitar_secciones_vacias(md) + "\n"


def quitar_secciones_vacias(md: str) -> str:
    """Saca los títulos sin contenido hasta el próximo título de igual o mayor nivel."""
    lineas = md.split("\n")
    salida: list[str] = []
    for i, linea in enumerate(lineas):
        m = re.match(r"^(#{2,6}) ", linea)
        if m:
            nivel = len(m.group(1))
            vacia = True
            for siguiente in lineas[i + 1:]:
                ms = re.match(r"^(#{2,6}) ", siguiente)
                if ms and len(ms.group(1)) <= nivel:
                    break
                if siguiente.strip() and not ms:
                    vacia = False
                    break
            if vacia:
                continue
        salida.append(linea)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(salida)).strip()
