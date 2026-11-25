"""
rag/chunking.py — Limpieza + chunking POR TOKENS, respetando las secciones de la guía.

Cómo se corta una guía:
1. Se parte por secciones ("## Hacer", "### Senderismo"): un fragmento nunca mezcla
   dos secciones, así cada uno tiene una sección precisa en sus metadatos.
2. Dentro de cada sección, ventanas de hasta CHUNK_TOKENS tokens (500 por defecto)
   con CHUNK_OVERLAP tokens de solapamiento (60), medidos con el TOKENIZER DEL
   MODELO DE EMBEDDINGS (no con palabras ni caracteres).
3. El corte se corre hacia atrás hasta el último fin de párrafo, de ítem o de
   oración dentro de la ventana, para no partir una ficha a la mitad.
4. El texto de cada fragmento se toma del ORIGINAL con los offsets del tokenizer
   (no se decodifican tokens): no se pierden tildes ni mayúsculas.

`Tokenizador` es un protocolo: en modo real es el tokenizer de HuggingFace del
modelo de embeddings (rag/embeddings.py); en mock y tests, `TokenizadorSimple`
(palabras y signos), que no descarga nada.
"""

import re
from typing import Protocol

from rag.documentos import DocumentoGuia, Fragmento

FIN_NATURAL: re.Pattern[str] = re.compile(r"\n\s*\n|\n(?=- )|(?<=[.!?:])\s")


class Tokenizador(Protocol):
    def offsets(self, texto: str) -> list[tuple[int, int]]:
        """Span de caracteres (inicio, fin) de cada token del texto, sin tokens especiales."""
        ...


class TokenizadorSimple:
    """Palabras y signos sueltos (modo mock y tests). Cuenta parecido a un tokenizer subword real."""

    def offsets(self, texto: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\w+|[^\w\s]", texto)]


def secciones(markdown: str) -> list[tuple[str, str]]:
    """[(ruta de sección, texto)] — 'Comprender', 'Hacer > Senderismo'... El texto previo al primer título es 'Introducción'."""
    pila: list[tuple[int, str]] = []
    salida: list[tuple[str, list[str]]] = [("Introducción", [])]
    for linea in markdown.split("\n"):
        if m := re.match(r"^(#{1,6}) (.+)$", linea):
            nivel = len(m.group(1))
            if nivel == 1:  # el título del documento
                continue
            pila = [(n, t) for n, t in pila if n < nivel] + [(nivel, m.group(2).strip())]
            salida.append((" > ".join(t for _, t in pila), []))
        else:
            salida[-1][1].append(linea)
    return [(ruta, "\n".join(lineas).strip()) for ruta, lineas in salida if "\n".join(lineas).strip()]


def ventanas(texto: str, tok: Tokenizador, max_tokens: int, overlap: int) -> list[tuple[str, int]]:
    """Corta `texto` en ventanas de ≤ max_tokens con `overlap` tokens de solapamiento. Devuelve (texto, tokens)."""
    spans = tok.offsets(texto)
    if not spans:
        return []
    if len(spans) <= max_tokens:
        return [(texto.strip(), len(spans))]
    salida: list[tuple[str, int]] = []
    inicio = 0
    while inicio < len(spans):
        fin = min(inicio + max_tokens, len(spans))
        if fin < len(spans):
            # Correr el corte al último fin natural (párrafo, ítem u oración) en la segunda mitad de la ventana.
            desde_char, hasta_char = spans[inicio + max_tokens // 2][0], spans[fin - 1][1]
            cortes = [m.end() for m in FIN_NATURAL.finditer(texto, desde_char, hasta_char)]
            if cortes:
                limite = cortes[-1]
                fin = max(i for i in range(inicio + 1, fin + 1) if spans[i - 1][1] <= limite)
        trozo = texto[spans[inicio][0]:spans[fin - 1][1]].strip()
        if trozo:
            salida.append((trozo, fin - inicio))
        if fin >= len(spans):
            break
        inicio = max(fin - overlap, inicio + 1)
    return salida


def fragmentar(doc: DocumentoGuia, tok: Tokenizador, max_tokens: int, overlap: int, huella: str) -> list[Fragmento]:
    """Todos los fragmentos de una guía. `huella` identifica el modelo y los parámetros (va en doc_hash)."""
    fragmentos: list[Fragmento] = []
    doc_hash = f"{doc.hash}-{huella}"
    for ruta, texto in secciones(doc.texto):
        for trozo, n_tokens in ventanas(texto, tok, max_tokens, overlap):
            if n_tokens < 5:  # restos sin contenido (un título suelto, un "Más lejos:")
                continue
            fragmentos.append(Fragmento(
                id=f"guia:{doc.slug}#{len(fragmentos):03d}", slug=doc.slug, destino=doc.destino, seccion=ruta,
                texto=trozo, url=doc.url, titulo=doc.titulo, licencia=doc.licencia, orden=len(fragmentos),
                tokens=n_tokens, doc_hash=doc_hash,
            ))
    return fragmentos
