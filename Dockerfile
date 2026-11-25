# Imagen de la API (entrega final). La usa docker-compose.yml junto con Redis 8.
#   docker compose up --build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELOS_CACHE_DIR=/modelos \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Modelos ONNX dentro de la imagen (embeddings del RAG, re-ranker y embedding de la memoria): la API arranca
# sin descargar nada. Se eligen con las mismas variables del .env (docker-compose.yml las pasa como build args).
# Esta capa va ANTES de copiar el código: cambiar el código no vuelve a descargar ~1,6 GB de modelos.
ARG EMBEDDINGS_MODELO
ARG RERANKER_MODELO
ARG RERANKER
COPY scripts/descargar_modelos.py scripts/descargar_modelos.py
RUN python scripts/descargar_modelos.py

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
# La ingesta del RAG corre sola al arrancar (INGESTA_AL_ARRANCAR=true): indexa si el índice está vacío o
# si cambió el corpus; si no, no toca nada.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
