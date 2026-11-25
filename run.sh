#!/usr/bin/env bash
# Alternativa sin Docker (Linux / macOS / Git Bash). Un solo comando:
#   ./run.sh           API real (usa el .env: claves, REDIS_URL, LangSmith)
#   ./run.sh --mock    API sin claves ni descargas (LLM y Tavily simulados, fakeredis)
#   ./run.sh --tests   corre los tests (sin claves)
# Crea el entorno virtual con Python 3.12 si no existe, instala las dependencias fijadas,
# corre la ingesta del RAG (idempotente) y levanta uvicorn en http://localhost:8000/docs
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidato in python3.12 python3 python; do
  if command -v "$candidato" >/dev/null 2>&1 && "$candidato" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    PY="$candidato"; break
  fi
done
if [ -z "$PY" ] && command -v py >/dev/null 2>&1 && py -3.12 --version >/dev/null 2>&1; then PY="py -3.12"; fi
if [ -z "$PY" ]; then echo "Hace falta Python 3.12 (https://www.python.org/downloads/)." >&2; exit 1; fi

if [ -d .venv/Scripts ]; then BIN=.venv/Scripts; else BIN=.venv/bin; fi
if [ ! -x "$BIN/python" ] && [ ! -x "$BIN/python.exe" ]; then
  echo "==> Creando el entorno virtual (.venv) con $PY"
  $PY -m venv .venv
  if [ -d .venv/Scripts ]; then BIN=.venv/Scripts; else BIN=.venv/bin; fi
fi
echo "==> Instalando dependencias (requirements.txt)"
"$BIN/python" -m pip install -q --upgrade pip
"$BIN/python" -m pip install -q -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Creé .env a partir de .env.example: completá ANTHROPIC_API_KEY y TAVILY_API_KEY (o usá --mock)."
fi

case "${1:-}" in
  --tests)
    exec "$BIN/python" -m pytest -q ;;
  --mock)
    export MODO_MOCK=1
    echo "==> Modo mock: sin claves, sin descargas, estado en memoria (fakeredis)" ;;
  *)
    echo "==> Ingesta del RAG (solo indexa lo nuevo o lo que cambió; la primera vez descarga los modelos ONNX)"
    "$BIN/python" -m rag.ingesta ;;
esac

echo "==> API en http://localhost:8000/docs  (Ctrl+C para cortar)"
exec "$BIN/python" -m uvicorn api.main:app --host 0.0.0.0 --port 8000
