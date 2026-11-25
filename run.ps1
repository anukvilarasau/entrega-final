# Alternativa sin Docker para Windows (PowerShell). Un solo comando:
#   .\run.ps1           API real (usa el .env: claves, REDIS_URL, LangSmith)
#   .\run.ps1 -Mock     API sin claves ni descargas (LLM y Tavily simulados, fakeredis)
#   .\run.ps1 -Tests    corre los tests (sin claves)
# Crea el entorno virtual con Python 3.12 si no existe, instala las dependencias fijadas,
# corre la ingesta del RAG (idempotente) y levanta uvicorn en http://localhost:8000/docs
param([switch]$Mock, [switch]$Tests)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    try { & py -3.12 --version | Out-Null } catch { throw "Hace falta Python 3.12 (https://www.python.org/downloads/)." }
    Write-Host "==> Creando el entorno virtual (.venv) con Python 3.12"
    & py -3.12 -m venv .venv
}
$python = ".venv\Scripts\python.exe"
Write-Host "==> Instalando dependencias (requirements.txt)"
& $python -m pip install -q --upgrade pip
& $python -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Falló la instalación de dependencias." }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> Creé .env a partir de .env.example: completá ANTHROPIC_API_KEY y TAVILY_API_KEY (o usá -Mock)."
}

if ($Tests) { & $python -m pytest -q; exit $LASTEXITCODE }
if ($Mock) {
    $env:MODO_MOCK = "1"
    Write-Host "==> Modo mock: sin claves, sin descargas, estado en memoria (fakeredis)"
} else {
    Write-Host "==> Ingesta del RAG (solo indexa lo nuevo o lo que cambió; la primera vez descarga los modelos ONNX)"
    & $python -m rag.ingesta
    if ($LASTEXITCODE -ne 0) { throw "Falló la ingesta del RAG." }
}
Write-Host "==> API en http://localhost:8000/docs  (Ctrl+C para cortar)"
& $python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
