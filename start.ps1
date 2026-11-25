# start.ps1 — launch all three servers for the merged OCR app.
#
#   medocr   (images/screenshots -> PaddleOCR + Gemini) on port 8001
#   PaddleOS (PDFs -> PaddleOCR + Ollama)               on port 8002
#   gateway  (routes uploads, serves the upload page)   on port 8000
#
# Each backend runs in its OWN virtualenv from its OWN working directory, so
# their conflicting dependencies never meet. The gateway only forwards bytes.
#
# Prereqs:
#   - medocr\.env has GEMINI_API_KEY (and optional GEMINI_MODEL).
#   - Ollama is running with the model in PaddleOS (default qwen2.5:3b).
#   - gateway venv exists (see "Gateway venv" below); if missing this script
#     creates it and installs gateway\requirements.txt.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\start.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$medocrPy   = Join-Path $root "medocr\venv\Scripts\python.exe"
$paddlePy   = Join-Path $root "PaddleOS\.venv\Scripts\python.exe"
$gatewayDir = Join-Path $root "gateway"
$gatewayPy  = Join-Path $gatewayDir ".venv\Scripts\python.exe"

# --- Gateway venv (create once) ---------------------------------------------
if (-not (Test-Path $gatewayPy)) {
    Write-Host "Creating gateway virtualenv..." -ForegroundColor Cyan
    python -m venv (Join-Path $gatewayDir ".venv")
    & $gatewayPy -m pip install --upgrade pip
    & $gatewayPy -m pip install -r (Join-Path $gatewayDir "requirements.txt")
}

# --- sanity checks ----------------------------------------------------------
foreach ($p in @($medocrPy, $paddlePy, $gatewayPy)) {
    if (-not (Test-Path $p)) { throw "Python not found: $p" }
}

function Start-Server($title, $py, $workdir, $port) {
    Write-Host "Starting $title on port $port ..." -ForegroundColor Green
    $cmd = "Set-Location '$workdir'; & '$py' -m uvicorn app:app --host 127.0.0.1 --port $port"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd | Out-Null
}

Start-Server "medocr"   $medocrPy (Join-Path $root "medocr")   8001
Start-Server "PaddleOS" $paddlePy (Join-Path $root "PaddleOS") 8002
Start-Server "gateway"  $gatewayPy $gatewayDir                 8000

# --- wait for both backends to answer (models load at startup) --------------
function Wait-Up($name, $url, $timeoutSec = 120) {
    Write-Host "Waiting for $name ($url) ..." -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 | Out-Null
            Write-Host "  $name is up." -ForegroundColor Green
            return $true
        } catch { Start-Sleep -Seconds 2 }
    }
    Write-Host "  $name did not respond within $timeoutSec s (check its window)." -ForegroundColor Yellow
    return $false
}

Wait-Up "medocr"   "http://127.0.0.1:8001/" | Out-Null
Wait-Up "PaddleOS" "http://127.0.0.1:8002/" | Out-Null
Wait-Up "gateway"  "http://127.0.0.1:8000/" | Out-Null

Write-Host ""
Write-Host "All set. Open the merged app:  http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  (medocr stays on :8001, PaddleOS on :8002 if you want them directly.)"
Start-Process "http://127.0.0.1:8000"
