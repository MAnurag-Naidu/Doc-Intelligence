"""
app.py — Routing gateway.

  GET  /         -> one upload page for images, screenshots, and PDFs
  POST /analyze  -> forward the file to the right OCR engine and
                    return one normalized shape:
                      images / screenshots -> medocr   (PaddleOCR + Gemini)
                      PDFs                  -> PaddleOS (PaddleOCR + Ollama)

This process never imports Paddle or any LLM. It only forwards bytes over
HTTP to the two existing servers, so their separate virtualenvs stay intact.
"""
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

# --- routing config ---------------------------------------------------------
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
PDF_EXT = {".pdf"}
ALLOWED = IMAGE_EXT | PDF_EXT
MAX_MB = 10

MEDOCR_URL = "http://127.0.0.1:8001/analyze"
PADDLEOS_URL = "http://127.0.0.1:8002/analyze"

# Gemini and (CPU) Ollama can be slow — a multi-page PDF through a local LLM can
# run for several minutes. Use a short connect timeout but a long read timeout so
# big reports aren't cut off. Override the read timeout (seconds) with
# GATEWAY_UPSTREAM_TIMEOUT; set it to 0 to wait indefinitely.
_read_timeout = float(os.getenv("GATEWAY_UPSTREAM_TIMEOUT", "600"))
UPSTREAM_TIMEOUT = httpx.Timeout(_read_timeout or None, connect=10.0)

app = FastAPI(title="OCR Gateway — image→medocr, PDF→PaddleOS")


@app.get("/")
def home():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


def _engine_for(ext: str) -> str:
    if ext in IMAGE_EXT:
        return "medocr"
    if ext in PDF_EXT:
        return "paddleos"
    raise HTTPException(
        400, f"Unsupported file type {ext}. Use: {', '.join(sorted(ALLOWED))}"
    )


# --- response normalization -------------------------------------------------
def _normalize_medocr(data: dict) -> dict:
    """medocr returns {ocr_confidence, ocr_text, result:{patient,report,tests,notes}, timing}."""
    result = data.get("result") or {}
    return {
        "engine": "medocr",
        "filename": data.get("filename"),
        "processing_seconds": data.get("processing_seconds"),
        "timing": data.get("timing") or {},
        "patient": result.get("patient") or {},
        "report": result.get("report") or {},
        "tests": [
            {
                "name": t.get("name"),
                "value": t.get("value"),
                "unit": t.get("unit"),
                "reference_range": t.get("reference_range"),
                "flag": t.get("flag"),
            }
            for t in (result.get("tests") or [])
        ],
        "notes": result.get("notes"),
        "other_findings": [],
        "raw_text": data.get("ocr_text") or "",
        "raw": data,
    }


def _normalize_paddleos(data: dict) -> dict:
    """PaddleOS returns {extracted:{patient_name,...,tests,other_findings}, ocr_pages, timing}."""
    extracted = data.get("extracted") or {}
    pages = data.get("ocr_pages") or []
    raw_text = "\n\n".join(
        "\n".join(p.get("rows") or []) for p in pages
    )
    return {
        "engine": "paddleos",
        "filename": data.get("filename"),
        "processing_seconds": data.get("processing_seconds"),
        "timing": data.get("timing") or {},
        "patient": {
            "name": extracted.get("patient_name"),
            "age": extracted.get("age"),
            "sex": extracted.get("sex"),
        },
        "report": {
            "report_type": extracted.get("report_title"),
            "report_date": extracted.get("report_date"),
            "sample_type": extracted.get("sample_type"),
            "method": extracted.get("method"),
            "model": data.get("model"),
        },
        "tests": [
            {
                "name": t.get("test_name"),
                "value": t.get("value"),
                "unit": t.get("unit"),
                "reference_range": t.get("reference_range"),
                "flag": t.get("flag"),
            }
            for t in (extracted.get("tests") or [])
        ],
        "notes": None,
        "other_findings": [
            {"name": f.get("name"), "result": f.get("result")}
            for f in (extracted.get("other_findings") or [])
        ],
        "raw_text": raw_text,
        "raw": data,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1) Validate type -> decide engine
    ext = Path(file.filename or "").suffix.lower()
    engine = _engine_for(ext)

    # 2) Validate size
    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_MB} MB)")

    target = MEDOCR_URL if engine == "medocr" else PADDLEOS_URL

    # 3) Forward the original bytes to the chosen backend
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(
                target,
                files={
                    "file": (
                        file.filename,
                        content,
                        file.content_type or "application/octet-stream",
                    )
                },
            )
    except httpx.ConnectError:
        raise HTTPException(
            502,
            f"The {engine} server is not reachable at {target}. "
            f"Start it first (see start.ps1).",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            504,
            f"The {engine} server took longer than {int(_read_timeout)}s. "
            f"Large PDFs on a CPU-only LLM can be slow — raise GATEWAY_UPSTREAM_TIMEOUT.",
        )

    # 4) Bubble up backend errors with their message
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = None
        raise HTTPException(resp.status_code, detail or f"{engine} error: {resp.text}")

    data = resp.json()
    # medocr can return 200 with an {"error": ...} body when OCR finds nothing
    if isinstance(data, dict) and data.get("error"):
        raise HTTPException(422, data["error"])

    normalized = (
        _normalize_medocr(data) if engine == "medocr" else _normalize_paddleos(data)
    )
    normalized["gateway_seconds"] = round(time.time() - start, 1)
    return normalized
