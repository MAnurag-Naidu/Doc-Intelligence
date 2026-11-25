"""
app.py — FastAPI web server.
  GET  /         -> the upload page
  POST /analyze  -> upload a report, get JSON back
"""
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from src.ocr import get_ocr, run_ocr
from src.llm import extract, MODEL
from src.postprocess import postprocess

ALLOWED = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_MB = 10
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading OCR models (once)...")
    get_ocr()                       # warm up, so the first upload isn't slow
    print("Ready ✅  Open http://localhost:8000")
    yield

app = FastAPI(title="PaddleOS – Lab Report Extractor", lifespan=lifespan)


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/analyze")
def analyze_report(file: UploadFile = File(...)):
    # 1) Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {ext}. Use: {', '.join(sorted(ALLOWED))}")

    # 2) Validate size
    content = file.file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_MB} MB)")

    # 3) Save with a random name, process, then always delete
    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    temp_path.write_bytes(content)
    t0 = time.time()
    try:
        pages = run_ocr(str(temp_path))
        if not any(p["rows"] for p in pages):
            raise HTTPException(422, "No text found – check image quality")
        t1 = time.time()
        report = postprocess(extract(pages))
        t2 = time.time()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        os.remove(temp_path)        # never keep patient files on disk

    return {
        "filename": file.filename,
        "model": MODEL,
        "extracted": report.model_dump(),
        "ocr_pages": pages,
        "timing": {"ocr_seconds": round(t1 - t0, 1), "llm_seconds": round(t2 - t1, 1)},
        "processing_seconds": round(t2 - t0, 1),
    }