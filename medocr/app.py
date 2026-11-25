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

from extractor import analyze, get_ocr

ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
MAX_MB = 10
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading OCR models (once)...")
    get_ocr()                       # warm up, so the first upload isn't slow
    print("Ready ✅  Open http://localhost:8000")
    yield

app = FastAPI(title="MedOCR", lifespan=lifespan)


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/analyze")
def analyze_report(file: UploadFile = File(...)):
    # 1) Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {ext}. Use: {', '.join(ALLOWED)}")

    # 2) Validate size
    content = file.file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_MB} MB)")

    # 3) Save to a temp file with a random name, analyze, then always delete it
    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    temp_path.write_bytes(content)
    start = time.time()
    try:
        output = analyze(str(temp_path))
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        os.remove(temp_path)       # never keep patient files on disk

    output["filename"] = file.filename
    output["processing_seconds"] = round(time.time() - start, 1)
    return output