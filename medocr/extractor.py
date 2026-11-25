"""
extractor.py — the core logic, separate from the web server.
  1. PaddleOCR reads the image (locally)
  2. Rows are rebuilt from word positions
  3. Gemini turns rows into structured JSON
  4. Python calculates high/low/normal flags
"""
import os
import time
from google.genai import errors
import json
import re
import statistics
from dotenv import load_dotenv
from google import genai
from google.genai import types
from paddleocr import PaddleOCR

load_dotenv()

# ---------- OCR engine: created once, reused for every upload ----------
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # Smaller, faster models — made for CPUs / laptops
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            # Turn off oneDNN — it's causing the slowdown on this machine
            enable_mkldnn=True,
        )
    return _ocr


# ---------- Step 1 + 2: OCR and rebuild rows ----------
def ocr_rows(image_path):
    res = get_ocr().predict(image_path)[0]

    items = []
    for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        items.append({"text": text, "score": float(score), "x": x1,
                      "yc": (y1 + y2) / 2, "h": y2 - y1})

    if not items:
        return "", 0.0, []

    tolerance = statistics.median(i["h"] for i in items) * 0.6
    items.sort(key=lambda i: i["yc"])
    rows = []
    for it in items:
        if rows and abs(it["yc"] - rows[-1]["yc"]) <= tolerance:
            rows[-1]["items"].append(it)
            ys = [i["yc"] for i in rows[-1]["items"]]
            rows[-1]["yc"] = sum(ys) / len(ys)
        else:
            rows.append({"yc": it["yc"], "items": [it]})

    lines = []
    for r in rows:
        cells = sorted(r["items"], key=lambda i: i["x"])
        lines.append("  |  ".join(c["text"] for c in cells))

    avg_conf = sum(i["score"] for i in items) / len(items)
    low_conf = [i["text"] for i in items if i["score"] < 0.85]
    return "\n".join(lines), avg_conf, low_conf


# ---------- Step 3: LLM extraction ----------
PROMPT = """You are extracting data from an OCR'd medical lab report.
The text below was rebuilt row-by-row from word positions, so:
- a test's NAME may appear one row above or below its VALUE,
- reference ranges may span two lines, e.g. (M) for male and (F) for female,
- there may be small OCR spelling mistakes (e.g. "RFF DOCTOR" = "REF DOCTOR").

Rules:
- Use ONLY values present in the text. Never guess or invent.
- If something is missing or unclear, use null.
- For each test, choose the reference range matching the patient's sex if known.
- "ref_low" and "ref_high" are the numeric bounds of that chosen range (numbers only, no units).
- Clean obvious OCR noise in units (e.g. "Mill/Cumm.." -> "Mill/Cumm").
- Do NOT judge whether values are high or low.

Return JSON in exactly this shape:
{
  "patient": {"name": str|null, "age": str|null, "sex": str|null, "patient_id": str|null},
  "report":  {"lab_name": str|null, "report_date": str|null, "ref_doctor": str|null, "report_type": str|null},
  "tests": [{"name": str, "value": str, "unit": str|null, "reference_range": str|null,
             "ref_low": number|null, "ref_high": number|null}],
  "notes": str|null
}

OCR TEXT:
"""

def thinking_for(model):
    """Gemini 2.5 and Gemini 3+ use different 'thinking' settings."""
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0)      # 2.5: turn thinking off
    return types.ThinkingConfig(thinking_level="low")       # 3.x: minimal thinking


def extract_with_llm(ocr_text):
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=60_000,                                    # give up after 60 s (ms)
            retry_options=types.HttpRetryOptions(attempts=2),  # retry once, then move on
        ),
    )
    models = [os.environ["GEMINI_MODEL"], os.environ.get("GEMINI_FALLBACK_MODEL")]

    last_error = None
    for model in [m for m in models if m]:
        try:
            response = client.models.generate_content(
                model=model,
                contents=PROMPT + ocr_text,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                    thinking_config=thinking_for(model),
                ),
            )
            print(f"LLM answered using: {model}")
            return json.loads(response.text)
        except (errors.ServerError, errors.ClientError) as e:   # busy (5xx) or not allowed (4xx)
            print(f"{model} failed ({e.code}), trying next model...")
            last_error = e

    raise last_error   # every model failed
# ---------- Step 4: flags calculated in Python ----------
def to_number(value):
    """'9,000' -> 9000.0 ; '2.80' -> 2.8 ; 'ADEQUATE' -> None"""
    if value is None:
        return None
    match = re.search(r"-?\d+(\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None

def add_flags(data):
    for t in data.get("tests", []):
        v, lo, hi = to_number(t.get("value")), t.get("ref_low"), t.get("ref_high")
        if v is None or lo is None or hi is None:
            t["flag"] = None
        elif v < lo:
            t["flag"] = "low"
        elif v > hi:
            t["flag"] = "high"
        else:
            t["flag"] = "normal"
    return data


# ---------- Full pipeline ----------
def analyze(image_path):
    t0 = time.time()
    text, avg_conf, low_conf = ocr_rows(image_path)
    t1 = time.time()
    if not text:
        return {"error": "No text found in the image."}

    data = add_flags(extract_with_llm(text))
    t2 = time.time()

    return {
        "ocr_confidence": round(avg_conf * 100, 1),
        "low_confidence_words": low_conf,
        "ocr_text": text,
        "result": data,
        "timing": {
            "ocr_seconds": round(t1 - t0, 1),
            "llm_seconds": round(t2 - t1, 1),
        },
    }