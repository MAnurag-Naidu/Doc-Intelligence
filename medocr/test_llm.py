import os
import json
import statistics
from dotenv import load_dotenv
from google import genai
from google.genai import types
from paddleocr import PaddleOCR

load_dotenv()  # reads GEMINI_API_KEY and GEMINI_MODEL from .env

IMAGE = "sample.jpeg"   # use a dummy / blurred report — this text is sent to Google


# ---------- 1) OCR with PaddleOCR (runs locally) ----------
def ocr_rows(image_path):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    res = ocr.predict(image_path)[0]

    items = []
    for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        items.append({"text": text, "score": float(score), "x": x1,
                      "yc": (y1 + y2) / 2, "h": y2 - y1})

    # group pieces into rows by vertical position (same as test_rows.py)
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
    return "\n".join(lines), avg_conf


# ---------- 2) LLM turns messy rows into structured JSON ----------
PROMPT = """You are extracting data from an OCR'd medical lab report.
The text below was rebuilt row-by-row from word positions, so:
- a test's NAME may appear one row above or below its VALUE,
- reference ranges may span two lines, e.g. (M) for male and (F) for female,
- there may be small OCR spelling mistakes (e.g. "RFF DOCTOR" = "REF DOCTOR").

Rules:
- Use ONLY values present in the text. Never guess or invent.
- If something is missing or unclear, use null.
- For each test, pick the reference range that matches the patient's sex if known,
  and set "flag" by comparing the value with that range ("high", "low", "normal"), else null.

Return JSON in exactly this shape:
{
  "patient": {"name": str|null, "age": str|null, "sex": str|null, "patient_id": str|null},
  "report":  {"lab_name": str|null, "report_date": str|null, "ref_doctor": str|null, "report_type": str|null},
  "tests": [{"name": str, "value": str, "unit": str|null, "reference_range": str|null,
             "flag": "high"|"low"|"normal"|null}],
  "notes": str|null
}

OCR TEXT:
"""


def extract_with_llm(ocr_text):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=os.environ["GEMINI_MODEL"],
        contents=PROMPT + ocr_text,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",  # forces JSON output
            temperature=0,                          # no creativity — be consistent
        ),
    )
    return json.loads(response.text)


# ---------- 3) Run ----------
if __name__ == "__main__":
    print("Running OCR...")
    text, conf = ocr_rows(IMAGE)
    print(f"OCR done. Avg confidence: {conf:.2%}")

    print("Sending to Gemini...")
    data = extract_with_llm(text)

    print("\n===== Structured JSON =====")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    print("\n===== Tests table =====")
    print(f"{'TEST':<25}{'VALUE':<12}{'UNIT':<15}{'RANGE':<30}FLAG")
    for t in data.get("tests", []):
        print(f"{t['name']:<25}{str(t['value']):<12}{str(t['unit']):<15}"
              f"{str(t['reference_range']):<30}{t['flag']}")