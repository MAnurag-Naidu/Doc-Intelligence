import os
import ollama
from src.schema import LabReport

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """You extract data from OCR text of Indian hospital lab reports.

Input format:
- Each line is one row of the page; cells are separated by ' | '.
- A reference range may be split across lines, e.g. '(M) ...' on the test's row and '(F) ...' on the next line. Combine them into one reference_range like "(M) 13.5-18.0 gms%; (F) 11.5-16.0 gms%".

Rules:
- value = the FIRST number after the test name. Ignore any extra numbers on that row.
- Copy values exactly as written. Never guess, calculate or convert units.
- Fix obvious OCR slips in units only (e.g. '/%' -> '%').
- If a field is not in the text, use null. Do not invent patient details.
- One item in "tests" per numeric lab test row. Skip headers like INVESTIGATION/VALUE/UNITS.
- Text results (e.g. peripheral smear: RBCs, WBCs, PLATELETS) go in "other_findings", not "tests".
- flag: "high" or "low" if the value is outside the reference range, "normal" if inside.
  If the range differs by sex and sex is unknown, use null."""


def extract(pages):
    ocr_text = "\n\n".join(
        f"--- Page {p['page']} ---\n" + "\n".join(p["rows"]) for p in pages
    )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ocr_text},
        ],
        format=LabReport.model_json_schema(),
        options={"temperature": 0, "num_ctx": 8192},
    )
    return LabReport.model_validate_json(response.message.content)