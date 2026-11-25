import os
import ollama
from src.schema import LabReport

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """You convert OCR text of a lab report into JSON.
Each line is one table row. Cells are separated by ' | '.
The row pattern is: TEST NAME | VALUE | UNIT | REFERENCE RANGE

EXAMPLE
Input rows:
HAEMOGLOBIN | 11.8 | gms% | 82.60 % | (M) 13.5 - 18.0 gms%
(F) 11.5- 16.0 gms%
NEUTROPHILS | 81 | % | 40-70%
LYMPOCYTES | 12 | /% | 20-40%

Output tests:
[
 {"test_name": "HAEMOGLOBIN", "value": "11.8", "unit": "gms%", "reference_range": "(M) 13.5-18.0 gms%; (F) 11.5-16.0 gms%", "flag": null},
 {"test_name": "NEUTROPHILS", "value": "81", "unit": "%", "reference_range": "40-70%", "flag": "high"},
 {"test_name": "LYMPOCYTES", "value": "12", "unit": "%", "reference_range": "20-40%", "flag": "low"}
]

RULES
1. Make one test for EVERY row that has a test name followed by a number. Do not skip any row.
2. value = the first number right after the test name. Copy it exactly.
3. A line starting with (F) belongs to the test row just above it.
4. Rows with text results (RBCs, WBCs, PLATELETS under peripheral smear) go in other_findings, one item each.
5. Never invent text that is not in the input. Unknown fields = null.
6. flag: compare value with range. If range has (M) and (F) and sex is unknown, flag = null."""


def extract(pages):
    ocr_text = "\n\n".join(
        f"--- Page {p['page']} ---\n" + "\n".join(p["rows"]) for p in pages
    )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Input rows:\n" + ocr_text},
        ],
        format=LabReport.model_json_schema(),
        options={"temperature": 0, "num_ctx": 4096},   # 8192 -> 4096: less RAM, faster
        keep_alive="30m",                               # keep model loaded between uploads
    )
    return LabReport.model_validate_json(response.message.content)