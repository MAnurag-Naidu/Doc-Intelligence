import json
import time

from src.ocr import run_ocr
from src.llm import extract
from src.postprocess import postprocess

t0 = time.time()
pages = run_ocr("samples/test.jpeg")
t1 = time.time()
report = postprocess(extract(pages))
t2 = time.time()

print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
print(f"\nOCR: {t1 - t0:.1f}s | LLM: {t2 - t1:.1f}s | Tests: {len(report.tests)} | Findings: {len(report.other_findings)}")