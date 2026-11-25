import json
import sys
import time
from pathlib import Path

from src.ocr import run_ocr
from src.llm import extract
from src.postprocess import postprocess


def process(path):
    print(f"\n📄 {path}")
    t0 = time.time()
    pages = run_ocr(str(path))          # Paddle needs a plain string path
    if not any(p["rows"] for p in pages):
        print("❌ No text found – check the file path/format or image quality.")
        return
    t1 = time.time()
    report = postprocess(extract(pages))
    t2 = time.time()

    out = Path("outputs") / f"{Path(path).stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "source": str(path),
        "extracted": report.model_dump(),
        "ocr_pages": pages,
        "timing_sec": {"ocr": round(t1 - t0, 1), "llm": round(t2 - t1, 1)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ {len(report.tests)} tests, {len(report.other_findings)} findings → {out}")
    print(f"⏱  OCR {t1 - t0:.1f}s | LLM {t2 - t1:.1f}s")
    for t in report.tests:
        flag = f"  ⚠ {t.flag.upper()}" if t.flag in ("high", "low") else ""
    print(f"   {t.test_name}: {t.value} {t.unit or ''}{flag}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <file or folder>")

    target = Path(sys.argv[1])
    if target.is_dir():
        files = [f for f in target.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf")]
        for f in files:
            process(f)
    else:
        process(target)