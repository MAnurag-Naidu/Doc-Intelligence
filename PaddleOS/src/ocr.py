import os
from pathlib import Path
from statistics import median

import cv2
import fitz                      # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR

from src.preprocess import preprocess, prepare

MIN_SCORE = 0.5
UNWARP = os.getenv("OCR_UNWARP", "1") == "1"   # straightening for phone photos
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
MIN_WORDS_FOR_TEXT_PDF = 20                    # fewer words than this = scanned page
PDF_RENDER_DPI = 150                           # for scanned PDF pages

_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=UNWARP,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _ocr


# ---------- Shared: turn positioned text into table rows ----------
def group_into_rows(texts, scores, boxes):
    items = []
    for text, score, box in zip(texts, scores, boxes):
        if score < MIN_SCORE or not str(text).strip():
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        items.append({"text": text, "x": x1, "yc": (y1 + y2) / 2, "h": y2 - y1})
    if not items:
        return []

    tolerance = median(i["h"] for i in items) * 0.5
    items.sort(key=lambda i: i["yc"])
    rows, current = [], [items[0]]
    for item in items[1:]:
        row_y = sum(i["yc"] for i in current) / len(current)
        if abs(item["yc"] - row_y) <= tolerance:
            current.append(item)
        else:
            rows.append(current)
            current = [item]
    rows.append(current)
    return [" | ".join(i["text"] for i in sorted(r, key=lambda i: i["x"])) for r in rows]


# ---------- Path 1: digital PDF – read the text layer (no OCR) ----------
def pdf_page_cells(page):
    """Merge PDF words into cells: words on the same line with small gaps = one cell."""
    words = sorted(page.get_text("words"), key=lambda w: (w[5], w[6], w[0]))  # block, line, x
    cells = []
    for x0, y0, x1, y1, word, block, line, _ in words:
        if cells:
            c = cells[-1]
            same_line = (c["block"], c["line"]) == (block, line)
            small_gap = x0 - c["box"][2] < (y1 - y0) * 1.5
            if same_line and small_gap:
                c["text"] += " " + word
                c["box"] = [c["box"][0], min(c["box"][1], y0), x1, max(c["box"][3], y1)]
                continue
        cells.append({"text": word, "box": [x0, y0, x1, y1], "block": block, "line": line})
    return [c["text"] for c in cells], [c["box"] for c in cells]


# ---------- Path 2: images & scanned PDF pages – PaddleOCR ----------
def ocr_image(image, unwarp):
    res = list(get_ocr().predict(image, use_doc_unwarping=unwarp))[0]
    texts, scores, boxes = res["rec_texts"], res["rec_scores"], res["rec_boxes"]
    avg = float(sum(scores) / len(scores)) if len(scores) else 0.0
    return group_into_rows(texts, scores, boxes), round(avg, 3)


def render_pdf_page(page):
    pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
    return prepare(img)[0]


def run_ocr(path):
    path = str(path)
    ext = Path(path).suffix.lower()
    pages = []

    if ext in IMAGE_EXT:
        image, info = preprocess(path)
        print(f"   image: {info['original_size']} -> {info['processed_size']}")
        rows, conf = ocr_image(image, unwarp=UNWARP)       # photos may be tilted
        pages.append({"page": 1, "source": "ocr", "rows": rows, "avg_confidence": conf})
        return pages

    # PDF: decide page by page
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            texts, boxes = pdf_page_cells(page)
            if len(texts) >= MIN_WORDS_FOR_TEXT_PDF:
                rows = group_into_rows(texts, [1.0] * len(texts), boxes)
                pages.append({"page": i, "source": "text-layer", "rows": rows, "avg_confidence": 1.0})
                print(f"   page {i}: text layer ({len(texts)} cells) – OCR skipped")
            else:
                image = render_pdf_page(page)
                rows, conf = ocr_image(image, unwarp=False)    # PDF pages are flat
                pages.append({"page": i, "source": "ocr", "rows": rows, "avg_confidence": conf})
                print(f"   page {i}: scanned -> OCR")
    return pages