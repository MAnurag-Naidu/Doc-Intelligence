import statistics
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

res = ocr.predict("sample.jpeg")[0]

# 1) Collect each text piece with its position
items = []
for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
    x1, y1, x2, y2 = [int(v) for v in box]   # box = left, top, right, bottom
    items.append({
        "text": text,
        "score": float(score),
        "x": x1,                  # left edge → column order
        "yc": (y1 + y2) / 2,      # vertical centre → which row
        "h": y2 - y1,             # height of the text
    })

# 2) Two pieces are on the same row if their centres are close vertically
median_h = statistics.median(i["h"] for i in items)
tolerance = median_h * 0.6

items.sort(key=lambda i: i["yc"])  # top to bottom
rows = []
for it in items:
    if rows and abs(it["yc"] - rows[-1]["yc"]) <= tolerance:
        rows[-1]["items"].append(it)
        ys = [i["yc"] for i in rows[-1]["items"]]
        rows[-1]["yc"] = sum(ys) / len(ys)       # update row's average height
    else:
        rows.append({"yc": it["yc"], "items": [it]})

# 3) Print each row left → right, and flag low-confidence words
print("\n===== Rebuilt rows =====\n")
for r in rows:
    cells = sorted(r["items"], key=lambda i: i["x"])
    line = "  |  ".join(c["text"] for c in cells)
    low = [c["text"] for c in cells if c["score"] < 0.85]
    flag = f"     ⚠ low confidence: {low}" if low else ""
    print(line + flag)