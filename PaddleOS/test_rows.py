from src.ocr import run_ocr

pages = run_ocr("samples/test.jpeg")

for page in pages:
    print(f"--- Page {page['page']} | avg confidence {page['avg_confidence']} ---")
    for row in page["rows"]:
        print(row)