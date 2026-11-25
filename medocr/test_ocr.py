from paddleocr import PaddleOCR

# Create the OCR engine once (first run downloads models, takes a few minutes)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,  # skip page-rotation check
    use_doc_unwarping=False,             # skip curved-page fixing
    use_textline_orientation=False,      # skip upside-down text check
)

# Run OCR on the image
result = ocr.predict("sample.jpeg")

for res in result:
    texts = res["rec_texts"]    # each line of text found
    scores = res["rec_scores"]  # confidence for each line (0 to 1)

    print("\n===== Extracted text =====")
    for text, score in zip(texts, scores):
        print(f"{score:.2f}  |  {text}")

    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nLines found: {len(texts)}   Avg confidence: {avg:.2%}")

    # Saves a copy of the image with boxes drawn around detected text
    res.save_to_img("output")