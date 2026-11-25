import sys
import time
from paddleocr import PaddleOCR

mkldnn = sys.argv[1] == "on"     # run with: python test_speed.py on   (or off)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    enable_mkldnn=mkldnn,
)

for run in (1, 2):
    t = time.time()
    res = ocr.predict("sample.jpeg")[0]
    scores = res["rec_scores"]
    print(f"oneDNN {'ON ' if mkldnn else 'OFF'} | run {run}: "
          f"{time.time() - t:.1f}s, lines {len(scores)}, "
          f"confidence {sum(scores) / len(scores):.1%}")