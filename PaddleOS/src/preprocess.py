import cv2

MAX_SIDE = 1600   # enough for printed reports; bigger = slower, not more accurate


def prepare(img):
    """Resize + grayscale + contrast boost on an image array (BGR)."""
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), [w, h], [img.shape[1], img.shape[0]]


def preprocess(path):
    """Read an image file from disk and prepare it."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    out, original, processed = prepare(img)
    return out, {"original_size": original, "processed_size": processed}