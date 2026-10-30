
from __future__ import annotations

import math
import os
import statistics
import threading
import re

from concurrent.futures import ThreadPoolExecutor


import numpy as np

from app.services.image_processing import ImageProcessingError

SUPPORTED_LANGS = ("en", "ar")
LOW_CONFIDENCE = 0.80 
ARABIC_RATIO = 0.30

ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_models: dict[str, object] = {}
_langs_locks = {lang: threading.Lock() for lang in SUPPORTED_LANGS}
_load_lock = threading.Lock()

_auto_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr-auto")


def _create_model(lang: str):
    from paddleocr import PaddleOCR  

    options = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,  
    )
    if os.getenv("OCR_DISABLE_MKLDNN") == "1":  
        options["enable_mkldnn"] = False

    codes = ["ar", "arabic"] if lang == "ar" else ["en"]
    last_error = None
    for code in codes:
        try:
            return PaddleOCR(lang=code, **options)
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise last_error


def get_model(lang: str):
    if lang not in SUPPORTED_LANGS:
        raise ImageProcessingError(f"Language not supported:{lang}")
    if lang not in _models:
        with _load_lock:
            if lang not in _models:  
                _models[lang] = _create_model(lang)
    return _models[lang]


def start_preload(langs: str | None = None) -> None:
    
    wanted = [l.strip() for l in (langs or os.getenv("OCR_PRELOAD", "en,ar")).split(",") if l.strip()]

    def run():
        for lang in wanted:
            try:
                get_model(lang)
                print(f"OCR model ready: {lang}")
            except Exception as e:  
                print(f"OCR preload failed for {lang}: {e}")

    threading.Thread(target=run, name="ocr-preload", daemon=True).start()


def crop_normalized(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
   
    values = (x, y, w, h)
    if not all(math.isfinite(v) for v in values):
        raise ImageProcessingError(" The coordinates are invalid.")
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1 + 1e-6 and y + h <= 1 + 1e-6):
        raise ImageProcessingError("The area outside the image boundaries")
    H, W = img.shape[:2]
    left, top = int(round(x * W)), int(round(y * H))
    right, bottom = min(W, int(round((x + w) * W))), min(H, int(round((y + h) * H)))
    if right - left < 1 or bottom - top < 1:
        raise ImageProcessingError("The area is very small.")
    return img[top:bottom, left:right].copy()


def _raw_items(model, img: np.ndarray):
  
    items = []
    if hasattr(model, "predict"):
        for res in model.predict(img):
            data = res.json.get("res", res) if hasattr(res, "json") else res
            polys = data.get("rec_polys")
            if polys is None:
                polys = data.get("dt_polys", [])
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [1.0] * len(texts))
            items += list(zip(polys, texts, scores))
    else: 
        for page in model.ocr(img, cls=True) or []:
            for poly, (text, score) in page or []:
                items.append((poly, text, score))
    return [(p, t, float(s)) for p, t, s in items if str(t).strip()]


def _group_lines(items, width: int, height: int, rtl: bool):
 
    boxes = []
    for poly, text, score in items:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "cy": (y0 + y1) / 2, "text": str(text), "score": score})
    if not boxes:
        return []

    boxes.sort(key=lambda b: b["cy"])
    tol = max(statistics.median(b["y1"] - b["y0"] for b in boxes) * 0.5, 4)
    groups, current = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b["cy"] - current[-1]["cy"]) <= tol:
            current.append(b)
        else:
            groups.append(current)
            current = [b]
    groups.append(current)

    lines = []
    for g in groups:
        g.sort(key=lambda b: b["x0"], reverse=rtl)
        x0, y0 = min(b["x0"] for b in g), min(b["y0"] for b in g)
        x1, y1 = max(b["x1"] for b in g), max(b["y1"] for b in g)
        lines.append({
            "text": " ".join(b["text"] for b in g),
            "confidence": round(statistics.mean(b["score"] for b in g), 3),


            
            "box": [round(float(v), 4) for v in (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)],
        })
    return lines


##هون عشان يفحص هل هي لغه عربي ولا انجليزي


def arabic_ratio(text: str) -> float:

  
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ARABIC_CHAR.match(c)) / len(letters)


def _read_with(img: np.ndarray, lang: str):
    model = get_model(lang)
    with _langs_locks[lang]:
        return _raw_items(model, img)


def read_text(img: np.ndarray, lang: str = "auto") -> dict:



    if lang == "auto":
       
        ar_job = _auto_pool.submit(_read_with, img, "ar")
        en_job = _auto_pool.submit(_read_with, img, "en")
        ar_items, en_items = ar_job.result(), en_job.result()
        if arabic_ratio(" ".join(t for _, t, _ in ar_items)) >= ARABIC_RATIO:
            detected, items = "ar", ar_items
        else:
            detected, items = "en", en_items
    else:
        detected = lang
        items = _read_with(img, lang)

    h, w = img.shape[:2]
    lines = _group_lines(items, w, h, rtl=detected == "ar")
    confidence = round(statistics.mean(l["confidence"] for l in lines), 3) if lines else 0.0
    return {
        "lang": detected,
        "auto_detected": lang == "auto",
        "text": "\n".join(l["text"] for l in lines),
        "lines": lines,
        "confidence": confidence,
        "low_confidence": bool(lines) and confidence < LOW_CONFIDENCE,
        "found_text": bool(lines),
    }