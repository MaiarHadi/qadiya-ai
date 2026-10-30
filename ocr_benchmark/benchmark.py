"""
مقارنة EasyOCR مع PaddleOCR على نفس الصور (O.1).

التشغيل (من مجلد qadiya-ai):
    python ocr_benchmark/benchmark.py
    python ocr_benchmark/benchmark.py --engines easyocr          # محرك واحد بس
    python ocr_benchmark/benchmark.py --preprocess none            # بدون Enhance

كل تشغيلة بتدمج نتائجها مع اللي قبل (محرك جديد ما بيمسح نتائج محرك تاني).

النتائج بتنكتب بـ ocr_benchmark/results/:
    summary.md   ← الجداول للتوثيق
    results.csv  ← كل صورة × محرك × تجهيز
    outputs.json ← النص اللي طلّعه كل محرك (للمراجعة)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # عشان نقدر نستورد app.services.enhance
from app.services.enhance import enhance_image  # noqa: E402

TESTSET = HERE / "testset"
RESULTS = HERE / "results"


# ─────────────────────────────────────────────
# 1) المقاييس
# ─────────────────────────────────────────────
ARABIC_MARKS = re.compile(r"[ً-ْٰـ]")  # تشكيل + تطويل


def normalize(text: str) -> str:
    """بنقارن المعنى مش الشكل: حروف صغيرة، مسافات موحدة، بدون تشكيل."""
    text = unicodedata.normalize("NFKC", text)
    text = ARABIC_MARKS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return re.sub(r"\s+", " ", text).strip().lower()


def levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(truth: str, guess: str) -> float:
    """Character Error Rate: نسبة الأحرف الغلط (0 = مثالي)."""
    t, g = normalize(truth), normalize(guess)
    return levenshtein(t, g) / max(len(t), 1)


def wer(truth: str, guess: str) -> float:
    """Word Error Rate: نسبة الكلمات الغلط."""
    t, g = normalize(truth).split(), normalize(guess).split()
    return levenshtein(t, g) / max(len(t), 1)


def keys_found(keys: list[str], guess: str) -> int:
    """المعلومات المهمة للعبة (رقم حساب، ساعة، نمرة...) انقرت صح؟"""
    g = normalize(guess).replace(" ", "")
    return sum(1 for k in keys if normalize(k).replace(" ", "") in g)


# ─────────────────────────────────────────────
# 2) ترتيب الكلمات لسطور (المحركات بترجع مربعات مش نص مرتب)
# ─────────────────────────────────────────────
def boxes_to_text(items: list[tuple[np.ndarray, str]], rtl: bool) -> str:
    if not items:
        return ""
    rows = []
    for box, text in items:
        pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        rows.append((pts[:, 1].mean(), pts[:, 0].mean(), pts[:, 1].max() - pts[:, 1].min(), text))
    rows.sort(key=lambda r: r[0])
    tol = max(statistics.median(r[2] for r in rows) * 0.5, 4)

    lines, current = [], [rows[0]]
    for r in rows[1:]:
        if abs(r[0] - current[-1][0]) <= tol:
            current.append(r)
        else:
            lines.append(current)
            current = [r]
    lines.append(current)
    return "\n".join(
        " ".join(r[3] for r in sorted(line, key=lambda r: r[1], reverse=rtl)) for line in lines
    )


# ─────────────────────────────────────────────
# 3) المحركات (كل واحد إله نفس الشكل: lang → دالة بتاخد صورة BGR وبترجع نص)
# ─────────────────────────────────────────────
class EasyOCREngine:
    name = "easyocr"

    def __init__(self):
        import easyocr

        self.version = easyocr.__version__
        self._easyocr = easyocr
        self.readers = {}

    def load(self, lang: str):
        if lang not in self.readers:
            langs = ["ar", "en"] if lang == "ar" else ["en"]
            self.readers[lang] = self._easyocr.Reader(langs, gpu=False, verbose=False)
        return self.readers[lang]

    def read(self, img_bgr: np.ndarray, lang: str) -> str:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = self.load(lang).readtext(rgb, detail=1, paragraph=False)
        return boxes_to_text([(box, text) for box, text, _conf in out], rtl=lang == "ar")


class PaddleOCREngine:
    name = "paddleocr"
    extra: dict = {}  # بيتعبى من --paddle-no-mkldnn

    def __init__(self):
        import paddleocr

        self.version = getattr(paddleocr, "__version__", "?")
        self.major = int(str(self.version).split(".")[0]) if self.version[0].isdigit() else 3
        self._paddleocr = paddleocr
        self.models = {}

    def load(self, lang: str):
        if lang in self.models:
            return self.models[lang]
        P = self._paddleocr.PaddleOCR
        if self.major >= 3:
            last_error = None
            for code in (["ar", "arabic"] if lang == "ar" else ["en"]):
                try:
                    self.models[lang] = P(
                        lang=code,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=True,
                        **self.extra,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    last_error = e
            else:
                raise last_error
        else:
            self.models[lang] = P(use_angle_cls=True, lang="arabic" if lang == "ar" else "en", show_log=False)
        return self.models[lang]

    def read(self, img_bgr: np.ndarray, lang: str) -> str:
        model = self.load(lang)
        items = []
        if self.major >= 3:
            for res in model.predict(img_bgr):
                data = res.json.get("res", res) if hasattr(res, "json") else res
                polys = data.get("rec_polys") if data.get("rec_polys") is not None else data.get("dt_polys", [])
                items += list(zip(polys, data.get("rec_texts", [])))
        else:
            for page in model.ocr(img_bgr, cls=True) or []:
                for box, (text, _score) in page or []:
                    items.append((box, text))
        return boxes_to_text(items, rtl=lang == "ar")


ENGINES = {"easyocr": EasyOCREngine, "paddleocr": PaddleOCREngine}


# ─────────────────────────────────────────────
# 4) التجهيز قبل الـOCR (نفس خدمة الـEnhance تبعتنا)
# ─────────────────────────────────────────────
def preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img
    h, w = img.shape[:2]
    factor = 3 if min(h, w) < 120 else 2 if min(h, w) < 400 else 1
    return enhance_image(img, denoise_strength=7, use_clahe=True, sharpen_amount=0.6, upscale_factor=factor)


# ─────────────────────────────────────────────
# 5) التشغيل
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--preprocess", nargs="+", default=["none", "enhance"], choices=["none", "enhance"])
    ap.add_argument("--paddle-no-mkldnn", action="store_true",
                    help="إذا Paddle طلّع NotImplementedError / ConvertPirAttribute على Windows")
    args = ap.parse_args()
    if args.paddle_no_mkldnn:
        PaddleOCREngine.extra = {"enable_mkldnn": False}

    cases = json.loads((TESTSET / "ground_truth.json").read_text(encoding="utf-8"))
    RESULTS.mkdir(exist_ok=True)
    rows, outputs, env = [], {}, {"os": platform.platform(), "cpu": platform.processor()}

    for engine_name in args.engines:
        try:
            engine = ENGINES[engine_name]()
        except ImportError as e:
            print(f"⚠️  {engine_name} مش منزل ({e}). بنكمل بدونه.")
            continue
        env[engine_name] = engine.version
        env[f"{engine_name}_python"] = platform.python_version()

        # وقت تحميل النماذج (أول مرة بتنزل من النت، فبنقيسه لحاله)
        load_times = {}
        for lang in sorted({c["lang"] for c in cases}):
            t0 = time.perf_counter()
            try:
                engine.load(lang)
                load_times[lang] = round(time.perf_counter() - t0, 2)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  {engine_name} ما قدر يحمّل لغة {lang}: {e}")
        env[f"{engine_name}_load_seconds"] = load_times

        # تسخين: أول قراءة دايماً أبطأ، ما بنحسبها
        warm = cv2.imread(str(TESTSET / cases[0]["file"]))
        engine.read(warm, "en")

        for mode in args.preprocess:
            for c in cases:
                if c["lang"] not in load_times:
                    continue
                img = cv2.imread(str(TESTSET / c["file"]))
                prepared = preprocess(img, mode)
                t0 = time.perf_counter()
                try:
                    text = engine.read(prepared, c["lang"])
                except Exception as e:  # noqa: BLE001
                    text = ""
                    print(f"⚠️  {engine_name}/{mode}/{c['id']}: {e}")
                seconds = time.perf_counter() - t0

                row = {
                    "engine": engine_name,
                    "preprocess": mode,
                    "case": c["id"],
                    "lang": c["lang"],
                    "category": c["category"],
                    "accuracy": round(max(0.0, 1 - cer(c["text"], text)) * 100, 1),
                    "cer": round(cer(c["text"], text), 3),
                    "wer": round(wer(c["text"], text), 3),
                    "keys_found": keys_found(c["keys"], text),
                    "keys_total": len(c["keys"]),
                    "seconds": round(seconds, 3),
                }
                rows.append(row)
                outputs[f"{engine_name}|{mode}|{c['id']}"] = {"truth": c["text"], "ocr": text}
                print(f"{engine_name:9} {mode:7} {c['id']:24} acc {row['accuracy']:5.1f}%  "
                      f"keys {row['keys_found']}/{row['keys_total']}  {seconds:.2f}s")

    if not rows:
        print("ما في نتائج. نزّلي محرك واحد على الأقل (شوفي requirements-ocr.txt).")
        return

    rows, outputs, env = merge_previous(rows, outputs, env)

    with open(RESULTS / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "outputs.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "summary.md").write_text(build_summary(rows, cases, env), encoding="utf-8")
    print(f"\n✅ النتائج بـ {RESULTS}")


NUMERIC = {"accuracy", "cer", "wer", "seconds"}


def merge_previous(rows, outputs, env):
    """نتائج المحركات اللي ما انشغلت هالمرة بتضل زي ما هي."""
    ran = {r["engine"] for r in rows}
    old_rows, old_out, old_env = [], {}, {}
    if (RESULTS / "results.csv").exists():
        with open(RESULTS / "results.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["engine"] in ran:
                    continue
                for k in r:
                    if k in NUMERIC:
                        r[k] = float(r[k])
                    elif k in ("keys_found", "keys_total"):
                        r[k] = int(r[k])
                old_rows.append(r)
    if (RESULTS / "outputs.json").exists():
        old_out = {k: v for k, v in json.loads((RESULTS / "outputs.json").read_text(encoding="utf-8")).items()
                   if k.split("|")[0] not in ran}
    if (RESULTS / "env.json").exists():
        old_env = {k: v for k, v in json.loads((RESULTS / "env.json").read_text(encoding="utf-8")).items()
                   if k.split("_")[0] not in ran}
    return old_rows + rows, {**old_out, **outputs}, {**old_env, **env}


def build_summary(rows, cases, env) -> str:
    combos = sorted({(r["engine"], r["preprocess"]) for r in rows})
    L = ["# OCR comparison: EasyOCR vs PaddleOCR", ""]
    L += [f"- Test images: {len(cases)} ({', '.join(sorted({c['category'] for c in cases}))})",
          "- Accuracy = 100% − character error rate. Key facts = numbers/names the game depends on, read exactly.",
          "- Time = average per image on CPU, after a warm-up run (model loading not included).", ""]

    L += ["## Overall", "", "| Engine | Preprocess | Accuracy | Key facts | Avg time / image |", "|---|---|---|---|---|"]
    for e, p in combos:
        rs = [r for r in rows if r["engine"] == e and r["preprocess"] == p]
        acc = statistics.mean(r["accuracy"] for r in rs)
        keys = sum(r["keys_found"] for r in rs) / max(sum(r["keys_total"] for r in rs), 1) * 100
        t = statistics.mean(r["seconds"] for r in rs)
        L.append(f"| {e} | {p} | {acc:.1f}% | {keys:.0f}% | {t:.2f}s |")

    L += ["", "## By category (accuracy)", ""]
    cats = sorted({r["category"] for r in rows})
    L += ["| Engine | Preprocess | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
    for e, p in combos:
        cells = []
        for cat in cats:
            rs = [r["accuracy"] for r in rows if r["engine"] == e and r["preprocess"] == p and r["category"] == cat]
            cells.append(f"{statistics.mean(rs):.0f}%" if rs else "—")
        L.append(f"| {e} | {p} | " + " | ".join(cells) + " |")

    L += ["", "## Per image (accuracy · key facts)", ""]
    L += ["| Image | " + " | ".join(f"{e} / {p}" for e, p in combos) + " |", "|---|" + "---|" * len(combos)]
    for c in cases:
        cells = []
        for e, p in combos:
            r = next((r for r in rows if r["engine"] == e and r["preprocess"] == p and r["case"] == c["id"]), None)
            cells.append(f"{r['accuracy']:.0f}% · {r['keys_found']}/{r['keys_total']}" if r else "—")
        L.append(f"| {c['id']} — {c['note']} | " + " | ".join(cells) + " |")

    L += ["", "## Environment", ""] + [f"- {k}: {v}" for k, v in env.items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
