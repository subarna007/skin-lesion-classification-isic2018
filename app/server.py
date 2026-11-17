"""DermAssist: web front end for the ISIC 2018 classifier (research prototype).

Run from the repository root:
    uvicorn app.server:app --port 8000
Environment variables:
    DERMASSIST_CKPT     checkpoint (default checkpoints/convnext_balanced.pth)
    DERMASSIST_RESULTS  evaluation JSON (default results/convnext_balanced.json)
"""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .image_checks import lesion_descriptors, mask_outline, quality_checks

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
CKPT = Path(os.environ.get("DERMASSIST_CKPT", ROOT / "checkpoints" / "convnext_balanced.pth"))
RESULTS = Path(os.environ.get("DERMASSIST_RESULTS", ROOT / "results" / "convnext_balanced.json"))
MAX_UPLOAD = 15 * 1024 * 1024

LESIONS = {
    "MEL": ("Melanoma", "malignant", "A malignant tumour of pigment cells; the most dangerous common skin cancer."),
    "NV": ("Melanocytic nevus", "benign", "A common mole made of pigment cells."),
    "BCC": ("Basal cell carcinoma", "malignant", "The most common skin cancer; grows slowly and rarely spreads."),
    "AKIEC": ("Actinic keratosis / intraepithelial carcinoma", "pre-malignant",
              "Sun-damage lesions that can progress to squamous cell carcinoma."),
    "BKL": ("Benign keratosis", "benign", "Seborrhoeic keratosis, solar lentigo or lichen-planus-like keratosis."),
    "DF": ("Dermatofibroma", "benign", "A small, firm, harmless skin nodule."),
    "VASC": ("Vascular lesion", "benign", "Angiomas and other blood-vessel lesions."),
}
REFER_CLASSES = {"MEL", "BCC", "AKIEC"}

app = FastAPI(title="DermAssist")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

classifier, load_error, results = None, None, {}
try:
    from .model import Classifier
    classifier = Classifier(CKPT)
except Exception as exc:                                    # noqa: BLE001
    load_error = f"{type(exc).__name__}: {exc}"
if RESULTS.exists():
    results = json.loads(RESULTS.read_text())


def _data_url(rgb_uint8):
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/model")
def model_info():
    info = {"loaded": classifier is not None, "error": load_error, "lesions": LESIONS}
    if classifier is not None:
        info.update(arch=classifier.arch, img_size=classifier.img_size, classes=classifier.classes)
    if results:
        info["evaluation"] = {k: results.get(k) for k in
                              ("protocol", "n_test", "test", "test_ci95", "melanoma_operating_point",
                               "malignant_vs_benign", "per_class")}
    return info


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(503, f"Model not loaded ({load_error}). Put the checkpoint at {CKPT}.")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 15 MB.")
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "This file could not be read as an image.") from None

    rgb = np.asarray(pil)
    ok, warnings = quality_checks(rgb)
    if not ok:
        raise HTTPException(400, " ".join(warnings))

    probs = classifier.predict(pil)
    order = np.argsort(probs)[::-1]
    classes = classifier.classes
    top = int(order[0])
    ranked = [{"code": classes[i], "name": LESIONS[classes[i]][0], "category": LESIONS[classes[i]][1],
               "prob": round(float(probs[i]), 4)} for i in order]

    mel_prob = float(probs[classes.index("MEL")])
    op = results.get("melanoma_operating_point") or {}
    threshold = op.get("threshold")
    mel_flag = threshold is not None and mel_prob >= threshold
    if classes[top] in REFER_CLASSES or mel_flag:
        reasons = []
        if classes[top] in REFER_CLASSES:
            reasons.append(f"most likely class is {LESIONS[classes[top]][0].lower()} ({LESIONS[classes[top]][1]})")
        if mel_flag:
            reasons.append("melanoma probability is above the screening threshold")
        triage = {"level": "refer", "reason": "; ".join(reasons)}
    else:
        triage = {"level": "routine", "reason": "no malignant class is most likely and the melanoma "
                                                "probability is below the screening threshold"}

    return {
        "prediction": ranked[0] | {"description": LESIONS[classes[top]][2]},
        "probabilities": ranked,
        "melanoma": {"prob": round(mel_prob, 4), "threshold": threshold, "flagged": bool(mel_flag),
                     "test_sensitivity": (op.get("test") or {}).get("sensitivity"),
                     "test_specificity": (op.get("test") or {}).get("specificity")},
        "triage": triage,
        "warnings": warnings,
        "descriptors": lesion_descriptors(rgb),
        "images": {"original": _data_url(np.asarray(pil.resize((classifier.img_size,) * 2))),
                   "gradcam": _data_url(classifier.gradcam(pil, top)),
                   "outline": _data_url(mask_outline(rgb))},
    }
