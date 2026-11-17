"""DermAssist: web front end for the ISIC 2018 skin lesion classifier.

Run from the repository root:
    uvicorn app.server:app --port 8000

Environment variables (optional):
    DERMASSIST_CKPT     checkpoint      (default checkpoints/convnext_balanced.pth)
    DERMASSIST_RESULTS  results folder  (default results/)
"""
import base64
import datetime as dt
import io
import json
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import store
from .content import HOSPITAL, LESIONS, REFER_CLASSES
from .image_checks import lesion_descriptors, mask_outline, quality_checks

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
GALLERY = Path(__file__).resolve().parent / "gallery"
CKPT = Path(os.environ.get("DERMASSIST_CKPT", ROOT / "checkpoints" / "convnext_balanced.pth"))
RESULTS_DIR = Path(os.environ.get("DERMASSIST_RESULTS", ROOT / "results"))
MAIN_RUN = "convnext_balanced"
MAX_UPLOAD = 15 * 1024 * 1024

app = FastAPI(title="DermAssist")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR, check_dir=False), name="results")
app.mount("/gallery", StaticFiles(directory=GALLERY, check_dir=False), name="gallery")

# ---------------------------------------------------------------- model + data
classifier, load_error = None, None
try:
    from .model import Classifier
    classifier = Classifier(CKPT)
except Exception as exc:                                    # noqa: BLE001
    load_error = f"{type(exc).__name__}: {exc}"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


MAIN_RESULT = _read_json(RESULTS_DIR / f"{MAIN_RUN}.json") or {}

gallery = None
if (GALLERY / "index.npz").exists():
    g = np.load(GALLERY / "index.npz", allow_pickle=False)
    gallery = {"emb": g["embeddings"].astype(np.float32), "labels": g["labels"], "images": g["images"]}


def _data_url(rgb_uint8, size=None, quality=88):
    img = Image.fromarray(rgb_uint8)
    if size:
        img = img.resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _lesion(code):
    return {"code": code, **LESIONS[code]}


# ---------------------------------------------------------------- pages
@app.get("/")
def page_research():
    return FileResponse(STATIC / "index.html")


@app.get("/analyse")
def page_analyse():
    return FileResponse(STATIC / "analyse.html")


@app.get("/hospital")
def page_hospital():
    return FileResponse(STATIC / "hospital.html")


@app.get("/appointments")
def page_appointments():
    return FileResponse(STATIC / "appointments.html")


# ---------------------------------------------------------------- model info
@app.get("/api/model")
def model_info():
    info = {"loaded": classifier is not None, "error": load_error, "gallery": gallery is not None,
            "gallery_size": int(len(gallery["labels"])) if gallery else 0}
    if classifier is not None:
        info.update(arch=classifier.arch, img_size=classifier.img_size)
    return info


@app.get("/api/results")
def results():
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name.endswith("_train.json"):
            continue
        r = _read_json(path)
        train = _read_json(path.with_name(path.stem + "_train.json")) or {}
        if not r:
            continue
        runs.append({k: r.get(k) for k in ("name", "protocol", "n_test", "config", "test", "test_ci95",
                                           "per_class", "confusion_matrix", "melanoma_operating_point",
                                           "malignant_vs_benign", "mean_average_precision")}
                    | {"n_train": train.get("n_train"), "n_val": train.get("n_val"),
                       "history": train.get("history"), "train_minutes": train.get("train_minutes")})
    return {"runs": runs, "main": MAIN_RUN}


# ---------------------------------------------------------------- analysis
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(503, f"The model is not loaded. Put the checkpoint at {CKPT} and restart. "
                                 f"({load_error})")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "The image is larger than 15 MB.")
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "This file could not be read as an image.") from None

    rgb = np.asarray(pil)
    ok, warnings = quality_checks(rgb)
    if not ok:
        raise HTTPException(400, " ".join(warnings))

    classes = classifier.classes
    probs, emb = classifier.predict(pil)
    order = [int(i) for i in np.argsort(probs)[::-1]]
    top = classes[order[0]]
    ranked = [{"code": classes[i], "name": LESIONS[classes[i]]["name"],
               "category": LESIONS[classes[i]]["category"], "prob": round(float(probs[i]), 4)} for i in order]

    op = MAIN_RESULT.get("melanoma_operating_point") or {}
    threshold = op.get("threshold")
    mel_prob = float(probs[classes.index("MEL")])
    mel_flag = threshold is not None and mel_prob >= threshold
    reasons = []
    if top in REFER_CLASSES:
        reasons.append(f"the most likely class is {LESIONS[top]['name'].lower()}, which is "
                       f"{LESIONS[top]['category']}")
    if mel_flag:
        reasons.append("the melanoma probability is above the screening threshold")
    triage = {"level": "refer" if reasons else "routine",
              "reason": "; ".join(reasons) if reasons else
              "no malignant class is most likely and the melanoma probability is below the screening threshold"}

    heat = classifier.heatmaps(pil, order[:3])
    size = classifier.img_size
    original = np.asarray(pil.resize((size, size)))

    similar = None
    if gallery is not None:
        sims = gallery["emb"] @ emb
        best = np.argsort(sims)[::-1][:6]
        similar = [{"image": str(gallery["images"][i]), "code": classes[int(gallery["labels"][i])],
                    "name": LESIONS[classes[int(gallery["labels"][i])]]["name"],
                    "similarity": round(float(sims[i]), 3),
                    "url": f"/gallery/thumbs/{gallery['images'][i]}.jpg"} for i in best]

    case = store.add("cases", {
        "top": top, "name": LESIONS[top]["name"], "category": LESIONS[top]["category"],
        "prob": round(float(probs[order[0]]), 4), "mel_prob": round(mel_prob, 4),
        "triage": triage["level"], "status": "new", "thumb": _data_url(original, 96, 80)}, "CASE")

    return {
        "case_id": case["id"],
        "prediction": ranked[0],
        "probabilities": ranked,
        "melanoma": {"prob": round(mel_prob, 4), "threshold": threshold, "flagged": bool(mel_flag),
                     "test_sensitivity": (op.get("test") or {}).get("sensitivity"),
                     "test_specificity": (op.get("test") or {}).get("specificity")},
        "triage": triage,
        "warnings": warnings,
        "descriptors": lesion_descriptors(rgb),
        "images": {"original": _data_url(original), "outline": _data_url(mask_outline(rgb)),
                   "heatmaps": {classes[i]: _data_url(h) for i, h in heat.items()}},
        "similar": similar,
        "info": _lesion(top),
    }


@app.get("/api/lesions")
def lesions():
    return {code: _lesion(code) for code in LESIONS}


# ---------------------------------------------------------------- cases
class StatusUpdate(BaseModel):
    status: Literal["new", "queued", "reviewed"]


@app.get("/api/cases")
def cases():
    items = store.load("cases")
    items.sort(key=lambda c: (c["status"] == "reviewed", c["triage"] != "refer", -c["mel_prob"]))
    return items


@app.get("/api/cases/{case_id}")
def case(case_id: str):
    c = store.get("cases", case_id)
    if c is None:
        raise HTTPException(404, "Case not found.")
    return c


@app.post("/api/cases/{case_id}/status")
def case_status(case_id: str, body: StatusUpdate):
    c = store.update("cases", case_id, status=body.status)
    if c is None:
        raise HTTPException(404, "Case not found.")
    return c


# ---------------------------------------------------------------- hospital + appointments
def _open_dates(weeks=4):
    today = dt.date.today()
    return [(today + dt.timedelta(days=d)).isoformat() for d in range(1, weeks * 7 + 1)
            if (today + dt.timedelta(days=d)).weekday() in HOSPITAL["appointment_days"]]


@app.get("/api/hospital")
def hospital():
    booked = {}
    for a in store.load("appointments"):
        booked.setdefault(a["date"], []).append(a["slot"])
    return HOSPITAL | {"dates": _open_dates(), "booked": booked}


class Appointment(BaseModel):
    patient_name: str = Field(min_length=2, max_length=80)
    age: int = Field(ge=0, le=120)
    sex: Literal["female", "male", "other"]
    phone: Optional[str] = Field(default=None, max_length=30)
    date: str
    slot: str
    case_id: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=500)


@app.get("/api/appointments")
def appointments():
    items = store.load("appointments")
    items.sort(key=lambda a: (a["date"], a["slot"]))
    return items


@app.post("/api/appointments")
def book(body: Appointment):
    if body.date not in _open_dates():
        raise HTTPException(400, "Choose one of the visiting dermatologist's dates.")
    if body.slot not in HOSPITAL["slots"]:
        raise HTTPException(400, "Choose one of the listed time slots.")
    if any(a["date"] == body.date and a["slot"] == body.slot for a in store.load("appointments")):
        raise HTTPException(409, "That time slot has just been booked. Choose another one.")
    case_summary = None
    if body.case_id:
        c = store.get("cases", body.case_id)
        if c is None:
            raise HTTPException(400, "The linked case was not found.")
        case_summary = {k: c[k] for k in ("id", "name", "category", "prob", "mel_prob", "triage", "thumb")}
        store.update("cases", body.case_id, status="queued")
    return store.add("appointments", body.model_dump() | {"case": case_summary}, "APT")


@app.delete("/api/appointments/{appointment_id}")
def cancel(appointment_id: str):
    if not store.remove("appointments", appointment_id):
        raise HTTPException(404, "Appointment not found.")
    return {"cancelled": appointment_id}
