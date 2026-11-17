"""Tiny JSON-file storage for analysed cases and demo appointments.

Files live in app/data/ (git-ignored). Image data is not stored, only the
model's summary of each case.
"""
import json
import threading
import time
import uuid
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
_lock = threading.Lock()


def _path(name):
    DATA.mkdir(exist_ok=True)
    return DATA / f"{name}.json"


def load(name):
    p = _path(name)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save(name, items):
    _path(name).write_text(json.dumps(items, indent=2), encoding="utf-8")


def add(name, item, prefix):
    with _lock:
        items = load(name)
        item = {"id": f"{prefix}-{uuid.uuid4().hex[:6].upper()}", "created": time.time(), **item}
        items.append(item)
        save(name, items)
        return item


def update(name, item_id, **fields):
    with _lock:
        items = load(name)
        for it in items:
            if it["id"] == item_id:
                it.update(fields)
                save(name, items)
                return it
    return None


def remove(name, item_id):
    with _lock:
        items = load(name)
        kept = [it for it in items if it["id"] != item_id]
        save(name, kept)
        return len(kept) != len(items)


def get(name, item_id):
    return next((it for it in load(name) if it["id"] == item_id), None)
