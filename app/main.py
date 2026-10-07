"""
app/main.py — the FastAPI app: the ONE required endpoint (POST /classify)
plus GET /health. Wires classifier.predict() and storage.save_prediction()
together; contains no inference or DB logic of its own.

- lifespan (the current, non-deprecated startup mechanism) loads the model
  ONCE before the server accepts any request — requests then take
  milliseconds, not seconds.
- POST /classify: image bytes -> predict -> save -> return prediction + id.
  Unreadable input -> HTTP 400 (explicit decision, plan §4.1).
"""

import hashlib
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.classifier import get_classifier
from app.storage import count_predictions, init_db, save_prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup (cleanup at shutdown): init DB, load model."""
    init_db()
    get_classifier()  # one-time model load BEFORE the first request
    print("startup complete: model loaded, DB ready")
    yield
    # nothing to clean up for a thin slice


app = FastAPI(
    title="Satellite Tile Intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Cheap liveness probe — the seed of the Part 3 Q2 monitoring answer."""
    model_ok, db_ok = True, True
    try:
        get_classifier()
    except Exception:
        model_ok = False
    try:
        count_predictions()
    except Exception:
        db_ok = False
    return {"model_loaded": model_ok, "db_ok": db_ok}


@app.post("/classify")
def classify(file: UploadFile = File(...)) -> dict:
    """THE required endpoint: tile in -> prediction stored -> result out."""
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400,
                            detail="Empty file — no image bytes received.")

    clf = get_classifier()
    tile_hash = hashlib.sha256(raw).hexdigest()
    filename = file.filename or "unknown"

    try:
        result = clf.predict(raw)
    except ValueError as exc:      # unreadable/unsupported input — caller's fault
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:       # model-side failure — our fault
        raise HTTPException(status_code=500,
                            detail=f"Classification failed: {exc}") from exc

    prediction_id = save_prediction(filename, tile_hash, result, clf.model_version)

    return {
        "prediction_id": prediction_id,   # traceable to the exact DB row
        "filename": filename,
        "predicted_label": result["predicted_label"],
        "confidence": round(result["confidence"], 4),
        "is_uncertain": result["is_uncertain"],
        "model_version": clf.model_version,
    }