"""
app/classifier.py — loads models/classifier_head.pkl ONCE and exposes a single
predict() used by BOTH the live server (app/main.py) and the evaluation script
(scripts/evaluate.py). No HTTP logic lives here, and there is no second copy
of preprocessing anywhere — per plan §6.

predict() contract (plan §9.4), returned as a dict with the four planned
fields: predicted_label, confidence, probabilities, is_uncertain.
Raises ValueError for bytes that aren't a decodable image — the caller
(main.py) converts that to HTTP 400.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from app.preprocessing import (
    build_feature_extractor,
    extract_features,
    load_image,
    preprocess,
)

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = MODELS_DIR / "classifier_head.pkl"


class TileClassifier:
    """Holds the full artifact (backbone + head + classes + threshold) in
    memory. Build once at startup; every predict() after that is read-only,
    which makes concurrent requests from FastAPI's threadpool safe."""

    def __init__(self, artifact_path: Path = ARTIFACT_PATH) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"{artifact_path} not found — run `python scripts/train_head.py` first."
            )
        with open(artifact_path, "rb") as f:
            art = pickle.load(f)

        self.backbone_name: str = art["backbone"]
        self.pipeline = art["pipeline"]          # scaler + head, self-contained
        self.classes: list[str] = art["classes"]
        self.threshold: float = art["threshold"]
        self.model_version: str = art["model_version"]
        self.feature_dim: int = art["feature_dim"]
        self.trained_at: str = art["trained_at"]

        self._extractor = build_feature_extractor(self.backbone_name)

    def predict(self, image_bytes: bytes) -> dict:
        """Raw image bytes -> prediction dict (the single shared inference path)."""
        img = load_image(image_bytes)            # raises ValueError on bad bytes
        batch = preprocess(img)                  # (1, 3, 224, 224)
        feats = extract_features(self._extractor, batch).numpy()  # (1, D)
        probs = self.pipeline.predict_proba(feats)[0]             # (n_classes,)

        top = int(np.argmax(probs))
        confidence = float(probs[top])
        return {
            "predicted_label": self.classes[top],        # raw label, ALWAYS stored
            "confidence": confidence,
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "is_uncertain": confidence < self.threshold, # policy flag, separate
        }


# Lazy singleton. main.py calls get_classifier() inside its lifespan startup
# hook — initializing it there (single-threaded, before serving begins) avoids
# any init race, and every later call just reuses the loaded instance.
_classifier: TileClassifier | None = None


def get_classifier() -> TileClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TileClassifier()
    return _classifier


def predict(image_bytes: bytes) -> dict:
    """Module-level convenience matching the plan's one-function contract."""
    return get_classifier().predict(image_bytes)