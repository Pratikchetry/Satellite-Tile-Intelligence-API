"""
tests/test_classify.py — fast sanity checks over the core layers.

"""

import hashlib
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import storage
from app.classifier import get_classifier
from app.preprocessing import (
    FEATURE_DIMS,
    build_feature_extractor,
    extract_features,
    image_to_tensor,
    load_image,
)

EVAL_DIR = PROJECT_ROOT / "data" / "eval_set"


def _sample_bytes() -> bytes:
    return sorted(EVAL_DIR.glob("*.png"))[0].read_bytes()


def test_classifier_loads_and_predict_contract():
    clf = get_classifier()
    r = clf.predict(_sample_bytes())
    assert set(r) == {"predicted_label", "confidence",
                      "probabilities", "is_uncertain"}
    assert r["predicted_label"] in clf.classes
    assert 0.0 <= r["confidence"] <= 1.0
    assert set(r["probabilities"]) == set(clf.classes)
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-6
    assert isinstance(r["is_uncertain"], bool)
    # policy check: flag == (confidence below threshold)
    assert r["is_uncertain"] == (r["confidence"] < clf.threshold)


def test_garbage_bytes_raise_valueerror():
    clf = get_classifier()
    try:
        clf.predict(b"this is definitely not an image")
    except ValueError:
        return  # expected
    raise AssertionError("expected ValueError for non-image bytes")


def test_preprocessing_shapes():
    img = load_image(_sample_bytes())
    t = image_to_tensor(img)
    assert tuple(t.shape) == (3, 224, 224)
    ext = build_feature_extractor("squeezenet1_1")
    feats = extract_features(ext, t.unsqueeze(0))
    assert tuple(feats.shape) == (1, FEATURE_DIMS["squeezenet1_1"])


def test_predict_is_deterministic():
    # Same bytes -> same result. Verifies eval() mode is active (e.g.
    # SqueezeNet's internal dropout must be OFF during inference).
    clf = get_classifier()
    raw = _sample_bytes()
    assert clf.predict(raw) == clf.predict(raw)


def test_storage_roundtrip_isolated_db():
    original = storage.DB_PATH
    try:
        with tempfile.TemporaryDirectory() as tmp:
            storage.DB_PATH = Path(tmp) / "test.sqlite3"  # isolate the test
            storage.init_db()
            clf = get_classifier()
            raw = _sample_bytes()
            result = clf.predict(raw)
            pid = storage.save_prediction(
                "test.png", hashlib.sha256(raw).hexdigest(),
                result, clf.model_version,
            )
            row = storage.get_prediction(pid)
            assert row is not None
            assert row["predicted_label"] == result["predicted_label"]
            assert abs(row["confidence"] - result["confidence"]) < 1e-9
            assert row["probabilities"] == result["probabilities"]
            assert bool(row["is_uncertain"]) == result["is_uncertain"]
            assert row["model_version"] == clf.model_version
            assert storage.get_prediction(999999) is None
    finally:
        storage.DB_PATH = original  # restore, whatever happened


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    if failures == 0:
        print("\nAll tests passed.")
    else:
        print(f"\n{failures} test(s) FAILED")
    sys.exit(1 if failures else 0)