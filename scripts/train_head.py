"""
scripts/train_head.py — ONE-TIME training step (plan §5). Not part of the
live offline service. Runs the backbone x head bake-off, picks the winner
(with the pre-agreed tie-break rule), derives the confidence threshold on a
held-out validation split, and saves everything the server needs into a
single artifact: models/classifier_head.pkl

Steps:
  A. Stratified train/val split of candidate_tiles/ (val never touches eval_set)
  B. 3 backbones x 3 heads = 9 combos, all scored on the validation split
  C. Threshold sweep for the winning combo (on the same validation split)
  D. Save one artifact: backbone + head + classes + threshold + version

Run from project root:  python scripts/train_head.py
"""

import pickle
import sys
import time
from pathlib import Path

# Allow `python scripts/train_head.py` from the project root: put the project
# root on sys.path so `app.` imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from app.preprocessing import (
    FEATURE_DIMS,
    build_feature_extractor,
    extract_features,
    image_to_tensor,
    load_image,
)

# --- Constants (fixed in advance, per plan §5) ------------------------------
SEED = 42
VAL_FRACTION = 0.15
BATCH_SIZE = 64
TIE_BREAK_MARGIN = 0.02      # "within ~2 points" counts as a tie (plan §5.B.4)
THRESHOLDS = np.arange(0.30, 0.96, 0.05)

DATA_DIR = PROJECT_ROOT / "data" / "candidate_tiles"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Smaller/simpler = preferred on a tie. Ranks chosen by parameter count /
# CPU cost; heads by conceptual simplicity (plan §5.B.4).
BACKBONE_SIMPLICITY = {"squeezenet1_1": 0, "mobilenet_v2": 1, "resnet18": 2}
HEAD_SIMPLICITY = {"logreg": 0, "svm": 1, "mlp": 2}


def make_head(name: str) -> Pipeline:
    """Unfitted sklearn pipeline. StandardScaler is part of EVERY head so the
    artifact contains one self-contained predict/predict_proba object."""
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif name == "svm":
        # Plain LinearSVC has NO predict_proba — our whole uncertainty system
        # depends on real probabilities, so it gets sigmoid calibration
        # (plan §5.B.2). Scoring (accuracy/macro-F1) is unaffected by this.
        clf = CalibratedClassifierCV(
            LinearSVC(C=1.0, dual="auto"), method="sigmoid", cv=5
        )
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                            random_state=SEED)
    else:
        raise ValueError(name)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def load_dataset() -> tuple[list[Path], list[str]]:
    """candidate_tiles/<class_name>/<image> -> (paths, labels), sorted."""
    paths, labels = [], []
    for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for img in sorted(class_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                paths.append(img)
                labels.append(class_dir.name)
    return paths, labels


def extract_for_split(extractor, paths: list[Path]) -> np.ndarray:
    """Batched, gradient-free feature extraction (plan §5.B.1)."""
    chunks = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = paths[i:i + BATCH_SIZE]
        tensors = [image_to_tensor(load_image(p.read_bytes())) for p in batch]
        chunks.append(extract_features(extractor, torch.stack(tensors)).numpy())
    return np.vstack(chunks)


def sweep_threshold(conf: np.ndarray, errors: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Pick the threshold with the largest gap between (fraction of real
    errors captured) and (fraction of all tiles flagged). Larger gap = the
    flag is doing real work, not just flagging everything."""
    n_errors = int(errors.sum())
    rows = []
    for t in THRESHOLDS:
        flagged = conf < t
        flag_rate = float(flagged.mean())
        capture = float((flagged & errors).sum() / n_errors) if n_errors else 0.0
        rows.append({"threshold": round(float(t), 2),
                     "flag_rate": round(flag_rate, 4),
                     "error_capture_rate": round(capture, 4),
                     "score": round(capture - flag_rate, 4)})
    table = pd.DataFrame(rows)
    best = table.sort_values(["score", "flag_rate"],
                             ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def main() -> None:
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    # --- Step A: stratified split ------------------------------------------
    paths, labels = load_dataset()
    train_paths, val_paths, y_train, y_val = train_test_split(
        paths, labels, test_size=VAL_FRACTION, stratify=labels,
        random_state=SEED,
    )
    classes = sorted(set(labels))
    print(f"Dataset: {len(paths)} tiles, {len(classes)} classes: {classes}")
    print(f"Split: {len(train_paths)} train / {len(val_paths)} validation "
          f"(stratified, seed={SEED})\n")

    head_builders = {"logreg": make_head, "svm": make_head, "mlp": make_head}
    results = []

    # --- Step B: bake off 9 combos -----------------------------------------
    for backbone in BACKBONE_SIMPLICITY:  # simplest backbone first
        ext = build_feature_extractor(backbone)
        print(f"[{backbone}] extracting features "
              f"(dim={FEATURE_DIMS[backbone]})...")
        X_train = extract_for_split(ext, train_paths)
        X_val = extract_for_split(ext, val_paths)

        for head_name, builder in head_builders.items():
            pipe = builder(head_name)
            t1 = time.time()
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            acc = accuracy_score(y_val, pred)
            mf1 = f1_score(y_val, pred, average="macro")
            results.append({"backbone": backbone, "head": head_name,
                            "val_accuracy": round(acc, 4),
                            "val_macro_f1": round(mf1, 4),
                            "fit_seconds": round(time.time() - t1, 1)})
            print(f"  {backbone:15s} + {head_name:6s} "
                  f"acc={acc:.3f}  macro_f1={mf1:.3f}")

    table = pd.DataFrame(results).sort_values(
        ["val_macro_f1", "val_accuracy"], ascending=False
    ).reset_index(drop=True)
    print("\n=== Bake-off results (validation split) ===")
    print(table.to_string(index=False))
    table.to_csv(MODELS_DIR / "bakeoff_results.csv", index=False)

    # --- Tie-break: prefer simple when scores are within noise -------------
    best_score = table["val_macro_f1"].max()
    contenders = table[table["val_macro_f1"] >= best_score - TIE_BREAK_MARGIN]
    tie_broken = len(contenders) > 1
    winner = min(
        contenders.itertuples(),
        key=lambda r: BACKBONE_SIMPLICITY[r.backbone] + HEAD_SIMPLICITY[r.head],
    )
    if tie_broken:
        print(f"\nTop scores within {TIE_BREAK_MARGIN:.0%} of each other -> "
              f"tie-break prefers SIMPLER: {winner.backbone} + {winner.head}")
    else:
        print(f"\nClear winner: {winner.backbone} + {winner.head}")

    # --- Step C: threshold sweep for the winner ----------------------------
    print(f"\nSweeping confidence threshold for {winner.backbone}+{winner.head} "
          f"(validation split)...")
    ext = build_feature_extractor(winner.backbone)
    X_train = extract_for_split(ext, train_paths)
    X_val = extract_for_split(ext, val_paths)

    pipe = make_head(winner.head).fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    conf = pipe.predict_proba(X_val).max(axis=1)
    errors = val_pred != np.array(y_val)

    threshold, sweep_table = sweep_threshold(conf, errors)
    print(sweep_table.to_string(index=False))
    sweep_table.to_csv(MODELS_DIR / "threshold_sweep.csv", index=False)
    flagged = conf < threshold
    print(f"\nChosen threshold: {threshold:.2f} -> flags "
          f"{flagged.mean():.1%} of tiles, captures "
          f"{(flagged & errors).sum()}/{errors.sum()} real errors")

    # --- Step D: save the single artifact ----------------------------------
    version = f"{winner.backbone}-{winner.head}-v1"
    artifact = {
        "backbone": winner.backbone,
        "head": winner.head,
        "pipeline": pipe,                       # scaler + head, self-contained
        "classes": [str(c) for c in pipe.classes_],
        "threshold": threshold,
        "model_version": version,
        "feature_dim": FEATURE_DIMS[winner.backbone],
        "val_accuracy": float(winner.val_accuracy),
        "val_macro_f1": float(winner.val_macro_f1),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = MODELS_DIR / "classifier_head.pkl"
    with open(out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved artifact: {out}  (model_version={version})")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()