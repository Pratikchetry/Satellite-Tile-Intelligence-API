"""
scripts/evaluate.py — scores the trained model on the 210-tile eval_set.

Plan §6 rules enforced here:
  - Calls app.classifier's predict() — the EXACT function the live server
    uses. There is no third copy of the pipeline anywhere.
  - eval_set is touched for the FIRST time here (the combo and threshold were
    chosen on candidate_tiles' validation split), so every number this prints
    is a confirmation on genuinely unseen data.

Run from project root:  caffeinate -i python scripts/evaluate.py
"""

import hashlib
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# SqueezeNet = many SMALL conv layers. For single-image CPU inference,
# multi-thread coordination per layer costs more than it saves, so 1 thread
# is typically the FASTEST setting here (batched training is different —
# there, big per-layer work makes threading worthwhile).
torch.set_num_threads(1)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from app.classifier import get_classifier

DATA = PROJECT_ROOT / "data"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    t_start = time.time()
    clf = get_classifier()

    eval_paths = sorted(
        p for p in (DATA / "eval_set").iterdir() if p.suffix.lower() in IMG_EXTS
    )
    labels_df = pd.read_csv(DATA / "eval_labels.csv")

    # --- 1. Integrity: eval must not overlap training data ------------------
    cand_hashes = {
        sha256(p)
        for p in (DATA / "candidate_tiles").rglob("*")
        if p.suffix.lower() in IMG_EXTS
    }
    overlaps = [p.name for p in eval_paths if sha256(p) in cand_hashes]
    print(f"[integrity] eval images: {len(eval_paths)} | "
          f"byte-identical overlaps with candidate_tiles: {len(overlaps)}",
          flush=True)
    if overlaps:
        raise SystemExit(f"FATAL: eval leak — {overlaps[:5]}")

    missing = {p.name for p in eval_paths} - set(labels_df["filename"])
    extra = set(labels_df["filename"]) - {p.name for p in eval_paths}
    if missing or extra:
        raise SystemExit(
            f"FATAL: eval_set/CSV mismatch — missing={sorted(missing)[:5]}, "
            f"extra={sorted(extra)[:5]}"
        )
    print("[integrity] every eval image has exactly one label row\n", flush=True)

    # --- 2. Predict every tile via the SHARED predict() ---------------------
    total = len(eval_paths)
    print(f"Running {total} tiles through the same predict() "
          f"the live server uses...", flush=True)
    records = []
    for i, p in enumerate(eval_paths, 1):
        r = clf.predict(p.read_bytes())
        records.append({"filename": p.name, **r})
        if i % 25 == 0 or i == total:
            elapsed = time.time() - t_start
            eta = elapsed / i * (total - i)
            print(f"  {i}/{total} classified "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)
    preds = pd.DataFrame(records).merge(labels_df, on="filename", how="inner")

    y_true = preds["true_label"]
    y_pred = preds["predicted_label"]

    # --- 3. Headline numbers -------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\nEval accuracy : {acc:.4f}  "
          f"({int((y_true == y_pred).sum())}/{len(preds)})")
    print(f"Eval macro-F1 : {mf1:.4f}")

    # val-vs-eval gap (overfitting check), read from the same single artifact
    with open(PROJECT_ROOT / "models" / "classifier_head.pkl", "rb") as f:
        art = pickle.load(f)
    print(f"val -> eval   : accuracy {art['val_accuracy']:.4f} -> {acc:.4f} "
          f"({acc - art['val_accuracy']:+.4f}), macro-F1 "
          f"{art['val_macro_f1']:.4f} -> {mf1:.4f} "
          f"({mf1 - art['val_macro_f1']:+.4f})")

    # --- 4. Confusion matrix + most-confused pair ----------------------------
    classes = clf.classes
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{c}" for c in classes],
        columns=[f"pred:{c}" for c in classes],
    )
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm_df.to_string())

    row_sums = cm.sum(axis=1)
    if (row_sums > 0).all():
        per_class = pd.Series(
            np.diag(cm) / row_sums, index=classes
        ).sort_values()
        print("\nPer-class recall (worst first):")
        print(per_class.round(3).to_string())

    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    if cm_off.sum() > 0:
        ti, pi = np.unravel_index(np.argmax(cm_off), cm_off.shape)
        print(f"\nMost confused pair: true {classes[ti]} predicted as "
              f"{classes[pi]} — {cm_off[ti, pi]} times")

    # --- 5. Uncertainty flag on UNSEEN data (threshold confirmation) ---------
    errors = (y_true != y_pred).to_numpy()
    uncertain = preds["is_uncertain"].to_numpy()
    n_err = int(errors.sum())
    flag_rate = float(uncertain.mean())
    capture = float((uncertain & errors).sum() / n_err) if n_err else 0.0
    err_if_flagged = float(errors[uncertain].mean()) if uncertain.any() else float("nan")
    err_if_clean = float(errors[~uncertain].mean()) if (~uncertain).any() else float("nan")

    print(f"\nUncertainty flag (threshold={clf.threshold}) on unseen eval data:")
    print(f"  flagged        : {flag_rate:.1%} of tiles")
    print(f"  errors captured: {(uncertain & errors).sum()}/{n_err} = {capture:.1%}")
    print(f"  error rate among flagged tiles  : {err_if_flagged:.1%}")
    print(f"  error rate among unflagged tiles: {err_if_clean:.1%}")
    if err_if_clean > 0:
        print(f"  a flagged tile is ~{err_if_flagged / err_if_clean:.1f}x more "
              f"likely to be wrong than an unflagged one")

    # --- 6. Save per-tile predictions as evidence -----------------------------
    out = PROJECT_ROOT / "models" / "eval_predictions.csv"
    preds.to_csv(out, index=False)
    print(f"\nSaved per-tile predictions: {out}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()