# 📊 Project Achievements — All Measured Results

Every number in this file was produced by running the actual pipeline and can
be reproduced: `python scripts/train_head.py` → `python scripts/evaluate.py`.
Raw evidence lives in `models/` (bake-off table, threshold sweep, per-tile
predictions).

---

## 🎯 Model Performance (210-tile held-out eval set)

| Metric | Result |
|---|---|
| Eval accuracy | **91.9%** (193 / 210 correct) |
| Eval macro-F1 | **0.920** |
| Validation → eval gap | **+2.0 points** (accuracy 89.9% → 91.9%) — selection choices generalized, no overfitting |
| Perfect classes | Forest 30/30, Residential 30/30 (100% recall) |
| Hardest class | River — 0.767 recall (confused with Highway ×5, the predicted hard pair) |

## 🚦 Uncertainty Policy (the system's key feature)

| Metric | Result |
|---|---|
| Threshold | **0.85** — derived by sweeping on a validation split, not guessed |
| Tiles flagged "uncertain" | **14.8%** of the eval set |
| Model errors captured by the flag | **70.6%** (12 of 17) |
| Error rate among flagged tiles | **38.7%** |
| Error rate among unflagged tiles | **2.8%** |
| Risk ratio | **~13.9×** — a flagged tile is ~14× more likely to be wrong |
| **Net effect** | **~85% of tiles auto-processed at 97.2% accuracy**; the risky 15% routed to review |

## 🏆 Model Selection (measured, not assumed)

| Metric | Result |
|---|---|
| Combinations tested | **9** (3 backbones × 3 heads: ResNet18, MobileNetV2, SqueezeNet1_1 × LogReg, calibrated SVM, MLP) |
| Spread of all 9 results | 88.1% – 89.6% macro-F1 (within ~1.5 points = statistical noise on 158 images) |
| Winner | **SqueezeNet1_1 + Logistic Regression** — top score AND simplest/fastest via pre-agreed tie-break rule |
| Head training time | **0.1 s** (vs 42–45 s for the SVM variants) |
| Training data | 892 tiles (85% of candidate_tiles); 158 held out for validation |

## ⚡ Performance (measured per-tile with `time.perf_counter`, all 210 tiles)

| Metric | Result |
|---|---|
| Median latency | **13.4 ms/tile** |
| p95 latency | **13.9 ms** |
| Max latency | **15.0 ms** (no outliers — tight distribution) |
| Throughput | **~75 tiles/s** on a single CPU thread |
| Full 210-tile evaluation | **4 seconds** end-to-end |
| Model load (one-time at startup) | ~2.3 s |

## 🔁 Reproducibility & Data Integrity

| Achievement | Evidence |
|---|---|
| **Deterministic pipeline** | Every randomness source seeded; a full retrain reproduced every reported number exactly across **3 independent runs** (bake-off table, threshold sweep, eval metrics — bit-for-bit identical) |
| **Zero data leakage** | Content-hash check: **0** byte-identical overlaps between the 210 eval tiles and the 1,050 training tiles |
| **Perfect label coverage** | Every eval image has exactly one ground-truth row in `eval_labels.csv` |
| **Dimension audit** | All **1,260** tiles programmatically verified to be exactly 64×64 |
| **Offline verified** | Tests + full evaluation passed with Wi-Fi physically disconnected |

## ✅ Quality Gates

| Achievement | Evidence |
|---|---|
| **Automated test suite** | **5/5 tests passing** (predict contract, garbage-input rejection → ValueError, determinism, preprocessing shapes, DB round-trip in an isolated temp DB) |
| **API behavior verified** | Happy path (200), malformed input (400 with clear message), liveness probe (`/health` → model_loaded + db_ok) |
| **Full audit trail** | Every prediction stored with filename, SHA-256 content hash, label, confidence, **full probability vector** (JSON), uncertain flag, model version, timestamp |
| **Traceability** | Every API response includes `prediction_id` → the exact SQLite row |
| **Fresh-start proof** | Database auto-creates on first run; first request after clean clone returns `prediction_id: 1` |

## 🧰 Engineering Practices Demonstrated

- **Shared single-source-of-truth inference path** — the eval script calls the exact same `predict()` the live server uses (no train/serve skew)
- **Honest scope control** — stubbed features explicitly documented (batch endpoint, query API, per-class calibration) rather than hidden
- **Honest limitations documented before being asked** — 5/17 errors were confident mistakes no static threshold can catch; SQLite single-writer and ~75 tiles/s are the first things to break at production load
- **One self-contained artifact** (`classifier_head.pkl`, 42 KB): backbone name + trained head + class list + threshold + version + training metadata

---

*All numbers from a 2026 MacBook Air, CPU-only, Python 3.12, torch 2.5.1 —
no GPU, no internet at inference time.*
