# 🛰️ Satellite Tile Intelligence API

> **CPU-friendly computer vision inference API for satellite land-use classification using PyTorch, transfer learning, FastAPI, confidence-based uncertainty detection, and SQLite.**

An end-to-end **Computer Vision + Machine Learning inference system** for classifying satellite image tiles into **7 land-use categories**.

The project focuses not only on model accuracy, but also on **reproducibility, uncertainty detection, API inference, prediction traceability, data integrity, and production-oriented ML engineering**.

<img width="5010" height="5605" alt="diagram" src="https://github.com/user-attachments/assets/a82d1fed-610d-4816-9285-fa5275da0817" />


---

## 🛠️ Tech Stack

### Programming & ML

<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50" height="50" alt="Python" title="Python"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="50" height="50" alt="PyTorch" title="PyTorch"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="50" height="50" alt="NumPy" title="NumPy"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="50" height="50" alt="Pandas" title="Pandas"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="50" height="50" alt="OpenCV" title="OpenCV"/>
</p>

**Python · PyTorch · Torchvision · NumPy · Pandas · Pillow · scikit-learn**

### Backend & API

<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" width="50" height="50" alt="FastAPI" title="FastAPI"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/sqlite/sqlite-original.svg" width="50" height="50" alt="SQLite" title="SQLite"/>
</p>

**FastAPI · REST API · SQLite · Pydantic · Uvicorn**

### Computer Vision & Machine Learning

**Computer Vision · Image Classification · Transfer Learning · CNNs · Feature Extraction · Logistic Regression · SVM · MLP · Model Evaluation · Confidence Scoring · Uncertainty Detection**

### Engineering & Quality

**pytest · SHA-256 · Deterministic Pipelines · Data Leakage Detection · Model Versioning · Reproducible Experiments · API Validation**

---

## 🚀 Highlights

- 🎯 **91.9% evaluation accuracy** — 193 / 210 tiles classified correctly
- 📊 **0.920 macro-F1** on the held-out evaluation set
- 🧠 Evaluated **9 model combinations** across 3 CNN backbones × 3 classifier heads
- 🏆 Selected **SqueezeNet 1.1 + Logistic Regression** for lightweight CPU inference
- ⚡ **13.4 ms median inference latency per tile**
- 🚦 Confidence-based uncertainty detection captures **70.6% of model errors**
- 🔍 Only **14.8% of evaluation tiles** are flagged for review
- 🌐 FastAPI REST API for real-time inference
- 🗄️ SQLite prediction audit trail with SHA-256 hashes and model metadata
- 🔁 Deterministic and reproducible training/evaluation pipeline
- 🔒 Zero byte-identical overlap between training and evaluation tiles
- 🧪 Automated test suite with **5/5 tests passing**
- 💻 Designed and verified for **CPU-only inference**
- 📦 Self-contained trained model artifact

---

## 🧠 What This Project Does

The system takes a satellite image tile and predicts its land-use category.

```text
Satellite Image
      │
      ▼
Image Preprocessing
      │
      ▼
Pretrained CNN Backbone
      │
      ▼
Feature Extraction
      │
      ▼
Logistic Regression Classifier
      │
      ▼
Class Probabilities
      │
      ├───────────────┐
      ▼               ▼
Prediction       Confidence
                      │
                      ▼
              Uncertainty Check
                      │
              ┌───────┴───────┐
              ▼               ▼
          Confident         Uncertain
          Prediction        → Review
              │
              ▼
       SQLite Audit Trail
```

The live API and evaluation pipeline use the **same inference function**, avoiding train/serve skew.

---

## 📊 Model Performance

### Held-out Evaluation Set

| Metric                      |                                Result |
| --------------------------- | ------------------------------------: |
| Evaluation accuracy         |                             **91.9%** |
| Correct predictions         |                         **193 / 210** |
| Macro-F1                    |                             **0.920** |
| Validation accuracy         |                             **89.9%** |
| Validation → evaluation gap |                       **+2.0 points** |
| Perfect classes             | **Forest, Residential — 100% recall** |
| Hardest class               |              **River — 0.767 recall** |

The River class was the most difficult category, with the main confusion occurring between **River and Highway**.

---

## 🚦 Confidence-Based Uncertainty Detection

Instead of treating every prediction as equally reliable, the system uses a validation-derived confidence threshold.

### Threshold

**0.85**

The threshold was selected through a validation sweep rather than manually guessed.

| Metric                                    |              Result |
| ----------------------------------------- | ------------------: |
| Evaluation tiles flagged uncertain        |           **14.8%** |
| Model errors captured                     | **70.6% (12 / 17)** |
| Error rate among flagged tiles            |           **38.7%** |
| Error rate among unflagged tiles          |            **2.8%** |
| Risk ratio                                |          **~13.9×** |
| Automatically processed tiles             |            **~85%** |
| Accuracy of automatically processed tiles |           **97.2%** |

### Why this matters

The system provides a simple **human-in-the-loop workflow**:

```text
Prediction
    │
    ▼
Confidence ≥ 0.85?
    │
 ┌──┴──┐
Yes    No
 │      │
 ▼      ▼
Auto   Human
Process Review
```

This makes the model more useful for workflows where **knowing when the model may be wrong** is as important as the prediction itself.

---

## 🏆 Model Selection

The project evaluated **9 combinations**:

### CNN Backbones

* ResNet18
* MobileNetV2
* SqueezeNet 1.1

### Classifier Heads

* Logistic Regression
* Calibrated SVM
* MLP

```text
3 CNN Backbones
       ×
3 Classifier Heads
       =
9 Model Combinations
```

### Selected Model

**SqueezeNet 1.1 + Logistic Regression**

The model was selected using measured validation performance and a pre-defined simplicity/efficiency tie-break rule.

| Metric                    |                  Result |
| ------------------------- | ----------------------: |
| Model combinations tested |                   **9** |
| Macro-F1 range            |       **88.1% – 89.6%** |
| Selected backbone         |      **SqueezeNet 1.1** |
| Selected classifier       | **Logistic Regression** |
| Head training time        |         **0.1 seconds** |
| Training tiles            |                 **892** |
| Validation tiles          |                 **158** |

The CNN backbone uses **ImageNet-pretrained weights with frozen feature extraction**, while the classifier head is trained on the extracted features.

---

## ⚡ CPU Performance

Inference was benchmarked using `time.perf_counter` across all 210 evaluation tiles.

| Metric                   |                 Result |
| ------------------------ | ---------------------: |
| Median latency           |     **13.4 ms / tile** |
| p95 latency              |     **13.9 ms / tile** |
| Maximum latency          |            **15.0 ms** |
| Throughput               | **~75 tiles / second** |
| Full 210-tile evaluation |         **~4 seconds** |
| Model loading            |       **~2.3 seconds** |

The system was tested on a **MacBook Air using CPU-only inference**, with no GPU dependency.

---

## 🌐 REST API

The trained model is exposed through a lightweight **FastAPI** service.

### Endpoints

| Endpoint    | Method | Purpose                         |
| ----------- | ------ | ------------------------------- |
| `/health`   | `GET`  | API and model health check      |
| `/classify` | `POST` | Classify a satellite image tile |

### Example Request

```bash
curl -X POST \
  -F "file=@sample_tile.jpg" \
  http://localhost:8000/classify
```

### Example Response

```json
{
  "prediction_id": 1,
  "filename": "sample_tile.jpg",
  "prediction": "Forest",
  "confidence": 0.943,
  "uncertain": false,
  "model_version": "squeezenet1_1-logreg-v1"
}
```

The API also validates malformed input and exposes a liveness check through `/health`.

---

## 🗄️ Prediction Audit Trail

Every prediction is persisted in SQLite.

Stored information includes:

* Filename
* SHA-256 content hash
* Predicted class
* Confidence score
* Full probability vector
* Uncertainty flag
* Model version
* Timestamp
* Prediction ID

This provides **prediction traceability and reproducibility** rather than treating the API as a stateless black box.

---

## 🔐 Data Integrity

The project includes explicit data-integrity checks.

### Verified

* **0 byte-identical overlaps** between training and evaluation images
* Every evaluation image has exactly one ground-truth label
* All **1,260 tiles** verified as exactly **64 × 64**
* Every randomness source seeded
* Three independent retraining runs reproduced the reported results
* Evaluation completed successfully with **Wi-Fi physically disconnected**

---

## 🧪 Testing & Quality Gates

The project includes automated tests covering:

* Prediction contract
* Garbage-input rejection
* Deterministic inference
* Image preprocessing shapes
* SQLite database round-trip

### Test Result

**5 / 5 tests passing**

API behavior was also verified for:

```text
Valid request       → 200
Malformed input     → 400
Health check        → model_loaded + db_ok
Clean database      → prediction_id: 1
```

---

## 🔁 Reproducibility

The complete evaluation can be reproduced with:

```bash
python scripts/train_head.py
python scripts/evaluate.py
```

The evaluation pipeline produces:

* Model bake-off results
* Threshold sweep
* Per-tile predictions
* Evaluation metrics
* Performance measurements

The evaluation script also uses the **same `predict()` function as the FastAPI server**, ensuring a single source of truth for inference.

---

## 📁 Project Structure

```text
satellite-tile-intelligence-api/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── database.py
│   └── ...
│
├── scripts/
│   ├── train_head.py
│   ├── evaluate.py
│   └── ...
│
├── tests/
│   └── ...
│
├── models/
│   ├── classifier_head.pkl
│   └── ...
│
├── data/
│   └── ...
│
├── ACHIEVEMENTS.md
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧩 Key Engineering Decisions

### 1. Transfer Learning

Instead of training a CNN from scratch, pretrained ImageNet models are used as frozen feature extractors.

This reduces training requirements and makes the system suitable for CPU-oriented deployment.

### 2. Lightweight Model Selection

Multiple CNN architectures and classifier heads were evaluated rather than assuming the largest model would perform best.

The final selection favored:

* Strong validation performance
* Low inference cost
* Simple deployment
* Fast classifier training

### 3. Uncertainty-Aware Predictions

The system does not blindly trust every prediction.

Predictions below the validated confidence threshold are marked as **uncertain** and can be routed for human review.

### 4. Single Inference Path

Both evaluation and production API inference use the same prediction logic.

```text
Evaluation ──────┐
                 ├──► predict()
FastAPI ─────────┘
```

This reduces the risk of **train/serve skew**.

### 5. Prediction Traceability

Each API prediction receives a unique `prediction_id` and is stored with its input hash, confidence, probabilities, timestamp, and model version.

---

## 📦 Model Artifact

The final model is stored as a single self-contained artifact:

```text
classifier_head.pkl
```

The artifact contains:

* Backbone name
* Trained classifier
* Class list
* Confidence threshold
* Model version
* Training metadata

Artifact size:

**~42 KB**

---

## ⚠️ Known Limitations

This project intentionally documents its current limitations.

### Current limitations

* Batch inference endpoint is not implemented
* Prediction query API is not implemented
* Per-class confidence calibration is not implemented
* SQLite uses a single-writer architecture
* CPU throughput is approximately 75 tiles/second
* **5 of 17 evaluation errors were confident mistakes**, meaning a static confidence threshold cannot detect every incorrect prediction

These are documented as future engineering improvements rather than hidden limitations.

---

## 🔮 Future Improvements

* [ ] Batch image inference
* [ ] Prediction history/query API
* [ ] Per-class probability calibration
* [ ] PostgreSQL for higher-concurrency workloads
* [ ] Docker-based production deployment
* [ ] Model monitoring and drift detection
* [ ] Automated model retraining pipeline
* [ ] Explainable predictions with Grad-CAM
* [ ] Prometheus/Grafana inference monitoring
* [ ] Cloud deployment
* [ ] CI/CD pipeline

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Pratikchetry/satellite-tile-intelligence-api.git
cd satellite-tile-intelligence-api
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 🧪 Run Tests

```bash
pytest -q
```

Expected result:

```text
5 passed
```

---

## 📈 Reproduce the Evaluation

```bash
python scripts/train_head.py
python scripts/evaluate.py
```

The generated evaluation artifacts can be inspected under:

```text
models/
```

---

## 🐳 Docker

Build the image:

```bash
docker build -t satellite-tile-intelligence-api .
```

Run the container:

```bash
docker run -p 8000:8000 satellite-tile-intelligence-api
```

Then open:

```text
http://localhost:8000/docs
```

---

## 📊 Project Achievements

For the complete measured results, reproducibility evidence, performance benchmarks, model selection results, and quality gates, see:

**[ACHIEVEMENTS.md](ACHIEVEMENTS.md)**

---

## 🎯 Skills Demonstrated

This project demonstrates practical experience with:

**Machine Learning**

* Model selection
* Classification
* Transfer learning
* Feature extraction
* Logistic Regression
* SVM
* MLP
* Model evaluation

**Computer Vision**

* Satellite imagery
* Image preprocessing
* CNN feature extraction
* Image classification
* Confidence scoring
* Uncertainty detection

**ML Engineering**

* Reproducible pipelines
* Data leakage prevention
* Model packaging
* Model versioning
* CPU inference optimization
* Prediction traceability

**Backend Engineering**

* FastAPI
* REST APIs
* Request validation
* Health checks
* SQLite persistence
* Docker

**Software Quality**

* Automated testing
* Deterministic inference
* Data integrity validation
* Audit trails
* Error handling

---

## 📌 Why This Project?

This project was designed to demonstrate more than simply training an image classifier.

The focus is on building a **complete ML inference system** that connects:

```text
Computer Vision
      +
Machine Learning
      +
Model Evaluation
      +
Uncertainty Detection
      +
REST API
      +
Database Persistence
      +
Testing
      +
Reproducibility
```

The result is a compact, CPU-friendly system that can move from **model experimentation → validated inference → API deployment → prediction auditing**.

---

## 👨‍💻 Author

**Pratik Chetry**

Data Scientist | Agentic AI Systems · LLMs · NLP · MCP · Python · SQL · Power BI

GitHub: **[Pratikchetry](https://github.com/Pratikchetry)**

---

## 📜 License

This project is available for educational and portfolio purposes.

---

<p align="center">
  <b>Built with Python, PyTorch, FastAPI and a focus on reliable ML inference.</b>
</p>
