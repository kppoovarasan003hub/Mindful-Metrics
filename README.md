# MindfulMetrics — Hybrid Student Wellbeing Classifier

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/framework-Flask%203.1.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.9.0-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-129%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Educational%20Demo-lightgrey.svg)](LICENSE)

An educational demonstration of a **hybrid machine-learning classifier**:
a `RandomForestClassifier` over five numeric behavioral inputs blended with a sparse `TfidfVectorizer` + `LogisticRegression` over a student reflection text.

Repository: **[https://github.com/kppoovarasan003hub/Mindful-Metrics](https://github.com/kppoovarasan003hub/Mindful-Metrics)**

---

> ## ⚠️ Important Limitation & Non-Clinical Disclaimer
>
> **This is not a mental-health assessment.** It is not a clinical diagnosis, a screening tool, or a validated medical instrument. It has never been tested against real clinical records or real student mental-health outcomes.
>
> Its held-out accuracy on synthetic data is approximately **63%** — it is wrong about one classification in three. **Do not use it, or any output of it, to make decisions about a real person.**
>
> See [MODEL_CARD.md](MODEL_CARD.md) for the full model card, held-out metrics, confusion matrices, and limitations.

---

## 🎯 Project Highlights

This project demonstrates a production-grade, honest machine-learning workflow end-to-end:

| Concern | How it is handled |
|---|---|
| **Data** | 100% synthetic, procedurally generated with fixed seed and deliberate label noise |
| **Split** | Stratified 80/20 train/test split (`random_state=42`), zero test data leakage |
| **Text Features** | TF-IDF with sublinear scaling, entirely **sparse** (no memory-heavy `.toarray()`) |
| **Ensembling** | Calibrated 0.7 numeric / 0.3 text probability blend |
| **Evaluation** | Held-out accuracy, macro Precision/Recall/F1, confusion matrix, permutation importance |
| **Explainability** | Genuine per-prediction input perturbation, not static global averages |
| **Serving** | Flask loads persisted artefacts; **strictly never trains at import time** |
| **Validation** | Centralized, strict server-side validation; every user error returns HTTP `400` with actionable details |
| **Privacy** | Reflection text processed purely in-memory; never stored, logged, or echoed back |

---

## 🏛️ Architecture

```
                    ┌──────────────┐
    Browser ───────►│    Flask     │
                    └──────┬───────┘
                           │  POST /predict
                           ▼
                    ┌──────────────┐
                    │  Validation  │  ranges, types, NaN/Inf,
                    └──────┬───────┘  emptiness → 400 with details
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌───────────────┐         ┌───────────────┐
      │ Numeric branch│         │  Text branch  │
      │ RandomForest  │         │ TF-IDF → LR   │  sparse throughout
      └───────┬───────┘         └───────┬───────┘
              │  0.7                    │  0.3
              └────────────┬────────────┘
                           ▼
                    ┌──────────────┐
                    │ Hybrid score │  + per-prediction explanation
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   Frontend   │  calibrated, non-clinical UI
                    └──────────────┘
```

### 📁 Project Layout

```
Mindful-Metrics/
├── app.py                     # Flask serving entrypoint (serving only, no training)
├── mindful_metrics/           # Core application package
│   ├── __init__.py
│   ├── config.py              # Validation bounds, feature definitions, and limits
│   ├── hybrid_model.py        # Inference pipeline, ensemble scoring, perturbation
│   ├── text_utils.py          # Text normalization and preprocessing
│   └── validation.py          # Centralized request validation schema
├── training/                  # Offline model training pipeline
│   ├── __init__.py
│   ├── make_dataset.py        # Synthetic dataset generator with controlled noise
│   └── train_model.py         # Explicit training, evaluation, and artifact export
├── data/                      # Dataset directory (contains .gitkeep; CSV generated)
│   └── synthetic_wellbeing.csv
├── models/                    # Serialized joblib models and metadata (reproducible)
│   ├── metadata.json          # Metrics, confusion matrix, and vocabulary details
│   ├── numeric_model.joblib
│   ├── text_model.joblib
│   └── vectorizer.joblib
├── templates/
│   └── index.html             # Accessible, responsive frontend
├── static/
│   ├── style.css              # Custom styling (glassmorphism, dark palette, responsive)
│   └── script.js              # Client-side validation, async submission, UI rendering
├── tests/                     # Comprehensive test suite (129 unit & integration tests)
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_model.py
│   └── test_validation.py
├── .gitignore                 # Optimized for Python, Flask, ML caches
├── MODEL_CARD.md              # Detailed evaluation metrics, fairness & limitation analysis
├── requirements.txt           # Pinned dependencies for Python 3.12
└── README.md                  # Project documentation
```

---

## 📊 Dataset Notice

> **The dataset is 100% synthetic** and designed exclusively for machine learning workflow demonstration. It must never be used to draw inferences about real student mental health.

Generated by `training/make_dataset.py` from a fixed random seed (`20240917`), featuring controlled label noise so classes are not trivially separable.

To swap with real research or institutional data:
1. Replace `data/synthetic_wellbeing.csv` with validated data.
2. Update the loader in `training/train_model.py`.
3. Re-train, re-evaluate, and update `MODEL_CARD.md` before making any claims.

---

## 🚀 Quickstart & Setup

### Prerequisites
- **Python 3.12** (tested on 3.12.10)
- `git`

### 1. Clone the Repository
```bash
git clone https://github.com/kppoovarasan003hub/Mindful-Metrics.git
cd Mindful-Metrics
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash / CMD):**
```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash
# or: .venv\Scripts\activate.bat   # CMD
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Dataset & Train Models
```bash
# Generate synthetic dataset (optional; train_model runs it automatically if absent)
python -m training.make_dataset

# Fit models and save artefacts to models/ directory (REQUIRED before server launch)
python -m training.train_model
```

### 5. Launch the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## ⚙️ Configuration

Configurable via environment variables (defaults are secure):

| Variable | Default | Purpose |
|---|---|---|
| `FLASK_DEBUG` | `0` | Debug mode. Defaults to **off**. Set `1` only during development. |
| `HOST` | `127.0.0.1` | Network interface binding. |
| `PORT` | `5000` | Port to listen on. |
| `CORS_ORIGINS` | *(empty)* | Comma-separated CORS allowlist. Empty = same-origin only. |

---

## 🧪 Testing

The repository includes a comprehensive test suite of **129 tests** covering API contracts, boundary validation, UTF-8 handling, security posture, and inference consistency:

```bash
# Run full suite
pytest

# Verbose output
pytest -v

# Run specific API tests
pytest tests/test_api.py -v
```

---

## 🔌 API Specification

### `POST /predict`

Scores student metrics and reflection text, returning calibrated risk probabilities and per-prediction explanations.

#### Request Body
```json
{
  "marks": 75,
  "attendance": 85,
  "sleep_hours": 7,
  "screen_time": 4,
  "assignment_delay": 1,
  "feedback": "Sleeping reasonably well and keeping on top of coursework."
}
```

#### Input Boundaries
| Field | Allowed Range | Validation Rules |
|---|---|---|
| `marks` | 0 – 100 | Finite numeric |
| `attendance` | 0 – 100 | Finite numeric |
| `sleep_hours` | 0 – 24 | Finite numeric |
| `screen_time` | 0 – 24 | Finite numeric |
| `assignment_delay` | 0 – 30 days | Finite numeric |
| `feedback` | 1 – 2000 chars | Non-empty string, trimmed |

#### Success Response (`200 OK`)
```json
{
  "risk_level": "MEDIUM",
  "confidence": 0.41,
  "confidence_note": "This is how strongly the demo model leaned toward this label. It is not a certainty and not a clinical probability.",
  "class_scores": {
    "LOW": 0.34,
    "MEDIUM": 0.41,
    "HIGH": 0.25
  },
  "branch_scores": {
    "numeric": { "LOW": 0.35, "MEDIUM": 0.45, "HIGH": 0.20 },
    "text": { "LOW": 0.32, "MEDIUM": 0.32, "HIGH": 0.36 }
  },
  "explanation": {
    "type": "per_prediction_perturbation",
    "features": [
      { "feature": "attendance", "effect": "lowers risk" },
      { "feature": "assignment_delay", "effect": "increases risk" }
    ],
    "caveat": "Local perturbation around input values."
  },
  "recommendations": [
    "Maintain consistent sleep hygiene",
    "Reach out to an academic advisor regarding workload scheduling"
  ],
  "disclaimer": "This is an educational ML demonstration, not a medical evaluation.",
  "is_high_risk": false,
  "support": {
    "heading": "Campus and Community Support",
    "body": "If you or someone you know is feeling overwhelmed, confidential support is available.",
    "options": ["College Counselling Center", "Student Support Services"]
  },
  "privacy_notice": "Your responses are processed in-memory and never retained."
}
```

#### Error Handling
- `400 Bad Request`: Payload validation failures (types, missing fields, out of range). Returns detailed list in `details`.
- `413 Content Too Large`: Payloads exceeding 64 KB.
- `500 Internal Server Error`: Generic safe message without stack traces or path leaks.

---

## 🔒 Privacy & Ethical Design

- **Zero Storage:** No database or persistence layer. Inputs are parsed in memory and discarded after response generation.
- **No Text Logging:** Reflection text is never written to log files, stdout, or error reports.
- **No Third-Party Analytics:** No remote tracking, telemetry, or third-party API dependencies.
- **Transparent Disagreement:** Displays both numeric and text branch predictions side-by-side so discrepancies are evident to the user.

---

## 📄 License

This project is licensed for educational and demonstration purposes. See [LICENSE](LICENSE) or project details for more information.