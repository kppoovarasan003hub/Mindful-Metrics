"""Train the demo models and write artefacts to ``models/``.

Run explicitly::

    python -m training.train_model

The Flask app never trains anything; it only loads what this script
produces (``app.py`` refuses to start if the artefacts are missing).

Evaluation honesty
------------------
The vectorizer, the RandomForest and the LogisticRegression are fitted on
the *training split only*. The vectorizer in particular must not see the
test split, otherwise every text metric below is inflated by leakage.

Three numbers are reported and kept clearly separated:

* ``train``  -- fitted-data metrics. High by construction. Not evidence
  of anything.
* ``test``   -- held-out metrics. The only generalisation estimate here.
* ``permutation`` -- how much held-out accuracy drops when a single
  feature is shuffled. This is the measurement that backs the
  "what influenced this result" panel.

Read MODEL_CARD.md before quoting any of these numbers. They describe how
well the model recovers *this generator's* pattern. They say nothing
about students.
"""

from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from mindful_metrics.text_utils import normalize_text
from training.make_dataset import (
    CSV_PATH,
    FEATURE_COLUMNS,
    build_rows,
    write_csv,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Must match the hybrid weighting in hybrid_model.py. Defined here so
# training can record the value that was actually used at fit time.
WEIGHT_NUMERIC = 0.7
WEIGHT_TEXT = 0.3

RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def _load_dataset() -> list[dict]:
    """Load the synthetic CSV if present, otherwise generate it."""
    if CSV_PATH.exists():
        import csv

        with CSV_PATH.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    rows = build_rows()
    write_csv(rows)
    print(f"Generated missing dataset -> {CSV_PATH}")
    return rows


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2],
            target_names=["LOW", "MEDIUM", "HIGH"],
            zero_division=0, output_dict=True),
    }


def main() -> None:
    rows = _load_dataset()

    y = np.array([int(r["risk_label"]) for r in rows])
    X_numeric = np.array(
        [[float(r[c]) for c in FEATURE_COLUMNS] for r in rows], dtype=float
    )
    X_text = [normalize_text(r["feedback"]) for r in rows]

    X_num_tr, X_num_te, y_tr, y_te = train_test_split(
        X_numeric, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    idx_tr, idx_te = train_test_split(
        np.arange(len(rows)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    X_text_tr = [X_text[i] for i in idx_tr]
    X_text_te = [X_text[i] for i in idx_te]

    print(f"rows={len(rows)}  train={len(idx_tr)}  test={len(idx_te)}")
    print("stratified split, random_state=%d" % RANDOM_STATE)

    # --- Text branch: fit vectorizer on TRAIN ONLY, keep it sparse -------
    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X_text_tr_mat = vectorizer.fit_transform(X_text_tr)   # sparse csr
    X_text_te_mat = vectorizer.transform(X_text_te)

    print(f"tf-idf vocabulary: {len(vectorizer.vocabulary_)} terms "
          f"(dense array would be "
          f"{len(idx_te) * len(vectorizer.vocabulary_):,} cells)")

    # --- Numeric branch --------------------------------------------------
    numeric_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
    )
    numeric_model.fit(X_num_tr, y_tr)

    text_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    text_model.fit(X_text_tr_mat, y_tr)   # accepts sparse directly

    # --- Branch metrics --------------------------------------------------
    numeric_train = _metrics(y_tr, numeric_model.predict(X_num_tr))
    numeric_test = _metrics(y_te, numeric_model.predict(X_num_te))
    text_train = _metrics(y_tr, text_model.predict(X_text_tr_mat))
    text_test = _metrics(y_te, text_model.predict(X_text_te_mat))

    hybrid_test = _metrics(y_te, _hybrid_labels(
        numeric_model, text_model, X_num_te, X_text_te_mat))

    # --- Permutation importance on HELD-OUT data -------------------------
    perm = _permutation_importance_holdout(
        numeric_model, X_num_te, y_te, vectorizer, text_model, X_text_te_mat
    )

    # --- Feature distribution, used by the app as a neutral baseline -----
    medians = np.median(X_numeric, axis=0).tolist()

    print("\n--- NUMERIC BRAND (RandomForest) ---")
    print("train accuracy :", numeric_train["accuracy"])
    print("test  accuracy :", numeric_test["accuracy"], " <- held-out")
    print("\n--- TEXT BRAND (TF-IDF + LogisticRegression) ---")
    print("train accuracy :", text_train["accuracy"])
    print("test  accuracy :", text_test["accuracy"], " <- held-out")
    print("\n--- HYBRID ENSEMBLE ---")
    print("test  accuracy :", hybrid_test["accuracy"], " <- held-out")
    print("test  macro F1 :", hybrid_test["f1_macro"])
    print("confusion matrix (rows=true LOW/MED/HIGH):")
    for row in hybrid_test["confusion_matrix"]:
        print("   ", row)

    print("\n--- PERMUTATION IMPORTANCE (held-out accuracy drop) ---")
    for name, drop in zip(FEATURE_COLUMNS, perm["importances_mean"]):
        print(f"  {name:20s} {drop:+.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(numeric_model, MODEL_DIR / "numeric_model.joblib")
    joblib.dump(text_model, MODEL_DIR / "text_model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")

    metadata = {
        "trained_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_rows": len(rows),
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
        "dataset": "synthetic (generated by training/make_dataset.py)",
        "dataset_notice": (
            "Synthetic data generated by a random generator. Not real "
            "students. Metrics describe recovery of this generator's "
            "pattern only and carry no clinical meaning."
        ),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_medians": medians,
        "risk_labels": RISK_LABELS,
        "weights": {"numeric": WEIGHT_NUMERIC, "text": WEIGHT_TEXT},
        "vocabulary_size": len(vectorizer.vocabulary_),
        "metrics": {
            "numeric_train": numeric_train,
            "numeric_test": numeric_test,
            "text_train": text_train,
            "text_test": text_test,
            "hybrid_test": hybrid_test,
            "permutation_importance_holdout": perm,
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved artefacts -> {MODEL_DIR}")
    print(f"Saved metrics   -> {METADATA_PATH}")


def _hybrid_labels(numeric_model, text_model, X_num, X_text_mat) -> np.ndarray:
    """Apply the same 0.7/0.3 blend the app uses."""
    p_num = numeric_model.predict_proba(X_num)
    p_txt = text_model.predict_proba(X_text_mat)
    blended = WEIGHT_NUMERIC * p_num + WEIGHT_TEXT * p_txt
    return numeric_model.classes_[np.argmax(blended, axis=1)]


def _permutation_importance_holdout(
    numeric_model, X_num_te, y_te, vectorizer, text_model, X_text_te_mat
) -> dict:
    """Accuracy drop when each numeric feature is shuffled on held-out data.

    Measured on the hybrid ensemble rather than the RandomForest alone, so
    the number shown in the UI reflects the system actually being served.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    baseline = accuracy_score(
        y_te,
        _hybrid_labels(numeric_model, text_model, X_num_te, X_text_te_mat),
    )

    means, stds = [], []
    n_repeats = 12
    for col in range(X_num_te.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_num_te.copy()
            rng.shuffle(X_shuffled[:, col])
            score = accuracy_score(
                y_te,
                _hybrid_labels(numeric_model, text_model,
                               X_shuffled, X_text_te_mat),
            )
            drops.append(baseline - score)
        means.append(float(np.mean(drops)))
        stds.append(float(np.std(drops)))

    return {
        "baseline_accuracy": round(float(baseline), 4),
        "importances_mean": [round(m, 4) for m in means],
        "importances_std": [round(s, 4) for s in stds],
        "n_repeats": n_repeats,
        "note": (
            "Positive = shuffling this feature hurt held-out accuracy, so "
            "the ensemble relied on it. Negative = it did not help."
        ),
    }


if __name__ == "__main__":
    main()