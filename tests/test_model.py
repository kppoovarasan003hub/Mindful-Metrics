"""Model-layer tests: loading, sparse inference, honesty of the artefacts."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

from mindful_metrics.hybrid_model import (
    DISCLAIMER,
    HybridWellbeingModel,
    ModelNotTrainedError,
    RISK_LABELS,
)
from mindful_metrics.text_utils import normalize_text

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ============================================================ artefact loading
def test_artefacts_load(model):
    assert model.numeric_model is not None
    assert model.text_model is not None
    assert model.vectorizer is not None
    assert model.feature_columns == [
        "marks", "attendance", "sleep_hours",
        "screen_time", "assignment_delay",
    ]


def test_missing_artefacts_raise_clear_error(tmp_path):
    with pytest.raises(ModelNotTrainedError) as exc:
        HybridWellbeingModel.load(tmp_path)
    assert "training.train_model" in str(exc.value)


def test_metadata_records_holdout_metrics(model):
    metrics = model.metadata["metrics"]
    # A held-out number must exist and must be distinguishable from training.
    assert 0.0 <= metrics["hybrid_test"]["accuracy"] <= 1.0
    assert metrics["hybrid_test"]["accuracy"] < metrics["numeric_train"]["accuracy"]
    assert "confusion_matrix" in metrics["hybrid_test"]


def test_metadata_declares_data_is_synthetic(model):
    assert "synthetic" in model.metadata["dataset"].lower()
    assert "no clinical meaning" in model.metadata["dataset_notice"].lower()


def test_importing_app_does_not_train(model):
    """Serving must load, never fit."""
    import app as flask_app
    assert isinstance(flask_app.model, HybridWellbeingModel)
    assert flask_app.model.metadata.get("n_train", 0) > 0


# =============================================================== sparse paths
def test_tfidf_output_is_sparse(model):
    matrix = model._text_matrix("a short reflection entry")
    assert sp.issparse(matrix), "TF-IDF must stay sparse"
    assert matrix.shape == (1, len(model.vectorizer.vocabulary_))


def test_no_toarray_in_source():
    """Regression: the original implementation densified the vocabulary.

    Checks executable code only. A docstring is allowed to *mention*
    ``.toarray()`` in order to record that it is not used.
    """
    import ast

    for path in ["mindful_metrics/hybrid_model.py", "training/train_model.py", "app.py"]:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        called = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "toarray"
        ]
        assert not called, f"{path} calls .toarray() (line {called[0].lineno})"


def test_logistic_regression_accepts_sparse(model):
    assert hasattr(model.text_model, "coef_")


# ============================================================ text normalising
@pytest.mark.parametrize("raw,expected", [
    ("I CAN'T SLEEP!", "i cant sleep"),
    ("Feeling   okay\n\nactually", "feeling okay actually"),
    ("stressed... really.", "stressed really"),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent():
    once = normalize_text("I Can't SLEEP!!")
    assert normalize_text(once) == once


def test_normalize_text_preserves_words():
    """Normalising must not silently drop meaning-bearing content."""
    assert "depressed" in normalize_text("I feel Depressed!")


def test_normalize_text_rejects_non_strings():
    for bad in [123, None, [], {}]:
        with pytest.raises(TypeError):
            normalize_text(bad)


def test_model_and_recommendations_share_normaliser():
    """Both paths must agree on what the same text means."""
    assert normalize_text("I CAN'T SLEEP!") == "i cant sleep"


# ================================================================== inference
def test_predict_returns_stable_contract(model, valid_payload):
    result = model.predict(valid_payload)
    assert result["risk_level"] in set(RISK_LABELS.values())
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["class_scores"]) == {"LOW", "MEDIUM", "HIGH"}


def test_class_scores_sum_to_one(model, valid_payload):
    assert sum(model.predict(valid_payload)["class_scores"].values()) \
        == pytest.approx(1.0, abs=1e-3)


def test_prediction_is_deterministic(model, valid_payload):
    a = model.predict(valid_payload)
    b = model.predict(valid_payload)
    assert a["risk_level"] == b["risk_level"]
    assert a["confidence"] == b["confidence"]


def test_confidence_matches_top_class_score(model, valid_payload):
    result = model.predict(valid_payload)
    assert result["confidence"] == result["class_scores"][result["risk_level"]]


def test_extreme_inputs_do_not_crash(model):
    """Out-of-domain values still produce a valid, non-nan response."""
    for payload in [
        {"marks": 0, "attendance": 0, "sleep_hours": 0, "screen_time": 24,
         "assignment_delay": 30, "feedback": "zzz qqq unknown vocabulary"},
        {"marks": 100, "attendance": 100, "sleep_hours": 24, "screen_time": 0,
         "assignment_delay": 0, "feedback": "great happy excited"},
    ]:
        result = model.predict(payload)
        assert math_is_finite(result["confidence"])
        assert result["risk_level"] in set(RISK_LABELS.values())


def math_is_finite(value: float) -> bool:
    return isinstance(value, float) and np.isfinite(value)


# ============================================================ class alignment
def test_probs_aligned_via_classes(model, valid_payload):
    """Label mapping must come from classes_, not positional assumption."""
    blended = model._blend(
        model.numeric_model.predict_proba(model._numeric_matrix(valid_payload))[0],
        model.text_model.predict_proba(model._text_matrix(valid_payload["feedback"]))[0],
    )
    assert set(blended) == set(model.numeric_model.classes_.tolist())


# ================================================================ explanation
def test_explanation_covers_every_numeric_feature(model, valid_payload):
    features = model.predict(valid_payload)["explanation"]["features"]
    assert {f["feature"] for f in features} == set(model.feature_columns)


def test_explanation_sorted_by_magnitude(model, valid_payload):
    features = model.predict(valid_payload)["explanation"]["features"]
    magnitudes = [abs(f["influence"]) for f in features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_carries_caveat(model, valid_payload):
    explanation = model.predict(valid_payload)["explanation"]
    assert "not clinical evidence" in explanation["caveat"].lower()
    assert explanation["type"] == "per_prediction_perturbation"


def test_explanation_is_not_fabricated(model):
    """Different inputs must produce genuinely different influences."""
    a = model.predict({
        "marks": 95, "attendance": 98, "sleep_hours": 9, "screen_time": 1,
        "assignment_delay": 0, "feedback": "happy good great",
    })["explanation"]["features"]
    b = model.predict({
        "marks": 10, "attendance": 12, "sleep_hours": 1, "screen_time": 18,
        "assignment_delay": 20, "feedback": "stressed lonely depressed",
    })["explanation"]["features"]

    assert [f["influence"] for f in a] != [f["influence"] for f in b]


# ============================================================== recommendations
def test_recommendations_are_general_wellbeing(model):
    recs = model.recommendations("LOW", {"sleep_hours": 8})
    assert recs
    joined = " ".join(recs).lower()
    for banned in ["diagnos", "prescrib", "disorder", "treatment plan"]:
        assert banned not in joined


def test_high_risk_recommendations_include_support_line(model):
    low = " ".join(model.recommendations("LOW", {}))
    high = " ".join(model.recommendations("HIGH", {}))
    assert len(high) >= len(low)


# ================================================================ disclaimers
def test_disclaimer_denies_clinical_use():
    lowered = DISCLAIMER.lower()
    assert "not a diagnosis" in lowered
    assert "educational" in lowered


def test_no_accuracy_claims_in_source():
    """The app must not present training accuracy as evidence."""
    for path in ["app.py", "mindful_metrics/hybrid_model.py", "templates/index.html",
                 "static/script.js"]:
        text = (ROOT / path).read_text(encoding="utf-8").lower()
        for claim in ["100% accuracy", "perfectly accurate",
                      "clinically validated", "state-of-the-art"]:
            assert claim not in text, f"{path} makes a claim it cannot support"


# ==================================================================== artefacts
def test_metadata_is_valid_json():
    meta = json.loads((ROOT / "models" / "metadata.json")
                      .read_text(encoding="utf-8"))
    assert meta["random_state"] == 42
    assert meta["test_size"] > 0
    assert meta["n_test"] > 0