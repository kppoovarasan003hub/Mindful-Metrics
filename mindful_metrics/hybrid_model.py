"""Hybrid numeric + text classifier for the student wellbeing demo.

This module is **inference only**. It loads artefacts produced by
``python -m training.train_model`` and never fits anything, so importing
it (which the Flask app does at start-up) costs a few milliseconds rather
than a model fit.

What the model is
-----------------
A RandomForest over five numeric inputs blended 0.7/0.3 with a
LogisticRegression over TF-IDF of the reflection text. Three classes:
LOW / MEDIUM / HIGH.

What the model is **not**
------------------------
A clinical instrument. It was fitted to a synthetic dataset produced by a
random generator. Its held-out accuracy on that dataset is around 0.63.
That figure describes how well it recovers a *synthetic pattern*; it is
not a claim about students, about mental health, or about anyone's
actual wellbeing. See MODEL_CARD.md.

Honesty notes embedded in the code
----------------------------------
* Class probabilities are aligned through ``classes_`` rather than assumed
  to be in 0/1/2 order. If a class were ever absent from training data the
  naive ``argmax(probs)`` would silently return the wrong label; going
  through ``classes_`` makes that impossible.
* ``explanation`` is a genuine *per-prediction* signal, computed by
  replacing one numeric feature at a time with the training median and
  measuring the change in the predicted class score. It describes this
  model's behaviour on this one input. It is not evidence about a person.
* TF-IDF stays sparse end to end; no ``.toarray()`` anywhere.
"""

from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np

from .text_utils import normalize_text

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

WEIGHT_NUMERIC = 0.7
WEIGHT_TEXT = 0.3

RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# Shown verbatim in the UI and asserted by the tests. Kept as a constant so
# the wording cannot drift between the disclaimer, the API and the page.
DISCLAIMER = (
    "Educational demonstration only. This is an experimental model output "
    "on synthetic data. It is not a diagnosis, a screening result, or a "
    "mental-health assessment, and it cannot determine how you are doing."
)

PRIVACY_NOTICE = (
    "Your entries are processed in memory to produce this demo result. "
    "They are not written to disk, not stored in a database, and not sent "
    "to any third-party service. Nothing is retained after the response "
    "is returned."
)

SUPPORT_LINES = (
    "A trusted friend or family member",
    "Your college counsellor or student support service",
    "A qualified mental-health professional",
    "Local emergency or crisis services if you are in immediate danger",
)

HIGH_RISK_MESSAGE = (
    "A higher-risk pattern was detected by this demonstration model."
)


class ModelNotTrainedError(RuntimeError):
    """Raised when the expected artefacts are missing."""


class HybridWellbeingModel:
    """Loads trained artefacts and answers prediction requests."""

    def __init__(
        self,
        numeric_model,
        text_model,
        vectorizer,
        metadata: dict,
    ) -> None:
        self.numeric_model = numeric_model
        self.text_model = text_model
        self.vectorizer = vectorizer
        self.metadata = metadata
        self.feature_columns: list[str] = list(metadata["feature_columns"])
        self.feature_medians: list[float] = list(metadata["feature_medians"])
        self.risk_labels: dict[int, str] = {
            int(k): v for k, v in metadata["risk_labels"].items()
        }

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, model_dir: pathlib.Path = MODEL_DIR) -> "HybridWellbeingModel":
        """Load the three joblib artefacts plus ``metadata.json``."""
        numeric_path = model_dir / "numeric_model.joblib"
        text_path = model_dir / "text_model.joblib"
        vec_path = model_dir / "vectorizer.joblib"
        meta_path = model_dir / "metadata.json"

        missing = [
            str(p)
            for p in (numeric_path, text_path, vec_path, meta_path)
            if not p.exists()
        ]
        if missing:
            raise ModelNotTrainedError(
                "Missing model artefacts: "
                + ", ".join(missing)
                + ". Run: python -m training.train_model"
            )

        return cls(
            numeric_model=joblib.load(numeric_path),
            text_model=joblib.load(text_path),
            vectorizer=joblib.load(vec_path),
            metadata=json.loads(meta_path.read_text(encoding="utf-8")),
        )

    # -- inference -------------------------------------------------------

    def _numeric_matrix(self, student: dict) -> np.ndarray:
        return np.array(
            [[float(student[c]) for c in self.feature_columns]], dtype=float
        )

    def _text_matrix(self, feedback: str):
        """Sparse TF-IDF row. Stays sparse all the way to the estimator."""
        return self.vectorizer.transform([normalize_text(feedback)])

    def _blend(self, numeric_probs: np.ndarray, text_probs: np.ndarray) -> dict:
        """Combine the two branches into ``{class_label: score}``.

        Goes through ``classes_`` so the mapping survives a training set
        that happens to be missing a class.
        """
        numeric_classes = self.numeric_model.classes_
        text_classes = self.text_model.classes_

        numeric_map = dict(zip(numeric_classes, numeric_probs))
        text_map = dict(zip(text_classes, text_probs))

        blended: dict[int, float] = {}
        for label in numeric_classes:
            score = WEIGHT_NUMERIC * float(numeric_map.get(label, 0.0))
            score += WEIGHT_TEXT * float(text_map.get(label, 0.0))
            blended[int(label)] = score

        return blended

    def predict(self, student: dict) -> dict:
        """Score one student record. ``student`` must already be validated."""
        numeric_row = self._numeric_matrix(student)
        text_row = self._text_matrix(student["feedback"])

        numeric_probs = self.numeric_model.predict_proba(numeric_row)[0]
        text_probs = self.text_model.predict_proba(text_row)[0]
        blended = self._blend(numeric_probs, text_probs)

        risk_index = max(blended, key=blended.get)
        confidence = blended[risk_index]

        return {
            "risk_level": self.risk_labels.get(risk_index, "UNKNOWN"),
            "risk_index": int(risk_index),
            # Named "confidence" for the model's own certainty, not the
            # student's. The UI is required to label it as model output.
            "confidence": round(float(confidence), 4),
            "class_scores": {
                self.risk_labels.get(k, str(k)): round(v, 4)
                for k, v in blended.items()
            },
            "branch_scores": {
                "numeric": {
                    self.risk_labels.get(int(c), str(c)): round(float(p), 4)
                    for c, p in zip(self.numeric_model.classes_, numeric_probs)
                },
                "text": {
                    self.risk_labels.get(int(c), str(c)): round(float(p), 4)
                    for c, p in zip(self.text_model.classes_, text_probs)
                },
            },
            "explanation": self._explain(student, risk_index, confidence),
        }

    def _explain(self, student: dict, risk_index: int, confidence: float) -> dict:
        """Per-prediction, model-relative influence of each numeric input.

        Method: hold everything else fixed, swap one feature to the median
        observed during training, and record how far the score for the
        returned class moves. Positive means the submitted value pushed the
        score *up* relative to a typical value.

        This is a description of model behaviour, not a clinical finding,
        and it deliberately reports a direction and rough magnitude rather
        than a precise percentage.
        """
        baseline_row = np.array(
            [float(student[c]) for c in self.feature_columns], dtype=float
        )
        influences: list[dict] = []

        for idx, name in enumerate(self.feature_columns):
            perturbed = baseline_row.copy()
            perturbed[idx] = self.feature_medians[idx]

            probs = self.numeric_model.predict_proba(perturbed.reshape(1, -1))[0]
            scores = self._blend(
                probs, self.text_model.predict_proba(self._text_matrix(student["feedback"]))[0]
            )
            delta = confidence - scores.get(risk_index, 0.0)

            influences.append({
                "feature": name,
                "label": name.replace("_", " ").capitalize(),
                "submitted_value": round(float(baseline_row[idx]), 2),
                "reference_value": round(float(self.feature_medians[idx]), 2),
                "influence": round(float(delta), 4),
                "direction": "increased" if delta > 0 else "decreased",
            })

        influences.sort(key=lambda d: abs(d["influence"]), reverse=True)

        return {
            "type": "per_prediction_perturbation",
            "method": (
                "Each value was swapped for the training median and the "
                "change in this result's score was measured."
            ),
            "features": influences,
            "caveat": (
                "These are model-derived signals from this demonstration. "
                "They describe how this model reacted to your input. They "
                "are not clinical evidence and say nothing about your "
                "actual wellbeing."
            ),
        }

    # -- static, non-diagnostic guidance ---------------------------------

    @staticmethod
    def recommendations(risk_level: str, student: dict) -> list[str]:
        """General wellbeing suggestions. Never diagnostic, never treatment."""
        suggestions = [
            "Consider keeping a regular sleep schedule and aiming for a "
            "consistent bedtime.",
            "Make time for breaks and for activities you enjoy.",
            "If academic pressure is affecting you, consider speaking with "
            "a faculty member or your college counsellor.",
            "Regular movement and daylight tend to support mood and "
            "concentration.",
            "Connecting with friends or study groups can make demanding "
            "periods easier to manage.",
        ]

        if risk_level == "HIGH":
            suggestions.append(
                "If things feel heavy right now, talking to someone you "
                "trust is a reasonable next step."
            )

        return suggestions