"""Flask app for the student wellbeing demo.

Serving only. This module loads pre-trained artefacts (see
``training/train_model.py``) and exposes a small, strict JSON API.

Security posture
----------------
* ``DEBUG`` defaults to **off** and is only enabled when ``FLASK_DEBUG=1``
  is set explicitly. The old ``app.run(debug=True)`` is gone.
* CORS is **off** unless ``CORS_ORIGINS`` names explicit origins. The demo
  serves its own frontend from the same origin, so no cross-origin access
  is needed.
* Request bodies are capped at 64 KB.
* Client errors return JSON with field-level detail. Server errors return a
  fixed JSON message and are logged by *type and route only*.

Privacy posture
---------------
Reflection text is read from the request, used to compute a result, and
discarded. It is never logged, never persisted, never included in an error
message. The UI states this, and it is true of this code.

Note on Flask logs: Flask's default handler logs the request line and
status. It does not log request bodies, so submitted reflection text does
not appear there. Application-level logging in this file records only
error class names and routes.
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from mindful_metrics.config import MAX_CONTENT_LENGTH
from mindful_metrics.hybrid_model import (
    DISCLAIMER,
    HIGH_RISK_MESSAGE,
    PRIVACY_NOTICE,
    SUPPORT_LINES,
    HybridWellbeingModel,
    ModelNotTrainedError,
)
from mindful_metrics.validation import ValidationError, validate_payload

# --------------------------------------------------------------------------
# Configuration, entirely environment-driven.
# --------------------------------------------------------------------------
DEBUG = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
PORT = int(os.environ.get("PORT", "5000"))
HOST = os.environ.get("HOST", "127.0.0.1")

# Comma-separated allowlist. Empty string => no cross-origin access at all,
# which is the correct default for a same-origin demo.
_RAW_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _RAW_ORIGINS.split(",") if o.strip()]

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.update(
    DEBUG=DEBUG,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    JSON_SORT_KEYS=False,
    # Belt and braces: never echo the request body back in an error page.
    PROPAGATE_EXCEPTIONS=False,
)

if ALLOWED_ORIGINS:
    from flask_cors import CORS

    CORS(app, resources={r"/predict": {"origins": ALLOWED_ORIGINS}})
    app.logger.info("CORS enabled for %d configured origin(s).",
                    len(ALLOWED_ORIGINS))

# --------------------------------------------------------------------------
# Model loading. Fails loudly at start-up rather than per request.
# --------------------------------------------------------------------------
logger = logging.getLogger("wellbeing_demo")
try:
    model = HybridWellbeingModel.load()
except ModelNotTrainedError as exc:
    # Logged and re-raised: running without a model would mean silently
    # returning 500s to every visitor.
    logger.error("Model artefacts missing: %s", exc)
    raise SystemExit(
        f"{exc}\n\nRun `python -m training.train_model` first."
    ) from exc


# --------------------------------------------------------------------------
# Error handling. Every response from the API is JSON.
# --------------------------------------------------------------------------
@app.errorhandler(ValidationError)
def _handle_validation_error(exc: ValidationError):
    return jsonify({"error": "Invalid input", "details": exc.details}), 400


@app.errorhandler(RequestEntityTooLarge)
def _handle_too_large(_exc):
    return jsonify({
        "error": "Invalid input",
        "details": ["Request body is too large (limit 64 KB)."],
    }), 413


@app.errorhandler(400)
def _handle_bad_request(_exc):
    # Reached for malformed JSON, which Flask rejects before our handler.
    return jsonify({
        "error": "Invalid input",
        "details": ["Request body must be valid JSON."],
    }), 400


@app.errorhandler(404)
def _handle_not_found(_exc):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def _handle_method_not_allowed(_exc):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(HTTPException)
def _handle_http_exception(exc: HTTPException):
    return jsonify({
        "error": "Unable to process the request.",
        "details": [],
    }), exc.code or 500


@app.errorhandler(Exception)
def _handle_unexpected(exc: Exception):
    # Deliberately logs the exception *type* and the route, never the message
    # and never the payload: an exception raised while handling reflection
    # text can quote that text.
    logger.error(
        "Unhandled %s while serving %s", type(exc).__name__, request.path
    )
    return jsonify({
        "error": "Unable to process the request.",
        "details": [],
    }), 500


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    """Serve the demo page with server-rendered policy text."""
    return render_template(
        "index.html",
        disclaimer=DISCLAIMER,
        privacy_notice=PRIVACY_NOTICE,
        support_lines=SUPPORT_LINES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # request.get_json() returns None on a malformed body rather than
        # raising, depending on version; either way it funnels into the
        # validator as a 400.
        payload = request.get_json(silent=True)
        student = validate_payload(payload)
    except ValidationError as exc:
        return jsonify({"error": "Invalid input", "details": exc.details}), 400

    try:
        result = model.predict(student)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Inference failed with %s on /predict", type(exc).__name__)
        return jsonify({
            "error": "Unable to process the request.",
            "details": [],
        }), 500

    risk_level = result["risk_level"]
    response = {
        "risk_level": risk_level,
        "confidence": result["confidence"],
        "confidence_note": (
            "This is how strongly the demo model leaned toward this label. "
            "It is not a certainty and not a clinical probability."
        ),
        "class_scores": result["class_scores"],
        "branch_scores": result["branch_scores"],
        "explanation": result["explanation"],
        "recommendations": model.recommendations(risk_level, student),
        "disclaimer": DISCLAIMER,
        "is_high_risk": risk_level == "HIGH",
        "support": {
            "heading": (
                HIGH_RISK_MESSAGE if risk_level == "HIGH"
                else "If you need support"
            ),
            "body": (
                "This result is only an experimental model output. It is "
                "not a diagnosis and cannot determine your mental-health "
                "status."
                if risk_level == "HIGH" else
                "This demonstration cannot assess your wellbeing. If "
                "something feels difficult, support is available."
            ),
            "options": list(SUPPORT_LINES),
        },
        "privacy_notice": PRIVACY_NOTICE,
    }

    return jsonify(response), 200


if __name__ == "__main__":
    # No debug=True. Bind to loopback unless the operator says otherwise.
    app.run(host=HOST, port=PORT, debug=DEBUG)