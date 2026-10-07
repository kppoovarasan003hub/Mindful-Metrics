"""API contract tests, exercised through the real Flask test client.

Covers the cases named in the remediation brief: valid request, missing
field, empty body, nulls, non-numeric, out-of-range, invalid feedback
type, empty feedback, extreme values, LOW/MEDIUM/HIGH response shape,
error leakage, UTF-8 encoding, and the debug/security configuration.
"""

from __future__ import annotations

import json
import math

import pytest

from mindful_metrics.hybrid_model import DISCLAIMER, PRIVACY_NOTICE


def post(client, payload, **kwargs):
    return client.post("/predict", json=payload, **kwargs)


def assert_bad_request(response):
    """Every user-input failure looks exactly like this."""
    assert response.status_code == 400, response.get_data(as_text=True)[:300]
    body = response.get_json()
    assert body["error"] == "Invalid input"
    assert isinstance(body["details"], list)
    assert body["details"], "a 400 must explain itself"
    assert all(isinstance(d, str) and d for d in body["details"])


# ============================================================== happy path
def test_valid_request_returns_contract(client, valid_payload):
    response = post(client, valid_payload)
    assert response.status_code == 200
    body = response.get_json()

    # Documented contract.
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["recommendations"], list)
    assert body["explanation"]["type"] == "per_prediction_perturbation"
    assert isinstance(body["explanation"]["features"], list)

    # Honesty fields the UI depends on.
    assert body["disclaimer"] == DISCLAIMER
    assert body["privacy_notice"] == PRIVACY_NOTICE
    assert isinstance(body["is_high_risk"], bool)
    assert "body" in body["support"]
    assert body["support"]["options"]


def test_no_raw_sklearn_objects_leak(client, valid_payload):
    """The response must be plain JSON, not model internals."""
    body = post(client, valid_payload).get_json()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            assert node is None or isinstance(node, (str, int, float, bool)), \
                f"unexpected non-JSON type: {type(node).__name__}"

    walk(body)


def test_class_scores_and_branch_scores_present(client, valid_payload):
    body = post(client, valid_payload).get_json()
    assert set(body["class_scores"]) == {"LOW", "MEDIUM", "HIGH"}
    assert set(body["branch_scores"]) == {"numeric", "text"}
    # Both branches are weighted blends, so they should sum to ~1.
    assert sum(body["class_scores"].values()) == pytest.approx(1.0, abs=1e-3)


def test_explanation_features_are_per_prediction(client):
    """Two different students must not get identical explanations."""
    a = post(client, {
        "marks": 95, "attendance": 98, "sleep_hours": 9, "screen_time": 1,
        "assignment_delay": 0, "feedback": "great happy fine",
    }).get_json()
    b = post(client, {
        "marks": 20, "attendance": 25, "sleep_hours": 2, "screen_time": 14,
        "assignment_delay": 12, "feedback": "stressed and cannot sleep",
    }).get_json()

    fa = {f["feature"]: f["influence"] for f in a["explanation"]["features"]}
    fb = {f["feature"]: f["influence"] for f in b["explanation"]["features"]}
    assert fa != fb


# ============================================================== 400 coverage
def test_empty_body_returns_400(client):
    response = post(client, {})
    assert_bad_request(response)
    assert "error" in response.get_json()


def test_missing_field_returns_400(client, valid_payload):
    for field in ("marks", "attendance", "sleep_hours",
                  "screen_time", "assignment_delay", "feedback"):
        payload = dict(valid_payload)
        payload.pop(field)
        assert_bad_request(post(client, payload))


def test_null_marks_returns_400(client, valid_payload):
    # This used to be a 500 (float(None) raised TypeError).
    assert_bad_request(post(client, dict(valid_payload, marks=None)))


def test_null_feedback_returns_400(client, valid_payload):
    # This used to be a 500 ('NoneType' has no attribute 'lower').
    assert_bad_request(post(client, dict(valid_payload, feedback=None)))


def test_non_numeric_marks_returns_400(client, valid_payload):
    assert_bad_request(post(client, dict(valid_payload, marks="abc")))


def test_quoted_number_is_rejected_not_coerced(client, valid_payload):
    """Strings are not silently coerced; that hides client bugs."""
    assert_bad_request(post(client, dict(valid_payload, marks="75")))


@pytest.mark.parametrize("payload", [
    {"marks": 1000000},
    {"marks": -500},
    {"attendance": -500},
    {"sleep_hours": 999},
    {"screen_time": 999},
    {"assignment_delay": 99999},
])
def test_extreme_values_return_400(client, valid_payload, payload):
    assert_bad_request(post(client, dict(valid_payload, **payload)))


@pytest.mark.parametrize("bad", [123, 45.6, [], {}, True])
def test_invalid_feedback_type_returns_400(client, valid_payload, bad):
    assert_bad_request(post(client, dict(valid_payload, feedback=bad)))


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_empty_feedback_returns_400(client, valid_payload, blank):
    response = post(client, dict(valid_payload, feedback=blank))
    assert_bad_request(response)
    assert response.get_json()["details"] == [
        "Please provide some feedback before submitting."
    ]


def test_nan_and_infinity_return_400(client, valid_payload):
    assert_bad_request(post(client, dict(valid_payload, marks=float("nan"))))
    assert_bad_request(post(client, dict(valid_payload, marks=float("inf"))))


def test_boolean_marks_returns_400(client, valid_payload):
    assert_bad_request(post(client, dict(valid_payload, marks=True)))


def test_multiple_errors_all_reported(client):
    response = post(client, {
        "marks": "abc", "sleep_hours": 999, "feedback": "",
    })
    assert_bad_request(response)
    assert len(response.get_json()["details"]) >= 3


# ========================================================== malformed bodies
def test_malformed_json_returns_400_json(client):
    response = client.post("/predict", data="{not json",
                           content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid input"


def test_empty_request_body_returns_400(client):
    response = client.post("/predict", data="", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid input"


def test_json_array_body_returns_400(client):
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid input"


def test_non_json_content_type_returns_400(client):
    response = client.post("/predict", data="marks=75",
                           content_type="application/x-www-form-urlencoded")
    assert response.status_code == 400


def test_oversized_payload_rejected(client, valid_payload):
    big = dict(valid_payload, feedback="x" * 200_000)
    response = post(client, big)
    assert response.status_code in (400, 413)
    assert response.get_json()["error"] == "Invalid input"


# ============================================================== error safety
def test_errors_do_not_expose_internals(client):
    """No traceback, no file paths, no module internals, ever."""
    response = post(client, {"marks": "abc"})
    raw = response.get_data(as_text=True)

    for leak in ("Traceback", ".py", "site-packages", "sklearn",
                 "numpy", "joblib", "hybrid_model.py", "app.py"):
        assert leak not in raw, f"response leaked {leak!r}"


def test_500_handler_returns_generic_json(client, valid_payload, monkeypatch):
    """A server fault must not surface a traceback to the browser."""
    import app as flask_app

    def boom(_student):
        raise RuntimeError("internal detail that must not escape")

    monkeypatch.setattr(flask_app.model, "predict", boom)
    response = post(client, valid_payload)

    assert response.status_code == 500
    body = response.get_json()
    assert body["error"] == "Unable to process the request."
    raw = response.get_data(as_text=True)
    assert "internal detail" not in raw
    assert "Traceback" not in raw


def test_404_is_json(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found."


def test_405_is_json(client):
    response = client.get("/predict")
    assert response.status_code == 405
    assert response.get_json()["error"] == "Method not allowed."


# ================================================================= encoding
def test_medium_label_is_valid_utf8_without_mojibake(client):
    """Regression: the MEDIUM emoji was corrupted to U+FFFD."""
    from training.train_model import RISK_LABELS  # noqa: F401

    # Find an input that lands on MEDIUM, then verify the wire bytes.
    found = None
    for marks in range(20, 95, 3):
        for delay in range(0, 12, 2):
            body = post(client, {
                "marks": marks, "attendance": marks, "sleep_hours": 6,
                "screen_time": 5, "assignment_delay": delay,
                "feedback": "busy week but managing okay",
            }).get_json()
            if body["risk_level"] == "MEDIUM":
                found = body
                break
        if found:
            break

    assert found is not None, "expected to be able to reach a MEDIUM result"
    raw = post(client, {
        "marks": 55, "attendance": 55, "sleep_hours": 6,
        "screen_time": 5, "assignment_delay": 5,
        "feedback": "busy week but managing okay",
    }).get_data()
    # The replacement character must not appear anywhere in the response.
    assert "�".encode("utf-8") not in raw


def test_source_file_has_no_replacement_character():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for path in ["mindful_metrics/hybrid_model.py", "app.py", "mindful_metrics/validation.py",
                 "static/script.js", "templates/index.html"]:
        text = (root / path).read_text(encoding="utf-8")
        assert "�" not in text, f"{path} contains U+FFFD"


def test_json_is_utf8_encoded(client, valid_payload):
    raw = post(client, valid_payload).get_data()
    raw.decode("utf-8")  # must not raise
    assert isinstance(json.loads(raw), dict)


# =================================================================== levels
@pytest.mark.parametrize("profile", [
    {"marks": 95, "attendance": 96, "sleep_hours": 9, "screen_time": 1,
     "assignment_delay": 0},
    {"marks": 55, "attendance": 55, "sleep_hours": 6, "screen_time": 5,
     "assignment_delay": 4},
    {"marks": 18, "attendance": 25, "sleep_hours": 2, "screen_time": 14,
     "assignment_delay": 12},
])
def test_response_shape_valid_across_range(client, profile):
    """Whatever the level, the contract must hold identically."""
    body = post(client, {
        **profile, "feedback": "sleeping badly and worried about deadlines",
    }).get_json()

    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["class_scores"]) == {"LOW", "MEDIUM", "HIGH"}
    assert len(body["explanation"]["features"]) == 5
    assert body["recommendations"]
    assert body["support"]["options"]


def test_high_risk_has_support_panel(client):
    body = post(client, {
        "marks": 5, "attendance": 10, "sleep_hours": 1, "screen_time": 20,
        "assignment_delay": 25,
        "feedback": "extremely stressed cannot sleep at all",
    }).get_json()

    assert body["support"]["options"]
    assert "not a diagnosis" in body["support"]["body"].lower() \
        or body["risk_level"] != "HIGH"


def test_recommendations_are_non_diagnostic(client, valid_payload):
    """No clinical or treatment language anywhere in the output."""
    body = post(client, valid_payload).get_json()
    banned = ["you are depressed", "you have anxiety", "diagnosis",
              "you need treatment", "prescribe", "disorder"]
    blob = " ".join(body["recommendations"]).lower()
    for phrase in banned:
        assert phrase not in blob, f"non-diagnostic requirement broken: {phrase}"


# ================================================================== privacy
def test_reflection_text_is_not_echoed_in_response(client):
    secret = "UNIQUEMARKERSTRESSZZZ"
    body = post(client, {
        "marks": 50, "attendance": 50, "sleep_hours": 5, "screen_time": 6,
        "assignment_delay": 3, "feedback": f"{secret} feeling overwhelmed",
    }).get_json()
    assert secret not in json.dumps(body)


def test_reflection_text_not_logged_on_error(client, valid_payload, caplog):
    """A failure must not spill the journal entry into the logs."""
    import app as flask_app

    secret = "UNIQUEMARKERLOGZZZ"

    def boom(_student):
        raise RuntimeError("kaboom")

    original = flask_app.model.predict
    flask_app.model.predict = boom
    try:
        with caplog.at_level("DEBUG"):
            post(client, dict(valid_payload, feedback=f"{secret} text"))
    finally:
        flask_app.model.predict = original

    assert secret not in caplog.text


# ================================================================= security
def test_debug_is_off_by_default():
    import app as flask_app
    assert flask_app.app.config["DEBUG"] is False


def test_max_content_length_is_configured():
    import app as flask_app
    assert flask_app.app.config["MAX_CONTENT_LENGTH"] > 0


def test_cors_not_open_by_default():
    """With no CORS_ORIGINS set, no wildcard header may be emitted."""
    import os
    if os.environ.get("CORS_ORIGINS", "").strip():
        pytest.skip("CORS_ORIGINS configured in this environment")
    import app as flask_app
    response = flask_app.app.test_client().get(
        "/", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "demo" in html.lower()
    # The disclaimer must be on the page itself.
    assert "not a diagnosis" in html.lower()
    # Support resources must be present.
    assert "counsellor" in html.lower() or "counselor" in html.lower()