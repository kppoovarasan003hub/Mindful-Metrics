"""Unit tests for the validation layer (no Flask involved)."""

from __future__ import annotations

import math

import pytest

from mindful_metrics.validation import ValidationError, validate_payload


def base(**overrides) -> dict:
    payload = {
        "marks": 75,
        "attendance": 85,
        "sleep_hours": 7,
        "screen_time": 4,
        "assignment_delay": 1,
        "feedback": "Sleeping reasonably well and keeping on top of things.",
    }
    payload.update(overrides)
    return payload


# ----------------------------------------------------------------- happy path
def test_valid_payload_passes():
    student = validate_payload(base())
    assert student["marks"] == 75.0
    assert isinstance(student["marks"], float)
    assert student["feedback"].strip() == student["feedback"]


def test_feedback_is_trimmed():
    student = validate_payload(base(feedback="   feeling okay   "))
    assert student["feedback"] == "feeling okay"


def test_boundary_values_are_accepted():
    student = validate_payload(base(
        marks=0, attendance=100, sleep_hours=0,
        screen_time=24, assignment_delay=30,
    ))
    assert student["marks"] == 0.0
    assert student["screen_time"] == 24.0
    assert student["assignment_delay"] == 30.0


# ------------------------------------------------------------------- missing
@pytest.mark.parametrize("field", [
    "marks", "attendance", "sleep_hours",
    "screen_time", "assignment_delay", "feedback",
])
def test_missing_field_is_rejected(field):
    payload = base()
    payload.pop(field)
    with pytest.raises(ValidationError) as exc:
        validate_payload(payload)
    assert field in " ".join(exc.value.details)


def test_empty_body_is_rejected():
    with pytest.raises(ValidationError):
        validate_payload({})


def test_none_body_is_rejected():
    with pytest.raises(ValidationError):
        validate_payload(None)


def test_non_object_body_is_rejected():
    with pytest.raises(ValidationError):
        validate_payload(["marks", 90])


# --------------------------------------------------------------------- null
@pytest.mark.parametrize("field", [
    "marks", "attendance", "sleep_hours",
    "screen_time", "assignment_delay", "feedback",
])
def test_null_field_is_rejected(field):
    with pytest.raises(ValidationError):
        validate_payload(base(**{field: None}))


# ---------------------------------------------------------------- non-numeric
@pytest.mark.parametrize("bad", ["abc", "", [1, 2], {"a": 1}, object()])
def test_non_numeric_marks_rejected(bad):
    with pytest.raises(ValidationError):
        validate_payload(base(marks=bad))


def test_bool_is_rejected_even_though_it_is_an_int():
    # bool subclasses int in Python; True must not become 1.0.
    with pytest.raises(ValidationError) as exc:
        validate_payload(base(marks=True))
    assert "boolean" in " ".join(exc.value.details)


# -------------------------------------------------------------- NaN / Infinity
def test_nan_is_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_payload(base(marks=float("nan")))
    assert "nan" in " ".join(exc.value.details).lower()


def test_infinity_is_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_payload(base(sleep_hours=float("inf")))
    assert "infinity" in " ".join(exc.value.details).lower()


def test_negative_infinity_is_rejected():
    with pytest.raises(ValidationError):
        validate_payload(base(attendance=float("-inf")))


# ------------------------------------------------------------------ out of range
@pytest.mark.parametrize("payload", [
    {"marks": 1000000},
    {"marks": -1},
    {"marks": 101},
    {"attendance": -500},
    {"sleep_hours": 999},
    {"screen_time": 25},
    {"assignment_delay": 31},
    {"assignment_delay": -3},
])
def test_out_of_range_is_rejected(payload):
    with pytest.raises(ValidationError):
        validate_payload(base(**payload))


def test_range_message_states_the_bounds():
    with pytest.raises(ValidationError) as exc:
        validate_payload(base(marks=1000000))
    assert "0" in exc.value.details[0] and "100" in exc.value.details[0]


# -------------------------------------------------------------------- feedback
@pytest.mark.parametrize("bad", [123, 45.6, [], {}, True, None])
def test_non_string_feedback_rejected(bad):
    with pytest.raises(ValidationError):
        validate_payload(base(feedback=bad))


@pytest.mark.parametrize("blank", ["", "   ", "\n\t  \n"])
def test_empty_feedback_rejected_with_friendly_message(blank):
    with pytest.raises(ValidationError) as exc:
        validate_payload(base(feedback=blank))
    assert exc.value.details == [
        "Please provide some feedback before submitting."
    ]


def test_overlong_feedback_rejected():
    with pytest.raises(ValidationError):
        validate_payload(base(feedback="x" * 2001))


def test_feedback_at_max_length_accepted():
    validate_payload(base(feedback="x" * 2000))


# ------------------------------------------------------------------ reporting
def test_all_problems_reported_at_once():
    payload = base(marks="abc", sleep_hours=999, feedback="")
    with pytest.raises(ValidationError) as exc:
        validate_payload(payload)
    # marks, sleep_hours and feedback should all be reported together.
    assert len(exc.value.details) >= 3


def test_details_is_always_a_list_of_strings():
    with pytest.raises(ValidationError) as exc:
        validate_payload(base(marks=-5))
    assert isinstance(exc.value.details, list)
    assert all(isinstance(d, str) for d in exc.value.details)