"""Centralised request validation for the demo API.

Design goals
------------
* Every rejection a user can trigger is a ``400`` with a JSON body of the
  form ``{"error": "Invalid input", "details": [...]}`` -- never a ``500``
  and never a traceback.
* All problems in a payload are reported at once, so a student filling in
  a form is not whack-a-moled one field at a time.
* Validation depends only on :mod:`config`, not on Flask, so it is easy to
  test in isolation.

Deliberate strictness
---------------------
Numeric fields must arrive as real JSON numbers. A quoted ``"85"`` is
rejected rather than coerced: coercion hides client bugs, and the demo
frontend already parses the values properly. ``bool`` is rejected even
though it is a subclass of ``int`` in Python. ``NaN``/``Infinity`` are
rejected explicitly -- note that Python's ``json`` module happily parses
the bare literals ``NaN`` and ``Infinity``, so they must be filtered by
hand rather than assumed absent.
"""

from __future__ import annotations

import math
from typing import Any

from .config import (
    EMPTY_FEEDBACK_MESSAGE,
    NUMERIC_FIELDS,
    REQUIRED_NUMERIC_FIELDS,
    TEXT_FIELD,
    TEXT_MAX_CHARS,
    TEXT_MIN_CHARS,
)


class ValidationError(Exception):
    """Raised when a payload cannot be accepted. Carries per-field detail."""

    def __init__(self, details: list[str]) -> None:
        self.details = details
        super().__init__("; ".join(details))


def _validate_number(name: str, value: Any, low: float, high: float) -> float:
    """Return ``value`` as a float, or raise ``ValidationError``."""
    where = f"'{name}'"

    if isinstance(value, bool):
        raise ValidationError([f"{where} must be a number, not a boolean."])
    if not isinstance(value, (int, float)):
        raise ValidationError(
            [
                f"{where} must be a number "
                f"(got {type(value).__name__})."
            ]
        )

    number = float(value)
    if math.isnan(number):
        raise ValidationError([f"{where} must be a real number, not NaN."])
    if math.isinf(number):
        raise ValidationError([f"{where} must be a finite number, not infinity."])
    if not (low <= number <= high):
        raise ValidationError(
            [f"{where} must be between {low:g} and {high:g} (got {number:g})."]
        )
    return number


def _validate_feedback(value: Any) -> str:
    """Return the trimmed reflection text, or raise ``ValidationError``."""
    if not isinstance(value, str):
        raise ValidationError(
            [
                f"'{TEXT_FIELD}' must be text describing how you are "
                f"feeling (got {type(value).__name__})."
            ]
        )

    stripped = value.strip()
    if len(stripped) < TEXT_MIN_CHARS:
        raise ValidationError([EMPTY_FEEDBACK_MESSAGE])
    if len(stripped) > TEXT_MAX_CHARS:
        raise ValidationError(
            [f"'{TEXT_FIELD}' must be at most {TEXT_MAX_CHARS} characters."]
        )
    return stripped


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate a decoded request body and return a clean student record.

    Raises :class:`ValidationError` carrying *all* detected problems.
    """
    if payload is None:
        raise ValidationError(["A JSON request body is required."])
    if not isinstance(payload, dict):
        raise ValidationError(
            [f"Request body must be a JSON object, not {type(payload).__name__}."]
        )

    details: list[str] = []
    student: dict[str, Any] = {}

    for name in REQUIRED_NUMERIC_FIELDS:
        low, high, _unit, hint = NUMERIC_FIELDS[name]
        if name not in payload:
            details.append(f"'{name}' is required. {hint}")
            continue
        try:
            student[name] = _validate_number(name, payload[name], low, high)
        except ValidationError as exc:
            details.extend(exc.details)

    if TEXT_FIELD not in payload:
        details.append(
            f"'{TEXT_FIELD}' is required. Describe how you are feeling "
            "in your own words."
        )
    else:
        try:
            student[TEXT_FIELD] = _validate_feedback(payload[TEXT_FIELD])
        except ValidationError as exc:
            details.extend(exc.details)

    if details:
        raise ValidationError(details)

    return student