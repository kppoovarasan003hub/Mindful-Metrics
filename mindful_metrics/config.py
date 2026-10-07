"""Validation constants and input contract for the demo API.

Everything the backend needs to decide whether a request is acceptable
lives here, so there is exactly one place to look when asking "what does
this endpoint accept?".

Ranges below were chosen from the domains the original demo already
implied, not copied blindly:

* ``marks`` / ``attendance``  -- percentages, 0-100 (the original UI
  already used ``min=0 max=100`` for both).
* ``sleep_hours`` / ``screen_time`` -- hours in a day, 0-24 (the original
  UI used ``min=0 max=24``). Realistic population values are narrower, but
  the API validates the *plausible* domain rather than rejecting unusual
  inputs outright; a 25h day is a data-entry slip, not a safety issue.
* ``assignment_delay`` -- calendar days an assignment has been overdue,
  0-30. The original UI constrained only the lower bound and left the
  upper bound unstated, so 30 days is a documented choice: it covers a
  whole teaching month and rejects nonsense like 10**9.
* ``feedback`` -- 1-2000 characters. Empty or whitespace-only is a 400.
"""

from __future__ import annotations

# Field name -> (minimum, maximum, unit, human-readable hint)
NUMERIC_FIELDS: dict[str, tuple[float, float, str, str]] = {
    "marks": (0.0, 100.0, "%", "Marks as a percentage, 0-100."),
    "attendance": (0.0, 100.0, "%", "Attendance as a percentage, 0-100."),
    "sleep_hours": (0.0, 24.0, "hours", "Hours of sleep in a day, 0-24."),
    "screen_time": (0.0, 24.0, "hours", "Hours of screen time in a day, 0-24."),
    "assignment_delay": (
        0.0,
        30.0,
        "days",
        "Days an assignment is overdue, 0-30.",
    ),
}

REQUIRED_NUMERIC_FIELDS = tuple(NUMERIC_FIELDS)

TEXT_FIELD = "feedback"
TEXT_MIN_CHARS = 1
TEXT_MAX_CHARS = 2000

# Guard against a very large JSON body before it is parsed into memory.
MAX_CONTENT_LENGTH = 64 * 1024

# Sent verbatim to the client; referenced by the UI and the model card.
EMPTY_FEEDBACK_MESSAGE = "Please provide some feedback before submitting."