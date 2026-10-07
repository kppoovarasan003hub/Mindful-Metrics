"""Shared fixtures for the test suite.

The model is loaded once per session rather than retrained, which is both
faster and a useful check that loading works outside of app start-up.
"""

from __future__ import annotations

import pathlib

import pytest

from mindful_metrics.hybrid_model import HybridWellbeingModel

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def model() -> HybridWellbeingModel:
    return HybridWellbeingModel.load()


@pytest.fixture(scope="session")
def client(model):
    """Flask test client. Importing ``app`` loads the same artefacts."""
    import app as flask_app

    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


@pytest.fixture()
def valid_payload() -> dict:
    """A payload that should always be accepted."""
    return {
        "marks": 75,
        "attendance": 85,
        "sleep_hours": 7,
        "screen_time": 4,
        "assignment_delay": 1,
        "feedback": "Sleeping reasonably well and keeping on top of things.",
    }