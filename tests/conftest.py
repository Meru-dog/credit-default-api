import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prediction_regression.json"


@pytest.fixture(scope="session")
def regression_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def valid_record(regression_fixture: dict) -> dict:
    return dict(regression_fixture["records"][0])


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client
