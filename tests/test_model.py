import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import (
    DEFAULT_MODEL_DIR,
    EXPECTED_FEATURES,
    METADATA_FILE,
    MODEL_FILE,
    LoadedModel,
    ModelValidationError,
    load_model,
    sha256_file,
)
from app.schemas import PredictionRequest


class FixedProbabilityPipeline:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, frame):
        return np.array([[1 - self.probability, self.probability]])


@pytest.fixture
def model_copy(tmp_path: Path) -> Path:
    for name in [MODEL_FILE, METADATA_FILE]:
        shutil.copy(DEFAULT_MODEL_DIR / name, tmp_path / name)
    return tmp_path


def rewrite_bundle(directory: Path, **changes) -> None:
    """Change the bundle and update its checksum so only the tested check can fail."""
    model_path = directory / MODEL_FILE
    metadata_path = directory / METADATA_FILE

    joblib.dump({**joblib.load(model_path), **changes}, model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_sha256"] = sha256_file(model_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_repository_model_loads():
    model = load_model()

    assert model.threshold == 0.17
    assert model.features == EXPECTED_FEATURES


def test_changed_model_file_is_rejected(model_copy):
    with (model_copy / MODEL_FILE).open("ab") as file:
        file.write(b"changed")

    with pytest.raises(ModelValidationError, match="checksum"):
        load_model(model_copy)


def test_missing_model_file_is_rejected(model_copy):
    (model_copy / MODEL_FILE).unlink()

    with pytest.raises(ModelValidationError, match="missing"):
        load_model(model_copy)


def test_changed_feature_list_is_rejected(model_copy):
    rewrite_bundle(model_copy, features=list(reversed(EXPECTED_FEATURES)))

    with pytest.raises(ModelValidationError, match="features"):
        load_model(model_copy)


def test_changed_threshold_is_rejected(model_copy):
    rewrite_bundle(model_copy, primary_threshold=0.5)

    with pytest.raises(ModelValidationError, match="threshold"):
        load_model(model_copy)


def test_service_does_not_start_with_invalid_model(model_copy, monkeypatch):
    with (model_copy / MODEL_FILE).open("ab") as file:
        file.write(b"changed")
    monkeypatch.setenv("MODEL_DIR", str(model_copy))

    with pytest.raises(ModelValidationError, match="checksum"):
        with TestClient(app):
            pass


@pytest.mark.parametrize(
    ("probability", "expected_review"),
    [(0.169, False), (0.17, True), (0.171, True)],
)
def test_threshold_boundary(valid_record, probability, expected_review):
    model = LoadedModel(
        pipeline=FixedProbabilityPipeline(probability),
        threshold=0.17,
        features=EXPECTED_FEATURES,
        version="test",
    )

    _, review = model.predict(PredictionRequest(**valid_record))

    assert review is expected_review


def test_fixture_belongs_to_the_repository_model(regression_fixture):
    assert regression_fixture["model_sha256"] == sha256_file(
        DEFAULT_MODEL_DIR / MODEL_FILE
    )


def test_predictions_match_stored_values(client, regression_fixture):
    records = regression_fixture["records"]
    expected = regression_fixture["expected_probabilities"]

    assert len(records) == len(expected) == 20
    for record, probability in zip(records, expected):
        body = client.post("/predict", json=record).json()
        assert body["default_probability"] == pytest.approx(probability, abs=1e-9)
