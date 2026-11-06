"""Load and verify the fixed model bundle."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import PredictionRequest

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILE = "credit_default_model.joblib"
METADATA_FILE = "model_metadata.json"
EXPECTED_FEATURES = list(PredictionRequest.model_fields)
EXPECTED_THRESHOLD = 0.17


class ModelValidationError(RuntimeError):
    """Raised when the model must not be served."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: object
    threshold: float
    features: list[str]
    version: str

    def predict(self, request: PredictionRequest) -> tuple[float, bool]:
        row = pd.DataFrame([request.model_dump()])[self.features]
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return probability, probability >= self.threshold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(model_dir: Path | None = None) -> LoadedModel:
    directory = Path(model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    model_path = directory / MODEL_FILE
    metadata_path = directory / METADATA_FILE

    if not model_path.is_file() or not metadata_path.is_file():
        raise ModelValidationError(f"Model files are missing in {directory}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Check the file before joblib.load because loading can execute code.
    if sha256_file(model_path) != metadata.get("model_sha256"):
        raise ModelValidationError("Model checksum does not match the metadata")

    bundle = joblib.load(model_path)

    if list(bundle["features"]) != EXPECTED_FEATURES:
        raise ModelValidationError("Model features do not match the API schema")
    if bundle["primary_threshold"] != EXPECTED_THRESHOLD:
        raise ModelValidationError("Model threshold does not match the fixed value")

    return LoadedModel(
        pipeline=bundle["pipeline"],
        threshold=float(bundle["primary_threshold"]),
        features=list(bundle["features"]),
        version=f"{metadata['artifact_name']}:{metadata['artifact_version']}",
    )
