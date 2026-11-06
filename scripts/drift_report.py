"""Demonstrate drift detection with held-out and deliberately shifted data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from app.model import DEFAULT_MODEL_DIR, METADATA_FILE, load_model

RANDOM_STATE = 42
TEST_SIZE = 0.2
DRIFT_SHARE = 0.5
PR_AUC_TOLERANCE = 1e-6
PREDICTION = "predicted_probability"
OUTPUT_DIR = Path("reports")
SOURCE_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features.copy()
    if features.shape[1] != len(SOURCE_COLUMNS):
        raise ValueError("Unexpected number of source columns")
    features.columns = SOURCE_COLUMNS
    return features, dataset.data.targets.iloc[:, 0].astype(int)


def shift_features(features: pd.DataFrame) -> pd.DataFrame:
    """Apply a deliberate change. This is not an estimate of real drift."""
    shifted = features.copy()
    shifted["LIMIT_BAL"] = shifted["LIMIT_BAL"] * 0.8
    for column in [column for column in shifted if column.startswith("PAY_") and "AMT" not in column]:
        shifted[column] = (shifted[column] + 1).clip(upper=9)
    for column in [column for column in shifted if column.startswith("BILL_AMT")]:
        shifted[column] = shifted[column] * 1.3
    for column in [column for column in shifted if column.startswith("PAY_AMT")]:
        shifted[column] = shifted[column] * 0.5
    return shifted


def with_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    scored = features.copy()
    scored[PREDICTION] = model.pipeline.predict_proba(features[model.features])[:, 1]
    return scored


def run_report(reference: pd.DataFrame, current: pd.DataFrame, name: str) -> dict:
    definition = DataDefinition(numerical_columns=list(reference.columns))
    result = Report([DataDriftPreset(drift_share=DRIFT_SHARE)]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    result.save_html(str(OUTPUT_DIR / f"drift_{name}.html"))

    metrics = result.dict()["metrics"]
    overall = next(
        metric["value"]
        for metric in metrics
        if metric["metric_name"].startswith("DriftedColumnsCount")
    )
    columns = {
        metric["config"]["column"]: {
            "method": metric["config"]["method"],
            "score": float(metric["value"]),
            "threshold": metric["config"]["threshold"],
        }
        for metric in metrics
        if metric["metric_name"].startswith("ValueDrift")
    }
    return {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drifted_columns": int(overall["count"]),
        "total_columns": len(columns),
        "drifted_share": float(overall["share"]),
        "dataset_drift": bool(overall["share"] >= DRIFT_SHARE),
        "prediction_drift": columns[PREDICTION],
        "columns": columns,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = load_model()
    features, target = load_data()
    selected = features[model.features]

    _, x_test, _, y_test = train_test_split(
        selected,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    probabilities = model.pipeline.predict_proba(x_test)[:, 1]
    pr_auc = average_precision_score(y_test, probabilities)
    metadata = json.loads((DEFAULT_MODEL_DIR / METADATA_FILE).read_text(encoding="utf-8"))
    if abs(pr_auc - metadata["test_pr_auc"]) > PR_AUC_TOLERANCE:
        raise ValueError("The test split or model does not match the recorded results")

    order = np.random.default_rng(RANDOM_STATE).permutation(len(x_test))
    half = len(x_test) // 2
    control_reference = x_test.iloc[order[:half]]
    control_current = x_test.iloc[order[half:]]

    summary = {
        "test_pr_auc": pr_auc,
        "control": run_report(
            with_predictions(model, control_reference),
            with_predictions(model, control_current),
            "control",
        ),
        "shifted": run_report(
            with_predictions(model, x_test),
            with_predictions(model, shift_features(x_test)),
            "shifted",
        ),
    }
    (OUTPUT_DIR / "drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for name in ["control", "shifted"]:
        result = summary[name]
        print(
            f"{name}: {result['drifted_columns']}/{result['total_columns']} columns drifted, "
            f"dataset_drift={result['dataset_drift']}, "
            f"prediction_score={result['prediction_drift']['score']:.3f}"
        )

    passed = not summary["control"]["dataset_drift"] and summary["shifted"]["dataset_drift"]
    print("Drift check passed" if passed else "Drift check failed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
