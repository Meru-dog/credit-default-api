import logging

import pytest

from app.model import EXPECTED_FEATURES, EXPECTED_THRESHOLD


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    body = client.get("/model-info").json()

    assert body["threshold"] == EXPECTED_THRESHOLD
    assert body["features"] == EXPECTED_FEATURES
    assert any("real customer data" in item for item in body["limitations"])


def test_predict_returns_the_contract(client, valid_record):
    response = client.post("/predict", json=valid_record)
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {
        "default_probability",
        "review",
        "threshold",
        "model_version",
    }
    assert 0 <= body["default_probability"] <= 1
    assert body["threshold"] == EXPECTED_THRESHOLD
    assert body["review"] == (body["default_probability"] >= EXPECTED_THRESHOLD)


@pytest.mark.parametrize("field", ["SEX", "MARRIAGE", "AGE", "EDUCATION", "threshold"])
def test_undefined_fields_are_rejected(client, valid_record, field):
    response = client.post("/predict", json={**valid_record, field: 1})

    assert response.status_code == 422


@pytest.mark.parametrize("field", EXPECTED_FEATURES)
def test_missing_fields_are_rejected(client, valid_record, field):
    del valid_record[field]

    assert client.post("/predict", json=valid_record).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LIMIT_BAL", 0),
        ("LIMIT_BAL", -1),
        ("PAY_0", -3),
        ("PAY_0", 10),
        ("PAY_0", 1.5),
        ("PAY_0", "1"),
        ("PAY_0", True),
        ("PAY_AMT1", -1),
        ("BILL_AMT1", "100"),
        ("BILL_AMT1", None),
    ],
)
def test_invalid_values_are_rejected(client, valid_record, field, value):
    response = client.post("/predict", json={**valid_record, field: value})

    assert response.status_code == 422


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_are_rejected(client, valid_record, value):
    valid_record["BILL_AMT1"] = "__VALUE__"
    body = str(valid_record).replace("'", '"').replace('"__VALUE__"', value)

    response = client.post(
        "/predict",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


def test_negative_bill_amount_is_allowed(client, valid_record):
    response = client.post("/predict", json={**valid_record, "BILL_AMT1": -100})

    assert response.status_code == 200


def test_validation_errors_do_not_echo_input(client, valid_record):
    response = client.post("/predict", json={**valid_record, "PAY_AMT1": -987654})

    assert response.status_code == 422
    assert "987654" not in response.text


def test_logs_do_not_contain_input_values(client, valid_record, caplog):
    valid_record["LIMIT_BAL"] = 987654

    with caplog.at_level(logging.INFO):
        response = client.post("/predict", json=valid_record)

    assert response.status_code == 200
    assert "path=/predict" in caplog.text
    assert "987654" not in caplog.text
