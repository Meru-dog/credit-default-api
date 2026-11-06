"""Smoke test a running API using only the Python standard library."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = sys.argv[1].rstrip("/")
FIXTURE = Path("tests/fixtures/prediction_regression.json")


def request(path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"content-type": "application/json", "user-agent": "credit-default-smoke-test"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

status, health = request("/health")
assert status == 200 and health["status"] == "ok", health

status, info = request("/model-info")
assert status == 200 and info["threshold"] == 0.17, info

for record, expected in zip(fixture["records"], fixture["expected_probabilities"]):
    status, body = request("/predict", record)
    assert status == 200, body
    assert abs(body["default_probability"] - expected) < 1e-9, body

status, _ = request("/predict", {**fixture["records"][0], "SEX": 1})
assert status == 422, status

print(f"Smoke test passed: {health['model_version']}")
