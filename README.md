# Credit Default Model API

A tested, containerized FastAPI service for a fixed credit default model and cost-based review threshold. The model comes from [credit-default-threshold](https://github.com/Meru-dog/credit-default-threshold).

**This is a learning demo. Do not submit real customer data, and do not use it for real credit decisions.**

- **Live API documentation:** https://credit-default-api-skqf.onrender.com/docs
- **Container image:** `ghcr.io/meru-dog/credit-default-api`
- **Detailed results:** [results.md](results.md)

The demo runs on Render's Free plan. It sleeps after 15 minutes without traffic, so the first request can take about a minute.

## What this project demonstrates

| Area | Implementation |
|---|---|
| Serving | FastAPI service with validated requests and a fixed threshold |
| Reproducibility | Pinned dependencies and checksum-verified model loading |
| Testing | 54 automated tests, including prediction regression tests |
| Delivery | GitHub Actions tests the code, smoke tests the image, and publishes it to GHCR |
| Deployment | Published image deployed to Render by digest |
| Monitoring demo | Evidently control and shifted-data drift reports |
| Privacy | Input values are not logged, stored, or echoed in validation errors |

```text
tests -> image build -> container smoke test -> publish to GHCR -> deploy by digest
```

## Example request

```bash
curl -X POST https://credit-default-api-skqf.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 1000, "BILL_AMT2": 1000, "BILL_AMT3": 1000,
    "BILL_AMT4": 1000, "BILL_AMT5": 1000, "BILL_AMT6": 1000,
    "PAY_AMT1": 500, "PAY_AMT2": 500, "PAY_AMT3": 500,
    "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500
  }'
```

The response contains `default_probability`, `review`, `threshold`, and `model_version`. The values above are synthetic.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and model version |
| GET | `/model-info` | Threshold, features, and limitations |
| POST | `/predict` | Default probability and review decision for one record |

## Run locally

Requires Python 3.13.

```bash
pip install uv
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
pytest
uvicorn app.main:app --port 8000 --no-access-log
```

Or run the published image:

```bash
docker run -p 7860:7860 ghcr.io/meru-dog/credit-default-api:latest
```

## Drift report

```bash
uv pip install -r requirements-drift.txt
python -m scripts.drift_report
```

The script retrieves the dataset at runtime and writes reports to `reports/`. It can also be run manually from the **Drift report** GitHub Actions workflow.

## Main limitations

- The 0.17 threshold relies on an assumed 5:1 cost ratio.
- The model was trained on 2005 Taiwanese credit card data.
- Excluding demographic fields does not establish fairness, and no fairness audit was performed.
- The service has no authentication or rate limiting.
- The shifted data demonstrates detection but does not represent real production drift.

## Data and model attribution

The included model was trained on the following dataset, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset itself is not included in this repository.

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
