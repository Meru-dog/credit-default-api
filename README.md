# Credit Default Model API

A FastAPI service for a fixed credit default model and cost-based review threshold. The model comes from [credit-default-threshold](https://github.com/Meru-dog/credit-default-threshold).

**This is a learning demo. Do not submit real customer data, and do not use it for real credit decisions.**

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

Interactive API documentation is available at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and model version |
| GET | `/model-info` | Threshold, features, and limitations |
| POST | `/predict` | Default probability and review decision for one record |

## Safeguards

- The model checksum is verified before loading, and the service stops if it does not match.
- The threshold of 0.17 cannot be changed through a request.
- Demographic and other undefined fields are rejected.
- Input values are not logged, stored, or echoed in validation errors.

## Data and model attribution

The included model was trained on the following dataset, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset itself is not included in this repository.

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
