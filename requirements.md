# Requirements: Credit Default Model API

## 1. Objective

Serve the fixed credit default model and decision threshold from the `credit-default-threshold` project through a tested, containerized HTTP API, and demonstrate a basic drift check.

The focus is operational engineering, not improving the model. This is a learning project and must not be used for real credit decisions.

## 2. Scope

### In scope

- A FastAPI service for the existing model and threshold
- Input validation and model integrity checks
- Automated tests and CI with GitHub Actions
- A Docker image published to GitHub Container Registry (GHCR)
- A public demo on Hugging Face Spaces
- An Evidently drift report using held-out and artificially shifted data
- English documentation in a new public repository

### Out of scope

- Model retraining, tuning, or threshold changes
- Automated retraining
- Authentication and rate limiting
- Kubernetes and autoscaling
- Collecting real customer data

## 3. Environment

| Item | Selection |
|---|---|
| Development | GitHub Codespaces |
| API | FastAPI, Pydantic, Uvicorn |
| Model runtime | scikit-learn, LightGBM, joblib |
| Tests | pytest, FastAPI TestClient |
| Container | Docker |
| CI/CD | GitHub Actions, GHCR |
| Deployment | Hugging Face Spaces (Docker) |
| Drift check | Evidently, `ucimlrepo` |

Current free-tier limits and platform terms must be checked against official documentation before deployment.

## 4. Model

| Item | Details |
|---|---|
| Source | W&B Artifact `credit-default-lgbm-small`, latest version |
| Contents | Fitted pipeline, thresholds, cost ratios, and feature list |
| Primary threshold | 0.17 |
| Location | Committed to the repository |
| Integrity check | Verify SHA-256 at startup and stop if it does not match |
| Metadata | Record artifact version, test metrics, and runtime versions |

Runtime library versions must match the environment that created the model. The joblib file must come only from this project because loading it can execute code.

The model is trained on Default of Credit Card Clients (Yeh, 2009, https://doi.org/10.24432/C55S3H), licensed under CC BY 4.0. The repository must provide this attribution.

## 5. Functional requirements

| ID | Requirement |
|---|---|
| F-01 | `GET /health` returns service status after the model has loaded |
| F-02 | `GET /model-info` returns the model version, threshold, features, and limitations |
| F-03 | `POST /predict` accepts one record with the 19 required features |
| F-04 | `/predict` returns default probability, review decision, threshold, and model version |
| F-05 | The review decision is true when probability is greater than or equal to 0.17 |
| F-06 | Clients cannot change the threshold through a request |
| F-07 | Invalid inputs return HTTP 422 |
| F-08 | The service refuses to start if the model checksum or feature list is invalid |
| F-09 | The API documentation states that it is a demo and must not receive real customer data |

### Input validation

| Feature | Rule |
|---|---|
| `LIMIT_BAL` | Number greater than 0 |
| `PAY_0`, `PAY_2` to `PAY_6` | Integer from -2 to 9 |
| `BILL_AMT1` to `BILL_AMT6` | Number; negative values are allowed |
| `PAY_AMT1` to `PAY_AMT6` | Number greater than or equal to 0 |
| Other fields | Rejected, including `SEX`, `MARRIAGE`, `AGE`, and `EDUCATION` |

## 6. Testing and CI/CD

### Automated tests

- Valid and invalid API requests
- Missing, extra, and out-of-range fields
- Threshold boundary behavior
- Model checksum and feature-list checks
- Regression test comparing predictions for synthetic fixtures with stored expected values

### GitHub Actions

| Trigger | Work |
|---|---|
| Pull request and push to `main` | Run tests, build the image, and smoke test `/health` and `/predict` |
| Push to `main` after success | Publish the image to GHCR with commit SHA and `latest` tags |
| Manual run | Generate the drift report and store it as a workflow artifact |

## 7. Deployment

- Deploy the Docker app to a public Hugging Face Space.
- Keep the repository, image, and Space on the same model version.
- Store no secrets in the image or repository.

## 8. Drift check

The check demonstrates detection using data from a fixed split with random seed 42. It retrieves the dataset at runtime and does not store it in the repository.

| Scenario | Reference | Current | Expected result |
|---|---|---|---|
| Control | One random half of the test set | Other half | No dataset-level drift |
| Shifted | Test set | Copy with deliberately shifted features | Drift detected |

The report covers input features and predicted probabilities. Actual outcomes are unavailable in this scenario, so drift is a proxy signal and does not prove lower performance.

## 9. Logging and privacy

- Log only time, path, status code, processing time, and model version.
- Do not log request bodies or input values.
- Do not store requests.

## 10. Process

| Step | Work | Output |
|---|---|---|
| 1. Setup | Create the repository and Codespace | Repository, environment |
| 2. Model import | Retrieve the model and record versions and checksum | Model, metadata |
| 3. API | Implement the endpoints and validation | FastAPI app |
| 4. Tests | Add automated tests | Test suite |
| 5. Container | Create and verify the Docker image | Dockerfile, image |
| 6. CI | Add test, build, and smoke-test automation | CI workflow |
| 7. CD | Publish images to GHCR | Published image |
| 8. Deployment | Deploy to Hugging Face Spaces | Public demo |
| 9. Drift check | Add the report and workflow | Drift report |
| 10. Reporting | Record results and limitations | `README.md`, `results.md` |
| 11. Publishing | Run final checks and publish | Public repository |

## 11. Non-functional requirements

| Item | Requirement |
|---|---|
| Reproducibility | Pin Python and dependency versions |
| Compute | Run on CPU only |
| Security | Store no secrets, avoid running the container as root, and load only the verified model |
| Compatibility | Predictions match stored values within a small numerical tolerance |
| Documentation | Publish API examples, limitations, and data attribution in English |

## 12. Acceptance criteria

- [ ] A new clone can run the tests and start the API using the documented steps.
- [ ] `/health`, `/model-info`, and `/predict` work locally and in the container.
- [ ] The service refuses to start with an invalid model checksum.
- [ ] Demographic and undefined input fields are rejected.
- [ ] Clients cannot change the 0.17 threshold.
- [ ] Prediction regression tests pass in CI.
- [ ] CI blocks image publishing when tests or smoke tests fail.
- [ ] A versioned image is published to GHCR.
- [ ] The public demo works and states its restrictions.
- [ ] Request values do not appear in logs.
- [ ] The drift report shows no dataset-level drift in the control and detects drift in the shifted scenario.
- [ ] The repository is public, in English, attributed, and free of secrets.

## 13. Assumptions and limitations

- The model and its limitations are inherited from `credit-default-threshold`, including the assumed cost ratio.
- Without authentication or rate limiting, the public service is suitable only as a demo.
- Free hosting may sleep, restart, or impose resource limits, so availability is not guaranteed.
- A deliberately shifted dataset demonstrates the mechanism but does not represent real production drift.
- This project does not satisfy legal, regulatory, or model-risk requirements for real credit decisions.