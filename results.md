# Results: Credit Default Model API

## 1. Summary

The fixed credit default model and 0.17 review threshold are served through a public FastAPI demo. The same image passes automated smoke tests in CI and is deployed to Render by digest.

Predictions were identical to the environment that exported the model on 20 synthetic records. The drift check reported no dataset-level drift in the control and detected drift in the deliberately shifted data.

This is a learning demo. It is not suitable for real customer data or credit decisions.

## 2. Delivered system

| Component | Details |
|---|---|
| API | FastAPI with `/health`, `/model-info`, and `/predict` |
| Model | `credit-default-lgbm-small:v0` |
| Threshold | 0.17, fixed |
| Runtime | Python 3.13, scikit-learn 1.6.1, LightGBM 4.6.0 |
| Container | Non-root image with a health check |
| Registry | `ghcr.io/meru-dog/credit-default-api` |
| Deployment | Render Free web service |
| Deployed image digest | `sha256:ecf1f53627b42b8741bbf6f7c855acd65e54d7361aa8deef8e53226fead02919` |

The deployed digest was built before later commits. Those commits did not change `app/`, `model/`, `requirements.txt`, or the `Dockerfile`.

## 3. Verification results

| Check | Result |
|---|---|
| Automated tests | 54 passed |
| Prediction regression | Largest difference 0.0 on 20 records in Codespaces; within 1e-9 in the local, CI, and Render smoke tests |
| Container user | UID 1000 |
| Container health | Healthy |
| Modified model file | Startup stopped with exit code 3 |
| Failing test in a pull request | `test` failed and `image` was skipped |
| Public image | Pulled without login and passed the smoke test |
| Render | Passed the smoke test through the public URL |

## 4. Drift check

Both scenarios use a fixed test split. The test PR-AUC reproduced the recorded value of 0.5541, confirming that the model and split matched before the reports ran.

| Scenario | Reference | Current | Drifted columns | Dataset drift | Prediction drift score |
|---|---:|---:|---:|---|---:|
| Control | 3,000 test rows | Other 3,000 test rows | 0 of 20 | No | 0.062 |
| Shifted | 6,000 test rows | Shifted copy | 19 of 20 | Yes | 1.142 |

The prediction drift score uses normalized Wasserstein distance with a threshold of 0.1. Dataset drift is flagged when at least 50% of columns drift.

Scores for every column are available in [docs/drift_summary.json](docs/drift_summary.json).

The shifted scenario applies these deliberate changes:

- `LIMIT_BAL` multiplied by 0.8
- `PAY_` status codes increased by 1, capped at 9
- `BILL_AMT` columns multiplied by 1.3
- `PAY_AMT` columns multiplied by 0.5

## 5. Design decisions

| Decision | Rationale |
|---|---|
| FastAPI instead of a model-serving framework | The small CPU model does not need batching or GPU scaling, and the serving mechanics stay visible |
| Model stored in the repository | Anyone can start the service without W&B credentials |
| Checksum verified before `joblib.load` | Loading a joblib file can execute code |
| Publish only after the smoke test | The exact image that passed is the image published |
| Deploy by digest | A digest identifies one image, unlike a movable tag |
| Validation errors omit input values | The default response echoes inputs and fails on NaN or Infinity |
| Render instead of Hugging Face Spaces | Hugging Face Docker Spaces now require paid billing |

## 6. Acceptance review

| Criterion | Status | Evidence |
|---|---|---|
| New clone can test and start the API | Passed | Clean CI checkout and Codespace setup |
| Endpoints work locally and in the container | Passed | Manual checks and smoke tests |
| Invalid model checksum stops startup | Passed | Automated tests and container run |
| Demographic and undefined fields are rejected | Passed | Automated and smoke tests |
| Clients cannot change the threshold | Passed | `threshold` field is rejected |
| Prediction regression tests pass in CI | Passed | CI workflow |
| CI blocks publishing on failure | Passed | Failing test skipped `image`; publishing follows the smoke test |
| Versioned image is published to GHCR | Passed | Image pulled without login |
| Public demo works and states restrictions | Passed | Render smoke test and API description |
| Request values do not appear in logs | Passed | Automated test and container log check |
| Drift report meets expected outcomes | Passed | Section 4 |
| Repository is public, in English, attributed, and free of secrets | Passed | Repository review |

## 7. Limitations

- The public service has no authentication or rate limiting, so it is suitable only as a demo.
- The Free plan sleeps after 15 minutes without traffic, takes about a minute to wake, and does not guarantee availability.
- Memory use on Render was not measured.
- Actual outcomes are unavailable, so drift is only a proxy signal and does not prove lower performance.
- The shifted data demonstrates detection but does not represent real production drift.
- The image is about 708 MB and was not size-optimized.
- The model inherits the limitations of `credit-default-threshold`, including the assumed cost ratio, 2005 Taiwanese data, and lack of a fairness audit.
- This project does not satisfy legal, regulatory, or model-risk requirements for real credit decisions.
