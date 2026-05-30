# Branch: `Deployment` — FastAPI + Docker

## What This Does

The trained model artifacts from Task 2 (`finalized_model.pkl` and `airport_encodings.json`) are served through a **FastAPI REST API** that accepts flight parameters and returns a predicted departure delay in JSON format. The application is containerized with Docker for consistent deployment across environments.

## API Endpoint

**`POST /predict/delays`**

Accepts flight scheduling inputs, encodes airport codes via `airport_encodings.json`, and passes structured data through the trained model pipeline — returning a predicted delay in minutes.

**`GET /`**

Health check — returns: `"Flight Delay Prediction API is running"`

### Swagger UI

| API Docs (/docs) |
|---|
| _placeholder_ |

---

## Unit Tests

Four test cases implemented with `pytest`:

| Test | Purpose |
|---|---|
| `test_root` | Verifies the base endpoint is operational |
| `test_predict_valid` | Valid inputs return a prediction (e.g., airport `SEA`) |
| `test_predict_invalid_airport` | Invalid airport code (`XXX`) returns an appropriate error |
| `test_missing_parameter` | Missing required field returns a validation error |

| pytest Output |
|---|
| _placeholder_ |

---

## Running Locally

```bash
uvicorn api:app --reload
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Docker

### Pull and Run from Registry

```bash
docker pull <registry-url>/flight-delay-api:latest
docker run -p 8000:8000 flight-delay-api
```

### Build Locally

```bash
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```

The container installs all pinned dependencies from `requirements.txt`, copies model artifacts, and starts the API on port `8000`.

> **Note on Python versions:** GitLab CI uses Python 3.14.3; the Docker image uses Python 3.12.1. Both have been tested and confirmed compatible.

---

## Project Structure (`Deployment`)

```
project/
├── api.py                        # FastAPI application
├── test_api.py                   # pytest unit tests
├── finalized_model.pkl           # Trained model pipeline from Task 2
├── airport_encodings.json        # Airport encoding map from Task 2
├── Dockerfile                    # Container definition
├── requirements.txt              # Pinned dependencies
└── README.md
```

---

## Key Libraries

| Library | Purpose |
|---|---|
| `mlflow` | Experiment tracking, pipeline orchestration |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for FastAPI |
| `pytest` | Unit testing |
| `scikit-learn` | Ridge regression, polynomial features, pipeline |
| `pandas` | Data manipulation |
| `dvc` | Dataset versioning |
| `docker` | Containerization |
