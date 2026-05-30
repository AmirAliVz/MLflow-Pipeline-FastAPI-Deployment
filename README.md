# Flight Delay Prediction — MLflow Pipeline & FastAPI Deployment

This repository contains a two-stage machine learning project that builds and deploys a **polynomial ridge regression model** to predict airport departure delays based on flight scheduling data from the Bureau of Transportation Statistics.

- **Branch `Pipeline`** — Data preprocessing pipeline and MLflow experiment tracking
- **Branch `Deployment`** — FastAPI REST API serving the trained model, containerized with Docker

> **Note on Dataset Availability**
> The raw dataset (`T_ONTIME_REPORTING.csv`) has been removed from this repository as it is sourced from a restricted context and cannot be shared publicly. The DVC metafile (`T_ONTIME_REPORTING.csv.dvc`) is retained to allow the dataset to be reproduced and versioned within the pipeline. All pipeline outputs and MLflow experiment artifacts are stored for reference and presentation purposes.

---

## Repo name & description

**Repo name:**
`flight-delay-mlflow-pipeline`

**Description:**
> End-to-end flight departure delay prediction: an MLflow pipeline that preprocesses BTS data and trains a polynomial ridge regression model, deployed as a FastAPI REST API and containerized with Docker.

---

## Branch Overview

| Branch | Purpose |
|---|---|
| `Pipeline` | Data ingestion, formatting, cleaning, model training, MLflow experiment tracking |
| `Deployment` | FastAPI `/predict/delays` endpoint, unit testing with pytest, Docker containerization |

---

# Branch: `Pipeline` — MLflow Pipeline

## What This Does

The pipeline ingests raw flight reporting data, formats and cleans it for modeling, trains a polynomial ridge regression model with cross-validated alpha selection, and logs all parameters, metrics, and artifacts to an MLflow experiment.

Three scripts are orchestrated sequentially by the `MLProject` file:

```
import_format_data.py  →  filter_clean_data.py  →  poly_regressor.py
```

### Stage 1 — Import & Format (`import_format_data.py`)
- Loads `T_ONTIME_REPORTING.csv` from the `data/` directory
- Drops null values (cancelled flights — not useful for delay prediction)
- Outputs `data/formatted_data.csv`
- Logs process details to `import_format_data_log.txt`

### Stage 2 — Filter & Clean (`filter_clean_data.py`)
- Filters to JFK departures only
- Removes duplicate rows
- Removes extreme delay outliers (> 1000 minutes)
- Outputs `data/cleaned_data.csv`
- Logs process details to `filter_clean_data_log.txt`

### Stage 3 — Train Model (`poly_regressor.py`)
- Trains a polynomial ridge regression model on cleaned data
- Logs to MLflow experiment `"Airport Departure Delays"`:
  - **Parameters:** `best_alpha`, `order` (polynomial degree)
  - **Metrics:** MSE, average predicted delay (minutes)
  - **Artifacts:** log files (`.txt`), performance plot (`.jpg`), `airport_encodings.json`, `finalized_model.pkl`

### MLflow Experiment Output

| MLflow UI — Run List | MLflow UI — Run Detail |
|---|---|
| _placeholder_ | _placeholder_ |

| Performance Plot |
|---|
| _placeholder_ |

---

## Environment Setup

```bash
# Clone the pipeline branch
git clone -b pipeline https://gitlab.com/wgu-gitlab-environment/student-repos/avazir1/d602-deployment-task-2.git

# Create and activate the Conda environment
conda env create -f pipeline_env.yaml
conda activate pipeline_env
```

> MLflow must also be installed in your **base** Conda environment (used to launch the pipeline runner commands).

---

## Running the Pipeline

### Full Pipeline (all 3 stages sequentially)

```bash
mlflow run . -e pipeline --experiment-name "Airport Departure Delays"
```

Default parameters: `num_alphas=20`, `order=1`

### Individual Stages

```bash
mlflow run . -e import_data
mlflow run . -e clean_data
mlflow run . -e train_model --experiment-name "Airport Departure Delays" -P num_alphas=20 -P order=1
```

### Custom Hyperparameters

```bash
mlflow run . -e pipeline -P num_alphas=30 -P order=2 --experiment-name "Airport Departure Delays"
```

### Launch MLflow UI

```bash
mlflow ui
```

> **Important:** Avoid spaces in your project directory name. MLflow tracking URIs use URL encoding (`%20` for spaces), which can cause path resolution failures when running locally.

---

## Dataset Schema

| Column | Type |
|---|---|
| `YEAR` | Integer |
| `MONTH` | Integer |
| `DAY` | Integer |
| `DAY_OF_WEEK` | Integer |
| `ORG_AIRPORT` | String |
| `DEST_AIRPORT` | String |
| `SCHEDULED_DEPARTURE` | Integer |
| `DEPARTURE_TIME` | Integer |
| `DEPARTURE_DELAY` | Integer |
| `SCHEDULED_ARRIVAL` | Integer |
| `ARRIVAL_TIME` | Integer |
| `ARRIVAL_DELAY` | Integer |

---

## Project Structure (`Pipeline`)

```
project/
├── data/
│   ├── T_ONTIME_REPORTING.csv.dvc    # DVC metafile — tracks raw dataset version
│   ├── formatted_data.csv            # Output of Stage 1
│   └── cleaned_data.csv              # Output of Stage 2
├── import_format_data.py             # Stage 1
├── filter_clean_data.py              # Stage 2
├── poly_regressor.py                 # Stage 3
├── MLproject                         # Pipeline orchestration
├── pipeline_env.yaml                 # Conda environment definition
└── README.md
```

---
---

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
