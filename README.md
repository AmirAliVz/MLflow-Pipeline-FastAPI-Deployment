# Flight Delay Prediction — MLflow Pipeline & FastAPI Deployment

This repository contains a two-stage machine learning project that builds and deploys a **polynomial ridge regression model** to predict airport departure delays based on flight scheduling data from the Bureau of Transportation Statistics.

- **Branch `Pipeline`** — Data preprocessing pipeline and MLflow experiment tracking
- **Branch `Deployment`** — FastAPI REST API serving the trained model, containerized with Docker

> **Note on Dataset Availability**
> The raw dataset (`T_ONTIME_REPORTING.csv`) has been removed from this repository as it is sourced from a restricted context and cannot be shared publicly. The DVC metafile (`T_ONTIME_REPORTING.csv.dvc`) is retained to allow the dataset to be reproduced and versioned within the pipeline. All pipeline outputs and MLflow experiment artifacts are stored for reference and presentation purposes.

---

## Branch Overview

| Branch | Purpose |
|---|---|
| `Pipeline` | Data ingestion, formatting, cleaning, model training, MLflow experiment tracking |
| `Deployment` | FastAPI `/predict/delays` endpoint, unit testing with pytest, Docker containerization |

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
