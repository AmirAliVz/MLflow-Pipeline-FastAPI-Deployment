from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Flight Delay Prediction API is running"
    }



def test_predict_valid():
    response = client.get(
        "/predict/delays",
        params={
            "arrival_airport": "SEA",  # use valid airport from the JSON
            "departure_time": 900,
            "arrival_time": 1130
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "predicted_average_departure_delay_minutes" in data
    assert isinstance(data["predicted_average_departure_delay_minutes"], float)


def test_predict_invalid_airport():
    response = client.get(
        "/predict/delays",
        params={
            "arrival_airport": "XXX",  # invalid airport
            "departure_time": 900,
            "arrival_time": 1130
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Arrival airport not recognized"


def test_missing_parameter():
    response = client.get(
        "/predict/delays",
        params={
            "arrival_airport": "SEA",
            "departure_time": 900
            # missing arrival_time
        }
    )

    assert response.status_code == 422



