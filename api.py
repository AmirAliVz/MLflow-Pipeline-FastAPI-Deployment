#!/usr/bin/env python
# coding: utf-8

# import statements
from fastapi import FastAPI, HTTPException
import json
import numpy as np
import pickle
import datetime
from sklearn.pipeline import Pipeline
# Import the airport encodings file

f = open('logs/airport_encodings.json')
 
# returns JSON object as a dictionary
airports = json.load(f)

# Load trained model
model = pickle.load(open("logs/finalized_model.pkl", "rb"))

def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1.

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport

    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    temp = np.zeros(len(airports))
    if airport in airports:
        temp[airports.get(airport)] = 1
        temp = temp.T
        return temp
    else:
        return None

# TODO:  write the back-end logic to provide a prediction given the inputs
# requires finalized_model.pkl to be loaded
# the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# YOUR CODE GOES HERE

# ----------------------------
# Backend Prediction Logic
# ----------------------------

def convert_time_to_seconds(time_value: int):
    """
    Converts HHMM time format to seconds since midnight.
    Example: 930 -> 09:30 -> seconds since midnight
    """

    time_str = str(time_value).zfill(4)

    hour = int(time_str[:2])
    minute = int(time_str[2:])

    seconds = hour * 3600 + minute * 60

    return seconds


def predict_delay(arrival_airport: str, departure_time: int, arrival_time: int):
    """
    Generates the feature vector required by the trained model
    and returns the predicted departure delay.
    """

    airport_vector = create_airport_encoding(arrival_airport, airports)


    if airport_vector is None:
        raise ValueError("Arrival airport not recognized")

    dep_seconds = convert_time_to_seconds(departure_time)
    arr_seconds = convert_time_to_seconds(arrival_time)


    # create feature vector
    features = np.hstack((airport_vector, [dep_seconds, arr_seconds]))
    features = features.reshape(1, -1)

    prediction = model.predict(features)

    return float(prediction[0][0])


# TODO:  write the API endpoints.
# YOUR CODE GOES HERE


# ----------------------------
# API Endpoints
# ----------------------------

#Initializing API
app = FastAPI()

# Root Endpoint
@app.get("/")
def root():
    return {"message": "Flight Delay Prediction API is running"}


# Prediction Endpoint
@app.get("/predict/delays")
def predict_delays(arrival_airport: str, departure_time: int, arrival_time: int):

    try:

        delay = predict_delay(arrival_airport, departure_time, arrival_time)

        return {
            "arrival_airport": arrival_airport,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "predicted_average_departure_delay_minutes": delay
        }

    except ValueError as e:

        raise HTTPException(status_code=400, detail=str(e))

    except Exception:

        raise HTTPException(status_code=500, detail="Prediction failed")