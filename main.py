"""Simple FastAPI server for collecting sensor data and sending updates.

This service provides:
- HTTP GET /data : returns all stored sensor records from `data.csv` as a list of dicts
- HTTP POST /data : accept `temperature` and `humidity`, append to `data.csv`, run prediction, and notify WebSocket clients
- WebSocket /ws : pushes the most recent data to connected clients when new data arrives
"""

from datetime import datetime  # used for timestamping new sensor records
from fastapi.middleware.cors import CORSMiddleware  # enable CORS for browser clients

from fastapi import FastAPI, WebSocket, Depends
from pydantic import BaseModel 

import numpy as np
import joblib  

import pandas as pd 
import asyncio, json 

app = FastAPI()

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorData(BaseModel):
    """Schema for incoming sensor payloads."""
    temperature: float  # degrees (e.g., Celsius)
    humidity: float     # relative humidity (0-100)


# Stores the most recently posted sensor reading (sent to WebSocket clients)
latest_data = None

# Load a pre-trained logistic regression model from disk.
# The code assumes `model.pkl` exists and exposes `predict` and `predict_proba`.
model = joblib.load("model.pkl") 


def predict(temperature, humidity):
    """

    - Converts humidity to a 'dryness' feature (100 - humidity)
    - Builds a 1x2 numpy array for the model
    - Returns the probability (0-100) 
    """
  

    # Convert humidity to dryness as a feature for the model
    dryness = 100 - humidity
    new_data = np.array([[temperature, dryness]])

    # Run the model to get a predicted class and the probability for positive class
    prediction = model.predict(new_data)
    prediction_proba = ((model.predict_proba(new_data)[:, 1])[0] * 100).round()

    print(prediction_proba)

    return prediction_proba


# Async event that signals when new data has been posted.
# WebSocket handlers wait on this event to push updates to clients.
update_event = asyncio.Event()


@app.get("/data")
async def get_data():
    """Return all stored sensor records from `data.csv` as JSON-serializable list."""
    df = pd.read_csv("data.csv")
    return df.to_dict(orient="records")


@app.post("/data")
async def post_data(data: SensorData):
    """Accept new sensor data, persist it, notify WebSocket clients, and return a prediction."""
    global latest_data

    # Build the new row: ISO timestamp, temperature, humidity
    new_entry = [datetime.now().isoformat(), data.temperature, data.humidity]

    # Keep an in-memory copy of the latest entry for quick access by websockets
    latest_data = {"datetime": new_entry[0], "temperature": new_entry[1], "humidity": new_entry[2]}

    # Append to CSV storage
    df = pd.read_csv("data.csv")
    df.loc[len(df)] = new_entry
    df.to_csv("data.csv", index=False)

    # Wake any awaiting websocket clients so they can push the new data
    update_event.set()

    # Run model prediction and return the probability
    prediction_proba = predict(data.temperature, data.humidity)

    return {"probability": prediction_proba}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Simple WebSocket endpoint that pushes the latest data whenever it changes.

    Behavior:
    - Accept a new websocket connection
    - Wait for the `update_event` to be set
    - Send `latest_data` (JSON string) to the client
    - Clear the event and repeat

    """
    await websocket.accept()
    try:
        while True:
            # Block until new data is available
            await update_event.wait()

            # Send the latest data as JSON text
            data = json.dumps(latest_data)
            await websocket.send_text(data)

            # Reset the event so we wait for the next update
            update_event.clear()
    except Exception:
        # A client disconnect or other error will typically raise here
        print("Client disconnected")

