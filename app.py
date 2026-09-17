"""
app.py
------
This file creates a web API using FastAPI so that other applications
(a website, a mobile app, Postman, curl, etc.) can send order details
and get back a predicted delivery time.

FLOW:
    1. When the server starts, we load our already-trained model from disk.
       (If no trained model exists yet, we train one automatically.)
    2. We define an endpoint: POST /predict-time
    3. The client sends order details as JSON.
    4. We run the model and send back the predicted delivery time as JSON.
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import train_model, load_model, predict_delivery_time, MODEL_FILE_PATH

# ----------------------------------------------------------------------
# Create the FastAPI application instance
# ----------------------------------------------------------------------
app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Predicts food delivery time (in minutes) using a simple Linear Regression model.",
    version="1.0.0"
)

# ----------------------------------------------------------------------
# Load (or train) the model ONCE when the server starts.
# This is more efficient than retraining on every single API request.
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_FILE_PATH):
    print("No saved model found. Training a new model...")
    train_model()

model = load_model()
print("Model loaded and ready to serve predictions!")


# ----------------------------------------------------------------------
# Define the shape of the INPUT the API expects, using Pydantic.
# FastAPI uses this to automatically validate incoming JSON requests
# (e.g., it will reject a request if "distance" is missing or not a number).
# ----------------------------------------------------------------------
class DeliveryRequest(BaseModel):
    distance: float = Field(..., gt=0, description="Distance between restaurant and customer, in km")
    order_hour: int = Field(..., ge=0, le=23, description="Hour of the day the order was placed (0-23)")
    rating: float = Field(..., ge=1, le=5, description="Restaurant rating (1.0 to 5.0)")
    traffic: int = Field(..., ge=0, le=2, description="Traffic level: 0=low, 1=medium, 2=high")

    class Config:
        json_schema_extra = {
            "example": {
                "distance": 5.5,
                "order_hour": 19,
                "rating": 4.2,
                "traffic": 2
            }
        }


# ----------------------------------------------------------------------
# Define the shape of the OUTPUT the API will return.
# ----------------------------------------------------------------------
class DeliveryResponse(BaseModel):
    predicted_delivery_time: float


# ----------------------------------------------------------------------
# Root endpoint - just a friendly welcome message so visiting "/" isn't empty.
# ----------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Food Delivery Time Prediction API is running. Go to /docs to try it out."}


# ----------------------------------------------------------------------
# Main prediction endpoint.
# ----------------------------------------------------------------------
@app.post("/predict-time", response_model=DeliveryResponse)
def predict_time(request: DeliveryRequest):
    """
    Takes order details and returns the predicted delivery time in minutes.
    """
    predicted_time = predict_delivery_time(
        model=model,
        distance=request.distance,
        order_hour=request.order_hour,
        rating=request.rating,
        traffic=request.traffic
    )

    return DeliveryResponse(predicted_delivery_time=predicted_time)


# ----------------------------------------------------------------------
# This allows running the app directly with: python app.py
# (Alternatively, and more commonly, run with: uvicorn app:app --reload)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
