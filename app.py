"""
app.py
------
FastAPI backend for the Food Delivery Time Prediction project.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import train_model, load_model, predict_delivery_time, MODEL_FILE_PATH


# ----------------------------------------------------------------------
# Create the FastAPI application ONCE
# ----------------------------------------------------------------------
app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Predicts food delivery time (in minutes) using a Linear Regression model.",
    version="1.0.0"
)


# ----------------------------------------------------------------------
# Allow the frontend to communicate with this API
# ----------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Load (or train) the model when the server starts
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_FILE_PATH):
    print("No saved model found. Training a new model...")
    train_model()

model = load_model()

print("Model loaded and ready to serve predictions!")


# ----------------------------------------------------------------------
# Input data
# ----------------------------------------------------------------------
class DeliveryRequest(BaseModel):
    distance: float = Field(
        ...,
        gt=0,
        description="Distance between restaurant and customer, in km"
    )

    order_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the day the order was placed (0-23)"
    )

    rating: float = Field(
        ...,
        ge=1,
        le=5,
        description="Restaurant rating (1.0 to 5.0)"
    )

    traffic: int = Field(
        ...,
        ge=0,
        le=2,
        description="Traffic level: 0=low, 1=medium, 2=high"
    )

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
# Output data
# ----------------------------------------------------------------------
class DeliveryResponse(BaseModel):
    predicted_delivery_time: float


# ----------------------------------------------------------------------
# Home endpoint
# ----------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "message": "Food Delivery Time Prediction API is running. Go to /docs to try it out."
    }


# ----------------------------------------------------------------------
# Prediction endpoint
# ----------------------------------------------------------------------
@app.post("/predict-time", response_model=DeliveryResponse)
def predict_time(request: DeliveryRequest):

    predicted_time = predict_delivery_time(
        model=model,
        distance=request.distance,
        order_hour=request.order_hour,
        rating=request.rating,
        traffic=request.traffic
    )

    return DeliveryResponse(
        predicted_delivery_time=predicted_time
    )


# ----------------------------------------------------------------------
# Run server with: python app.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )