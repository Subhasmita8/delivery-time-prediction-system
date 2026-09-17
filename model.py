"""
model.py
---------
This file handles:
    1. Training a Linear Regression model on our delivery data
    2. Evaluating how good the model is (using MAE)
    3. Saving the trained model to disk so the API can reuse it
    4. Loading the saved model and making predictions on new orders

WHY LINEAR REGRESSION?
Linear Regression is one of the simplest and most explainable ML models.
It assumes the target (delivery_time) is a weighted SUM of the input
features, like this:

    delivery_time = (w1 * distance) + (w2 * order_hour) +
                    (w3 * restaurant_rating) + (w4 * traffic_level) + b

Where w1, w2, w3, w4 are "coefficients" (weights) the model learns during
training, and b is the "intercept" (a base value). This makes it very easy
to explain: each coefficient tells us exactly how much delivery time
changes when that feature increases by 1 unit.
"""

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from preprocess import preprocess_data

MODEL_FILE_PATH = "delivery_model.pkl"


def train_model(csv_path: str = "data.csv"):
    """
    Trains the Linear Regression model and returns the trained model
    along with its evaluation score.
    """

    # Step 1: Get clean features (X) and target (y) using our preprocessing pipeline
    X, y = preprocess_data(csv_path)

    # Step 2: Split data into TRAINING set and TESTING set.
    # Why split? We train the model on one part of the data (80%) and test
    # it on data it has NEVER SEEN before (20%). This tells us how well the
    # model will perform on real, future orders - not just data it memorized.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Create and train the Linear Regression model.
    # ".fit()" is where the actual "learning" happens - the model looks at
    # X_train and y_train and figures out the best coefficients (weights)
    # that minimize prediction error.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model on the unseen TEST set.
    predictions = model.predict(X_test)

    # MAE = Mean Absolute Error = average of |actual - predicted| across all test rows.
    # It tells us, ON AVERAGE, how many minutes our predictions are off by.
    mae = mean_absolute_error(y_test, predictions)

    # Step 5: Print coefficients so we can explain what the model learned.
    feature_names = X.columns.tolist()
    print("===== MODEL TRAINING COMPLETE =====")
    print(f"Mean Absolute Error (MAE) on test data: {mae:.2f} minutes")
    print("\nModel Coefficients (impact of each feature on delivery time):")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name:25s} -> {coef:+.3f} minutes per unit increase")
    print(f"  {'intercept (base time)':25s} -> {model.intercept_:+.3f} minutes")
    print("====================================")

    # Step 6: Save the trained model to disk using joblib.
    # This lets our FastAPI app load the ALREADY-TRAINED model instantly,
    # instead of retraining every time the server starts.
    joblib.dump(model, MODEL_FILE_PATH)
    print(f"\nModel saved to '{MODEL_FILE_PATH}'")

    return model, mae


def load_model():
    """
    Loads the previously trained and saved model from disk.
    Used by the FastAPI app to make predictions without retraining.
    """
    return joblib.load(MODEL_FILE_PATH)


def predict_delivery_time(model, distance: float, order_hour: int, rating: float, traffic: int) -> float:
    """
    Uses the trained model to predict delivery time for ONE new order.

    Parameters map directly to the features the model was trained on:
        distance    -> distance in km
        order_hour  -> hour of day (0-23)
        rating      -> restaurant rating (1-5)
        traffic     -> traffic level already encoded as a number (0=low, 1=medium, 2=high)

    Returns:
        Predicted delivery time in minutes (float)
    """
    # The model expects input with the SAME COLUMN NAMES AND ORDER used during
    # training: [distance, order_hour, restaurant_rating, traffic_level_encoded]
    # We use a DataFrame (instead of a plain list) so scikit-learn recognizes
    # the feature names and doesn't raise a mismatch warning.
    import pandas as pd
    input_features = pd.DataFrame(
        [[distance, order_hour, rating, traffic]],
        columns=["distance", "order_hour", "restaurant_rating", "traffic_level_encoded"]
    )

    prediction = model.predict(input_features)

    # model.predict() returns an array (e.g. [34.2]), so we take the first value
    return round(float(prediction[0]), 2)


if __name__ == "__main__":
    # Running "python model.py" directly will train the model end-to-end.
    trained_model, test_mae = train_model()

    # Quick sanity check: predict delivery time for a sample order
    sample_prediction = predict_delivery_time(
        trained_model, distance=5.0, order_hour=13, rating=4.5, traffic=1
    )
    print(f"\nSample prediction -> 5km order at 1PM, 4.5-star restaurant, medium traffic:")
    print(f"Predicted delivery time: {sample_prediction} minutes")
