"""
preprocess.py
--------------
This file is responsible for cleaning the raw data and converting it into
a format that a Linear Regression model can understand.

Machine learning models only understand NUMBERS. So any text/category
columns (like "traffic_level") must be converted into numeric codes
before training. This process is called ENCODING.

We also handle MISSING VALUES here, because models cannot train on
empty/NaN cells.
"""

import pandas as pd

# ----------------------------------------------------------------------
# Mapping dictionary: converts traffic_level text into numbers.
# We use ordinal encoding (low < medium < high) because traffic has a
# natural order/ranking - it's not just a random category, it represents
# increasing severity. This ordering is important information for the model.
# ----------------------------------------------------------------------
TRAFFIC_MAPPING = {
    "low": 0,
    "medium": 1,
    "high": 2
}


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Step 1: Load the raw CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Handle missing values.

    Why: If any column has empty (NaN) values, the model training will
    crash or give wrong results. We choose to fill missing numeric values
    with the COLUMN MEAN, because mean is a simple, unbiased estimate
    that doesn't drastically change the data's overall pattern.

    (Alternative strategies could be median or dropping rows, but for a
    beginner project, mean-filling is simple and effective.)
    """
    df = df.copy()  # avoid modifying the original DataFrame accidentally

    # Fill missing restaurant_rating values with the average rating
    if df["restaurant_rating"].isnull().sum() > 0:
        mean_rating = df["restaurant_rating"].mean()
        df["restaurant_rating"] = df["restaurant_rating"].fillna(mean_rating)

    return df


def encode_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Convert the traffic_level text column ("low"/"medium"/"high")
    into numbers (0/1/2) using the TRAFFIC_MAPPING dictionary defined above.
    """
    df = df.copy()
    df["traffic_level_encoded"] = df["traffic_level"].map(TRAFFIC_MAPPING)
    return df


def preprocess_data(csv_path: str):
    """
    Full preprocessing pipeline. This is the ONE function that model.py
    will call to get clean, ready-to-train data.

    Steps performed (in order):
      1. Load raw data from CSV
      2. Fill missing values
      3. Encode traffic_level as numbers
      4. Split into features (X) and target (y)

    Returns:
        X (pd.DataFrame): input features used for prediction
        y (pd.Series): target variable (delivery_time) the model must learn to predict
    """
    df = load_data(csv_path)
    df = handle_missing_values(df)
    df = encode_traffic(df)

    # Features (inputs) the model will learn from
    feature_columns = ["distance", "order_hour", "restaurant_rating", "traffic_level_encoded"]
    X = df[feature_columns]

    # Target (output) the model is trying to predict
    y = df["delivery_time"]

    return X, y


if __name__ == "__main__":
    # Quick manual test: run "python preprocess.py" to see the cleaned data
    X, y = preprocess_data("data.csv")
    print("Features (X) sample:")
    print(X.head())
    print("\nTarget (y) sample:")
    print(y.head())
    print("\nAny missing values left in X?", X.isnull().sum().sum())
