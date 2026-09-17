"""
generate_data.py
-----------------
This script creates a SYNTHETIC (fake but realistic) dataset for the
Food Delivery Time Prediction project.

Why synthetic data?
Real delivery datasets are hard to get publicly, so we simulate one using
a simple formula that mimics how delivery time actually behaves in real life:
    - Longer distance  -> longer delivery time
    - Heavy traffic     -> longer delivery time
    - Peak hours (lunch/dinner) -> longer delivery time
    - Higher-rated restaurants -> slightly faster (better kitchen efficiency)

We also add random "noise" so the data looks realistic and not a perfect
straight line (real life always has some randomness).
"""

import numpy as np
import pandas as pd

# Set a random seed so results are reproducible (same data every time we run this)
np.random.seed(42)

# Number of rows (orders) we want in our dataset
NUM_ROWS = 300

# ---- 1. Generate raw feature columns ----

# distance in kilometers: most orders are between 1 km and 20 km
distance = np.round(np.random.uniform(1, 20, NUM_ROWS), 2)

# order_hour: hour of the day the order was placed (0-23, 24-hour format)
order_hour = np.random.randint(0, 24, NUM_ROWS)

# restaurant_rating: rating between 1.0 and 5.0
restaurant_rating = np.round(np.random.uniform(1.0, 5.0, NUM_ROWS), 1)

# traffic_level: categorical variable - low, medium, or high
traffic_level = np.random.choice(["low", "medium", "high"], size=NUM_ROWS, p=[0.4, 0.4, 0.2])

# ---- 2. Build delivery_time (target variable) using a realistic formula ----

# Base delivery time (minimum minutes needed regardless of anything else,
# e.g., time to pack the order and hand it to the delivery partner)
base_time = 10

# Every km adds roughly 2.5 minutes of travel time
distance_effect = distance * 2.5

# Traffic adds extra minutes: low = 0, medium = 8, high = 18
traffic_map_for_generation = {"low": 0, "medium": 8, "high": 18}
traffic_effect = np.array([traffic_map_for_generation[t] for t in traffic_level])

# Rush hours (12-14 lunch, 19-21 dinner) add extra delay due to more orders
rush_hour_effect = np.where(
    ((order_hour >= 12) & (order_hour <= 14)) | ((order_hour >= 19) & (order_hour <= 21)),
    7,   # extra minutes during rush hours
    0
)

# Higher-rated restaurants are usually a bit faster/more efficient
# (rating 5 -> saves ~4 minutes, rating 1 -> saves 0 minutes)
rating_effect = (restaurant_rating - 1) * -1.0

# Random noise to simulate real-world unpredictability (e.g. weather, delays)
noise = np.random.normal(0, 3, NUM_ROWS)

# Final delivery time = sum of all effects (and can't be negative)
delivery_time = base_time + distance_effect + traffic_effect + rush_hour_effect + rating_effect + noise
delivery_time = np.round(np.clip(delivery_time, 5, None), 1)  # minimum 5 minutes, rounded to 1 decimal

# ---- 3. Combine everything into a DataFrame ----
df = pd.DataFrame({
    "distance": distance,
    "order_hour": order_hour,
    "restaurant_rating": restaurant_rating,
    "traffic_level": traffic_level,
    "delivery_time": delivery_time
})

# ---- 4. Intentionally introduce a few missing values ----
# This is done ON PURPOSE so our preprocessing code has something real to clean.
# In real-world data, missing values are very common.
missing_indices = np.random.choice(df.index, size=5, replace=False)
df.loc[missing_indices, "restaurant_rating"] = np.nan

# ---- 5. Save to CSV ----
df.to_csv("data.csv", index=False)

print(f"Generated {len(df)} rows and saved to data.csv")
print(df.head())
