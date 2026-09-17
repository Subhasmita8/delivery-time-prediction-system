# 🍔 Food Delivery Time Prediction System (ETA Model)

A simple, beginner-friendly machine learning project that predicts how many
**minutes** a food delivery will take, based on distance, time of day,
restaurant rating, and traffic conditions.

This project is intentionally kept **simple and explainable** — it uses
**Linear Regression**, not complex "black box" models — so every part of it
can be understood and confidently explained (e.g. in an interview).

---

## 📁 Project Structure

```
food_delivery_eta/
├── data.csv              # Sample dataset (300 rows) of past deliveries
├── generate_data.py      # Script that generated the synthetic dataset
├── preprocess.py         # Cleans data & converts categories to numbers
├── model.py               # Trains the Linear Regression model & predicts
├── app.py                 # FastAPI web API that serves predictions
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🧠 What Problem Does This Solve?

When you order food online, the app shows an "estimated delivery time"
(ETA). This project builds a simplified version of that system:

**Given:** distance, order hour, restaurant rating, and traffic level
**Predict:** how many minutes the delivery will take

---

## 📊 The Dataset

`data.csv` contains 300 simulated orders with these columns:

| Column               | Meaning                                      |
|----------------------|-----------------------------------------------|
| `distance`           | Distance from restaurant to customer (km)     |
| `order_hour`         | Hour the order was placed (0–23)              |
| `restaurant_rating`  | Restaurant's rating (1.0–5.0)                 |
| `traffic_level`      | Traffic condition: `low`, `medium`, `high`    |
| `delivery_time`      | **Target** — actual delivery time (minutes)   |

The data was generated synthetically (see `generate_data.py`) using a
realistic formula: longer distance, heavier traffic, and rush hours all
increase delivery time, while higher-rated restaurants are slightly faster.
Random "noise" was added so it behaves like real-world data. A few missing
values were also added on purpose, so the preprocessing step has something
real to clean.

---

## 🧹 Preprocessing (`preprocess.py`)

Before training, the raw data goes through three steps:

1. **Load the CSV** into a pandas DataFrame.
2. **Handle missing values** — a few `restaurant_rating` values are missing
   (`NaN`). We fill them with the **column average**, a simple and safe way
   to avoid crashing the model without distorting the data too much.
3. **Encode categories as numbers** — `traffic_level` is text
   (`low`/`medium`/`high`), but ML models only understand numbers. We map:
   - `low` → 0
   - `medium` → 1
   - `high` → 2

   We use this simple 0/1/2 mapping (called **ordinal encoding**) because
   traffic has a natural order — "high" traffic is genuinely "more" than
   "low" traffic — so representing it as increasing numbers makes sense.

---

## 🤖 The Model (`model.py`)

We use **Linear Regression** from scikit-learn. It's the simplest possible
regression model, and it works like this:

```
delivery_time =  (w1 × distance)
               + (w2 × order_hour)
               + (w3 × restaurant_rating)
               + (w4 × traffic_level)
               + b
```

- `w1, w2, w3, w4` are **coefficients** — numbers the model learns during
  training. Each one tells you exactly how much delivery time changes when
  that feature goes up by 1 unit (with everything else held constant).
- `b` is the **intercept** — the "base" delivery time when all features are 0.

### Example coefficients from training:

| Feature                  | Coefficient | Meaning                                                  |
|---------------------------|:-----------:|-----------------------------------------------------------|
| `distance`                | +2.56       | Each extra km adds ~2.6 minutes                           |
| `order_hour`               | +0.17       | Slight increase later in the day                          |
| `restaurant_rating`        | -0.95       | Each extra rating point saves ~1 minute (better restaurants are faster) |
| `traffic_level` (encoded)  | +8.76       | Going from low→medium or medium→high traffic adds ~8.8 minutes |
| intercept                  | +10.22      | Base time (packing, handoff) even before other factors    |

This is the biggest advantage of Linear Regression: **you can explain
exactly why** the model predicted a certain number. There's no mystery.

### Training process:
1. Split data into 80% training / 20% testing.
2. Train (`model.fit()`) on the training set.
3. Evaluate on the test set (data the model has never seen).
4. Save the trained model to `delivery_model.pkl` using `joblib`, so the
   API doesn't need to retrain every time it starts.

---

## 📏 Evaluation: Mean Absolute Error (MAE)

We evaluate the model using **MAE (Mean Absolute Error)**:

```
MAE = average of |actual_time - predicted_time|  across all test orders
```

**In simple terms:** if MAE = 3.4 minutes, it means that, on average, our
model's prediction is off by about 3.4 minutes from the real delivery time.
That's a small, acceptable error for an ETA system — good enough to show
customers a helpful estimate.

We use MAE instead of more complex metrics because it's **directly
interpretable in minutes** — easy to explain to anyone, technical or not.

---

## 🌐 The API (`app.py`)

Built with **FastAPI**. It loads the trained model once at startup, then
serves predictions instantly for each request.

### Endpoint: `POST /predict-time`

**Request body:**
```json
{
  "distance": 5.5,
  "order_hour": 19,
  "rating": 4.2,
  "traffic": 2
}
```
(`traffic`: 0 = low, 1 = medium, 2 = high)

**Response:**
```json
{
  "predicted_delivery_time": 41.02
}
```

FastAPI also automatically validates input (e.g. rejects `order_hour: 30`
since it must be between 0–23) and generates interactive API docs.

---

## ▶️ How to Run This Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
```bash
python generate_data.py
```
This creates a fresh `data.csv`. A copy is already included, so you can skip this step.

### 3. Train the model
```bash
python model.py
```
This trains the Linear Regression model, prints the MAE and coefficients,
and saves the trained model to `delivery_model.pkl`.

### 4. Run the API server
```bash
uvicorn app:app --reload
```
The server will start at: `http://127.0.0.1:8000`

### 5. Try it out
Open your browser to `http://127.0.0.1:8000/docs` for an interactive UI,
**or** use curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict-time" \
  -H "Content-Type: application/json" \
  -d '{"distance": 5.5, "order_hour": 19, "rating": 4.2, "traffic": 2}'
```

Expected response:
```json
{"predicted_delivery_time": 41.02}
```


## ✅ Summary

| File              | Purpose                                              |
|-------------------|--------------------------------------------------------|
| `generate_data.py`| Creates the synthetic dataset (`data.csv`)             |
| `data.csv`        | The dataset (300 rows) used for training               |
| `preprocess.py`   | Cleans missing values & encodes categorical features    |
| `model.py`        | Trains Linear Regression, evaluates it, and predicts    |
| `app.py`          | FastAPI web service exposing `/predict-time`            |
| `requirements.txt`| List of Python packages needed to run the project        |

This project is small enough to fully understand end-to-end, but complete
enough to demonstrate the full ML workflow: **data → preprocessing → model
training → evaluation → deployment as an API.**
