# 🚴 Delivery Time Prediction (ETA System)

## 📌 Project Overview

This project predicts the **estimated delivery time (ETA)** for a food order based on simple factors like distance, time of order, restaurant rating, and traffic conditions.

The goal is to simulate how food delivery platforms estimate delivery time for customers.

---

## 🧠 Problem Statement

Customers want to know how long their order will take.
This project builds a machine learning model to predict delivery time in minutes.

---

## ⚙️ Features Used

* Distance (in km)
* Order Hour (0–23)
* Restaurant Rating (1–5)
* Traffic Level (low, medium, high)

---

## 🤖 Model Used

* Linear Regression (from scikit-learn)

### Why this model?

* Simple and easy to understand
* Shows how each feature affects delivery time

---

## 📊 Evaluation

* Used **Mean Absolute Error (MAE)**
* MAE shows average difference between predicted and actual time

---

## 🌐 API Endpoints

### 1. Predict Delivery Time

**POST /predict-time**

#### Input:

```json
{
  "distance": 5,
  "order_hour": 20,
  "rating": 4.2,
  "traffic": 2
}
```

#### Output:

```json
{
  "predicted_delivery_time": 32
}
```

---

## 🗂️ Project Structure

```
├── app.py
├── model.py
├── preprocess.py
├── data.csv
├── requirements.txt
└── README.md
```

## 📌 Key Learnings

* Basics of regression
* Feature impact on prediction
* Building API using FastAPI
* Working with real-world type data

---

## 🚀 Future Improvements

* Add more features like weather and traffic data
* Use advanced models for better accuracy
* Deploy the project online
