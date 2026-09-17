/* =======================================================================
   script.js — Frontend logic for the Delivery ETA Predictor page.

   This file reads the form values, calls the FastAPI backend's
   POST /predict-time endpoint, and updates the "departures board"
   readout with the predicted delivery time.
   ======================================================================= */

// ---------------------------------------------------------------------
// The base URL of the FastAPI backend. Since this HTML/CSS/JS is now a
// standalone site (it might be opened directly as a file, or hosted
// somewhere separate from the API), we point to the backend explicitly
// instead of using a relative path.
//
// If you run the API locally with "uvicorn app:app --reload", it will
// be available at this address by default. Change this if you deploy
// the backend somewhere else (e.g. a live server URL).
// ---------------------------------------------------------------------
const API_BASE_URL = "https://food-delivery-prediction-wfwt.onrender.com";

// ---- Grab references to the elements we need to read from / update ----
const distanceInput = document.getElementById("distance");
const hourInput = document.getElementById("hour");
const ratingInput = document.getElementById("rating");
const trafficSwitch = document.getElementById("traffic-switch");
const form = document.getElementById("order-form");
const submitBtn = document.getElementById("submit-btn");

const distanceValueEl = document.getElementById("distance-value");
const hourValueEl = document.getElementById("hour-value");
const ratingValueEl = document.getElementById("rating-value");

const resultDigits = document.getElementById("result-digits");
const resultDetail = document.getElementById("result-detail");

let selectedTraffic = 1; // default = medium, matches the pre-selected button in the HTML

// Convert a 24-hour value (0-23) into a friendly 12-hour label, e.g. "7:00 PM"
function formatHour(hour) {
  const period = hour >= 12 ? "PM" : "AM";
  let displayHour = hour % 12;
  if (displayHour === 0) displayHour = 12;
  return `${displayHour}:00 ${period}`;
}

// ---- Keep the little value labels next to each slider in sync ----
distanceInput.addEventListener("input", () => {
  distanceValueEl.textContent = `${parseFloat(distanceInput.value).toFixed(1)} km`;
});

hourInput.addEventListener("input", () => {
  hourValueEl.textContent = formatHour(parseInt(hourInput.value, 10));
});

ratingInput.addEventListener("input", () => {
  ratingValueEl.textContent = parseFloat(ratingInput.value).toFixed(1);
});

// ---- Traffic segmented control ----
// Clicking a button selects it and visually deselects the others.
trafficSwitch.addEventListener("click", (event) => {
  const button = event.target.closest(".traffic-switch__option");
  if (!button) return;

  trafficSwitch.querySelectorAll(".traffic-switch__option").forEach((btn) => {
    btn.classList.remove("is-selected");
    btn.setAttribute("aria-checked", "false");
  });

  button.classList.add("is-selected");
  button.setAttribute("aria-checked", "true");
  selectedTraffic = parseInt(button.dataset.value, 10);
});

// ---- Handle form submission: call the API and display the result ----
form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    distance: parseFloat(distanceInput.value),
    order_hour: parseInt(hourInput.value, 10),
    rating: parseFloat(ratingInput.value),
    traffic: selectedTraffic
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "Calculating...";
  resultDetail.classList.remove("is-error");
  resultDetail.textContent = "Calculating...";

  try {
    const response = await fetch(`${API_BASE_URL}/predict-time`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    const minutes = data.predicted_delivery_time;

    // Update the big readout with a small "settle in" animation
    resultDigits.textContent = minutes;
    resultDigits.classList.remove("is-updated");
    void resultDigits.offsetWidth; // restart the CSS animation
    resultDigits.classList.add("is-updated");

    const trafficLabel = { 0: "low", 1: "medium", 2: "high" }[selectedTraffic];
    resultDetail.textContent =
      `Based on ${payload.distance} km at ${formatHour(payload.order_hour)}, ` +
      `a ${payload.rating}-star restaurant, and ${trafficLabel} traffic.`;

  } catch (error) {
    resultDetail.classList.add("is-error");
    resultDetail.textContent =
      "Couldn't reach the prediction API. Make sure the backend is running at " + API_BASE_URL;
    console.error(error);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Estimate delivery time";
  }
});
