"""
app.py
------
Flask web application for the "Rising Waters" flood-prediction project.

Routes:
    /          -> home.html        (landing page)
    /Predict   -> index.html       (weather-parameter input form)  [GET]
    /predict   -> handles POST     (runs the model, redirects)     [POST]
    /chance    -> chance.html      (flood predicted)
    /no_chance -> no_chance.html   (no flood predicted)

Run:
    python app.py
Then open:
    http://127.0.0.1:5000/
"""

import os

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "floods.save")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "transform.save")

FEATURE_ORDER = ["AnnualRainfall", "CloudVisibility", "Temperature", "Humidity", "SeasonalRainfall"]

# Load the trained model + fitted scaler once at startup.
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/Predict")
def predict_form():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        annual_rainfall = float(request.form["annual_rainfall"])
        cloud_visibility = float(request.form["cloud_visibility"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        seasonal_rainfall = float(request.form["seasonal_rainfall"])
    except (KeyError, ValueError):
        return render_template(
            "index.html", error="Please fill in all five fields with valid numbers."
        )

    # Build a single-row DataFrame with the SAME column order used in training.
    input_df = pd.DataFrame(
        [[annual_rainfall, cloud_visibility, temperature, humidity, seasonal_rainfall]],
        columns=FEATURE_ORDER,
    )

    # Scale with the exact saved scaler, then predict with the saved model.
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if int(prediction) == 1:
        return redirect(url_for("chance"))
    return redirect(url_for("no_chance"))


@app.route("/chance")
def chance():
    return render_template("chance.html")


@app.route("/no_chance")
def no_chance():
    return render_template("no_chance.html")


if __name__ == "__main__":
    app.run(debug=True)
