from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    return {
        "xgb_item": joblib.load("xgb_item.pkl"),
        "xgb_train": joblib.load("xgb_train.pkl"),
        "xgb_reg_discount": joblib.load("xgb_reg_discount.pkl"),
        "xgb_reg_price": joblib.load("xgb_reg_price.pkl"),
        "mlb_item": joblib.load("mlb_item.pkl"),
        "mlb_train": joblib.load("mlb_train.pkl"),
    }


def create_features_from_date(date_str: str) -> pd.DataFrame:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return pd.DataFrame({
        "Year": [date_obj.year],
        "Month": [date_obj.month],
        "Day": [date_obj.day],
        "Weekday": [date_obj.weekday()],
        "IsWeekend": [1 if date_obj.weekday() >= 5 else 0],
        "Season": [((date_obj.month % 12) + 3) // 3 - 1],
        "Month_sin": [np.sin(2 * np.pi * date_obj.month / 12)],
        "Month_cos": [np.cos(2 * np.pi * date_obj.month / 12)]
    })


def create_app(load_real_models: bool = True) -> FastAPI:
    app = FastAPI()

    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

    if load_real_models:
        app.state.models = load_models()
    else:
        app.state.models = None

    @app.get("/", response_class=HTMLResponse)
    def home():
        return """
        <html>
        <head>
            <title>Rail Distributed Platform</title>
            <style>
                body {
                    background-image: url('/static/Screenshot_2026-04-04_142608.png');
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    height: 100vh;
                    color: white;
                }
                .overlay {
                    width: 100%;
                    height: 100vh;
                    background-color: rgba(0, 0, 0, 0.45);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .form-container {
                    background-color: rgba(0, 0, 0, 0.65);
                    padding: 30px;
                    border-radius: 12px;
                    width: 380px;
                    box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
                }
                h2 {
                    text-align: center;
                    margin-bottom: 25px;
                }
                label {
                    display: block;
                    margin-top: 10px;
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                input[type=text],
                input[type=number],
                input[type=tel],
                input[type=date] {
                    width: 100%;
                    padding: 10px;
                    border: none;
                    border-radius: 6px;
                    margin-bottom: 10px;
                    box-sizing: border-box;
                }
                input[type=submit] {
                    width: 100%;
                    padding: 12px;
                    background-color: #1E90FF;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                    cursor: pointer;
                    margin-top: 10px;
                }
                input[type=submit]:hover {
                    background-color: #0f6cc9;
                }
            </style>
        </head>
        <body>
            <div class="overlay">
                <div class="form-container">
                    <h2>Rail Distributed Platform</h2>
                    <form action="/predict" method="post">
                        <label for="Name">Name</label>
                        <input type="text" id="Name" name="Name" required>

                        <label for="Age">Age</label>
                        <input type="number" id="Age" name="Age" required>

                        <label for="Phone">Phone</label>
                        <input type="tel" id="Phone" name="Phone" required>

                        <label for="Date">Date</label>
                        <input type="date" id="Date" name="Date" required>

                        <input type="submit" value="Predict">
                    </form>
                </div>
            </div>
        </body>
        </html>
        """

    @app.post("/predict", response_class=HTMLResponse)
    def predict(
        Name: str = Form(...),
        Age: int = Form(...),
        Phone: str = Form(...),
        Date: str = Form(...)
    ):
        logger.info(f"Prediction triggered for {Name} on {Date}")

        X_new = create_features_from_date(Date)
        try:
            X_new = create_features_from_date(Date)

            models = app.state.models
            xgb_item = models["xgb_item"]
            xgb_train = models["xgb_train"]
            xgb_reg_discount = models["xgb_reg_discount"]
            xgb_reg_price = models["xgb_reg_price"]
            mlb_item = models["mlb_item"]
            mlb_train = models["mlb_train"]

            pred_items = mlb_item.inverse_transform(xgb_item.predict(X_new))
            pred_trains = mlb_train.inverse_transform(xgb_train.predict(X_new))
            pred_discounts = xgb_reg_discount.predict(X_new)
            pred_prices = xgb_reg_price.predict(X_new)

            html_output = f"""
            <html>
            <head>
                <title>Prediction Result</title>
            </head>
            <body>
                <h2>Rail Distributed Platform</h2>
                <h3>Prediction Result</h3>
                <p><b>Name:</b> {Name}</p>
                <p><b>Age:</b> {Age}</p>
                <p><b>Phone:</b> {Phone}</p>
                <p><b>Date:</b> {Date}</p>
            """

            if len(pred_items[0]) == 0 or len(pred_trains[0]) == 0:
                html_output += """
                <h3>No discounts predicted for this date.</h3>
                <a href="/">Go Back</a>
                </body>
                </html>
                """
                return html_output

            discount_list = [
                f"{item} at {train}: {discount:.0f}% off, now €{price:.2f}"
                for item, train, discount, price in zip(
                    pred_items[0],
                    pred_trains[0],
                    pred_discounts[0],
                    pred_prices[0]
                )
            ]

            html_output += "<h3>List of Discounts</h3><ul>"
            for d in discount_list:
                html_output += f"<li>{d}</li>"

            html_output += """
                </ul>
                <a href="/">Go Back</a>
            </body>
            </html>
            """
            return html_output

        except Exception as e:
            return f"""
            <html>
            <body>
                <h2>Error</h2>
                <p>{str(e)}</p>
                <a href="/">Go Back</a>
            </body>
            </html>
            """

    return app

    @app.post("/api/predict")
    def api_predict(Date: str = Body(...)):

        logger.info(f"API prediction requested for date: {Date}")

    # prediction logic...

app = create_app(load_real_models=True)