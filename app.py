from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import os
import logging

from routes import get_valid_trains

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# SETTINGS
# =========================================================
DATE_FORMAT_TRAINING = "%d-%m-%Y"   # CSV date format
CSV_FILE = "XCM_expanded.csv"

DATE_COL = "Date"
TRAIN_COL = "Train"
DISCOUNT_COL = "Discount_%"
PRICE_COL = "Discounted_Price"

TOP_K_TRAINS = 5
WALLPAPER_PATH = "/static/Screenshot_2026-04-04_142608.png"


# =========================================================
# LOAD MODELS
# =========================================================
def load_models():
    return {
        "train_model": joblib.load("train_model.pkl"),
        "discount_model": joblib.load("discount_model.pkl"),
        "price_model": joblib.load("price_model.pkl"),
        "train_le": joblib.load("train_label_encoder.pkl"),
        "feature_cols": joblib.load("feature_columns.pkl"),
    }


# =========================================================
# LOAD HISTORY DATA
# =========================================================
def load_history_data():
    df = pd.read_csv(CSV_FILE)

    required_cols = [DATE_COL, TRAIN_COL, DISCOUNT_COL, PRICE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[required_cols].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=DATE_FORMAT_TRAINING, errors="coerce")
    df[DISCOUNT_COL] = pd.to_numeric(df[DISCOUNT_COL], errors="coerce")
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")

    df = df.dropna(subset=[DATE_COL, TRAIN_COL, DISCOUNT_COL, PRICE_COL]).copy()

    df = (
        df.groupby([DATE_COL, TRAIN_COL], as_index=False)
        .agg({
            DISCOUNT_COL: "mean",
            PRICE_COL: "mean"
        })
    )

    df = df[df[DISCOUNT_COL] >= 0].copy()
    df = df[df[PRICE_COL] >= 0].copy()

    df["Year"] = df[DATE_COL].dt.year
    df["Month"] = df[DATE_COL].dt.month
    df["Day"] = df[DATE_COL].dt.day
    df["Weekday"] = df[DATE_COL].dt.weekday
    df["IsWeekend"] = (df["Weekday"] >= 5).astype(int)
    df["WeekOfYear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["Quarter"] = df[DATE_COL].dt.quarter
    df["DayOfYear"] = df[DATE_COL].dt.dayofyear

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)
    df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

    df = df.sort_values([TRAIN_COL, DATE_COL]).reset_index(drop=True)
    return df


# =========================================================
# TRAIN STATS
# =========================================================
def build_train_stats(df):
    return df.groupby(TRAIN_COL).agg(
        Train_freq=(TRAIN_COL, "count"),
        Train_avg_discount=(DISCOUNT_COL, "mean"),
        Train_avg_price=(PRICE_COL, "mean"),
        Train_discount_std=(DISCOUNT_COL, "std"),
        Train_price_std=(PRICE_COL, "std"),
    ).reset_index()


# =========================================================
# ADD HISTORY FEATURES
# =========================================================
def add_history_features(df, train_le):
    df = df.copy()

    # keep only train names known by label encoder
    df = df[df[TRAIN_COL].isin(train_le.classes_)].copy()

    df["Train_encoded"] = train_le.transform(df[TRAIN_COL])

    df = df.sort_values([TRAIN_COL, DATE_COL]).reset_index(drop=True)
    g = df.groupby(TRAIN_COL, group_keys=False)

    for lag in [1, 2, 3, 7, 14]:
        df[f"Discount_lag_{lag}"] = g[DISCOUNT_COL].shift(lag)
        df[f"Price_lag_{lag}"] = g[PRICE_COL].shift(lag)

    for win in [3, 5, 7, 14]:
        df[f"Discount_roll_mean_{win}"] = (
            g[DISCOUNT_COL].shift(1).rolling(win).mean().reset_index(level=0, drop=True)
        )
        df[f"Price_roll_mean_{win}"] = (
            g[PRICE_COL].shift(1).rolling(win).mean().reset_index(level=0, drop=True)
        )

    for win in [3, 7]:
        df[f"Discount_roll_std_{win}"] = (
            g[DISCOUNT_COL].shift(1).rolling(win).std().reset_index(level=0, drop=True)
        )
        df[f"Price_roll_std_{win}"] = (
            g[PRICE_COL].shift(1).rolling(win).std().reset_index(level=0, drop=True)
        )

    df["Discount_exp_mean"] = (
        g[DISCOUNT_COL].shift(1).expanding().mean().reset_index(level=0, drop=True)
    )
    df["Price_exp_mean"] = (
        g[PRICE_COL].shift(1).expanding().mean().reset_index(level=0, drop=True)
    )

    train_stats = build_train_stats(df)
    df = df.merge(train_stats, on=TRAIN_COL, how="left")

    df["Price_per_discount"] = df[PRICE_COL] / (1 - df[DISCOUNT_COL] / 100 + 1e-5)
    df["Train_x_Month"] = df["Train_encoded"] * df["Month"]
    df["Train_x_Weekend"] = df["Train_encoded"] * df["IsWeekend"]

    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    fill_zero_cols = [
        c for c in df.columns
        if ("lag_" in c) or ("roll_" in c) or (c in [
            "Discount_exp_mean", "Price_exp_mean",
            "Train_discount_std", "Train_price_std"
        ])
    ]
    for c in fill_zero_cols:
        df[c] = df[c].fillna(0)

    other_fill_cols = [
        "Train_freq", "Train_avg_discount", "Train_avg_price",
        "Price_per_discount", "Train_x_Month", "Train_x_Weekend"
    ]
    for c in other_fill_cols:
        if c in df.columns:
            if df[c].isna().all():
                df[c] = 0
            else:
                df[c] = df[c].fillna(df[c].median())

    return df, train_stats


# =========================================================
# BUILD FUTURE DATE FEATURES
# =========================================================
def build_features_for_future_date(
    history_df,
    future_date,
    train_name,
    label_encoder,
    train_stats_df,
    feature_cols
):
    future_date = pd.to_datetime(future_date)
    train_hist = history_df[history_df[TRAIN_COL] == train_name].sort_values(DATE_COL).copy()

    row = {}
    row["Train_encoded"] = int(label_encoder.transform([train_name])[0])

    row["Year"] = future_date.year
    row["Month"] = future_date.month
    row["Day"] = future_date.day
    row["Weekday"] = future_date.weekday()
    row["IsWeekend"] = int(row["Weekday"] >= 5)
    row["WeekOfYear"] = int(future_date.isocalendar().week)
    row["Quarter"] = ((future_date.month - 1) // 3) + 1
    row["DayOfYear"] = future_date.timetuple().tm_yday

    row["Month_sin"] = np.sin(2 * np.pi * row["Month"] / 12)
    row["Month_cos"] = np.cos(2 * np.pi * row["Month"] / 12)
    row["Weekday_sin"] = np.sin(2 * np.pi * row["Weekday"] / 7)
    row["Weekday_cos"] = np.cos(2 * np.pi * row["Weekday"] / 7)
    row["DayOfYear_sin"] = np.sin(2 * np.pi * row["DayOfYear"] / 365.25)
    row["DayOfYear_cos"] = np.cos(2 * np.pi * row["DayOfYear"] / 365.25)

    def get_lag(col, lag_days, default=0.0):
        target_day = future_date - pd.Timedelta(days=lag_days)
        matched = train_hist.loc[train_hist[DATE_COL] == target_day, col]
        return float(matched.iloc[0]) if len(matched) else default

    for lag in [1, 2, 3, 7, 14]:
        row[f"Discount_lag_{lag}"] = get_lag(DISCOUNT_COL, lag, 0.0)
        row[f"Price_lag_{lag}"] = get_lag(PRICE_COL, lag, 0.0)

    hist_before = train_hist[train_hist[DATE_COL] < future_date].sort_values(DATE_COL)

    for win in [3, 5, 7, 14]:
        d = hist_before[DISCOUNT_COL].tail(win)
        p = hist_before[PRICE_COL].tail(win)
        row[f"Discount_roll_mean_{win}"] = float(d.mean()) if len(d) else 0.0
        row[f"Price_roll_mean_{win}"] = float(p.mean()) if len(p) else 0.0

    for win in [3, 7]:
        d = hist_before[DISCOUNT_COL].tail(win)
        p = hist_before[PRICE_COL].tail(win)
        row[f"Discount_roll_std_{win}"] = float(d.std()) if len(d) > 1 else 0.0
        row[f"Price_roll_std_{win}"] = float(p.std()) if len(p) > 1 else 0.0

    row["Discount_exp_mean"] = float(hist_before[DISCOUNT_COL].mean()) if len(hist_before) else 0.0
    row["Price_exp_mean"] = float(hist_before[PRICE_COL].mean()) if len(hist_before) else 0.0

    stat_row = train_stats_df[train_stats_df[TRAIN_COL] == train_name]
    if len(stat_row):
        row["Train_freq"] = float(stat_row["Train_freq"].iloc[0])
        row["Train_avg_discount"] = float(stat_row["Train_avg_discount"].iloc[0])
        row["Train_avg_price"] = float(stat_row["Train_avg_price"].iloc[0])
        row["Train_discount_std"] = float(stat_row["Train_discount_std"].fillna(0).iloc[0])
        row["Train_price_std"] = float(stat_row["Train_price_std"].fillna(0).iloc[0])
    else:
        row["Train_freq"] = 0.0
        row["Train_avg_discount"] = 0.0
        row["Train_avg_price"] = 0.0
        row["Train_discount_std"] = 0.0
        row["Train_price_std"] = 0.0

    row["Price_per_discount"] = row["Train_avg_price"] / (1 - row["Train_avg_discount"] / 100 + 1e-5)
    row["Train_x_Month"] = row["Train_encoded"] * row["Month"]
    row["Train_x_Weekend"] = row["Train_encoded"] * row["IsWeekend"]

    feat = pd.DataFrame([row])

    for col in feature_cols:
        if col not in feat.columns:
            feat[col] = 0

    return feat[feature_cols]


# =========================================================
# PREDICT FOR DATE AND ROUTE
# =========================================================
def predict_for_date_and_route(
    future_date_str,
    departure,
    arrival,
    history_df,
    train_stats,
    models,
    top_k=TOP_K_TRAINS
):
    future_date = pd.to_datetime(future_date_str, format="%Y-%m-%d")

    train_model = models["train_model"]
    discount_model = models["discount_model"]
    price_model = models["price_model"]
    train_le = models["train_le"]
    feature_cols = models["feature_cols"]

    valid_trains = get_valid_trains(departure, arrival)

    if not valid_trains:
        return pd.DataFrame(columns=[
            "train",
            "train_probability",
            "predicted_discount_percent",
            "predicted_price"
        ])

    rows = []

    for train_name in valid_trains:
        if train_name not in list(train_le.classes_):
            continue

        feat = build_features_for_future_date(
            history_df=history_df,
            future_date=future_date,
            train_name=train_name,
            label_encoder=train_le,
            train_stats_df=train_stats,
            feature_cols=feature_cols
        )

        probs = train_model.predict_proba(feat)[0]
        class_idx = int(train_le.transform([train_name])[0])
        train_prob = float(probs[class_idx])

        discount_pred = max(float(discount_model.predict(feat)[0]), 0)
        price_pred = max(np.expm1(float(price_model.predict(feat)[0])), 0)

        rows.append({
            "train": train_name,
            "train_probability": round(train_prob, 4),
            "predicted_discount_percent": round(discount_pred, 2),
            "predicted_price": round(price_pred, 2)
        })

    result = pd.DataFrame(rows)

    if len(result) == 0:
        return result

    return result.sort_values(
        ["train_probability", "predicted_discount_percent", "predicted_price"],
        ascending=[False, False, True]
    ).reset_index(drop=True).head(top_k)


# =========================================================
# SIMPLE BASE PRICE
# =========================================================
def get_simple_route_price(arrival: str, departure: str) -> float:
    key = f"{departure.strip().lower()}_{arrival.strip().lower()}"

    route_prices = {
        "paris_lyon": 120.0,
        "lyon_paris": 115.0,
        "paris_brussels": 110.0,
        "brussels_paris": 108.0,
        "berlin_munich": 140.0,
        "munich_berlin": 138.0,
        "london_paris": 160.0,
        "paris_london": 158.0,
        "madrid_barcelona": 130.0,
        "barcelona_madrid": 128.0,
        "rome_milan": 125.0,
        "milan_rome": 123.0,
        "zurich_geneva": 118.0,
        "geneva_zurich": 117.0,
    }
    return route_prices.get(key, 99.0)


# =========================================================
# APP
# =========================================================
def create_app() -> FastAPI:
    app = FastAPI(title="Rail Booking")

    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

    models = load_models()
    raw_df = load_history_data()
    history_df, train_stats = add_history_features(raw_df, models["train_le"])

    app.state.models = models
    app.state.history_df = history_df
    app.state.train_stats = train_stats

    # -----------------------------------------------------
    # PAGE 1: NAME AND PHONE
    # -----------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def home():
        return f"""
        <html>
        <head>
            <title>Rail Booking</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: url('{WALLPAPER_PATH}') no-repeat center center fixed;
                    background-size: cover;
                }}
                .overlay {{
                    position: fixed;
                    inset: 0;
                    background: rgba(0, 0, 0, 0.45);
                }}
                .container {{
                    position: relative;
                    z-index: 2;
                    width: 440px;
                    margin: 110px auto;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 32px;
                    border-radius: 14px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.28);
                }}
                h2 {{
                    text-align: center;
                    color: #1f3c88;
                    margin-bottom: 24px;
                }}
                label {{
                    display: block;
                    font-weight: bold;
                    margin-top: 12px;
                    margin-bottom: 6px;
                }}
                input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    box-sizing: border-box;
                }}
                button {{
                    width: 100%;
                    margin-top: 22px;
                    padding: 12px;
                    border: none;
                    border-radius: 6px;
                    background: #28a745;
                    color: white;
                    font-size: 15px;
                    font-weight: bold;
                    cursor: pointer;
                }}
                button:hover {{
                    opacity: 0.92;
                }}
            </style>
        </head>
        <body>
            <div class="overlay"></div>
            <div class="container">
                <h2>Enter Passenger Details</h2>
                <form action="/route-page" method="post">
                    <label for="name">Name</label>
                    <input type="text" id="name" name="name" required>

                    <label for="phone">Phone</label>
                    <input type="tel" id="phone" name="phone" required>

                    <button type="submit">Next</button>
                </form>
            </div>
        </body>
        </html>
        """

    # -----------------------------------------------------
    # PAGE 2: DEPARTURE AND ARRIVAL
    # -----------------------------------------------------
    @app.post("/route-page", response_class=HTMLResponse)
    def route_page(
        name: str = Form(...),
        phone: str = Form(...)
    ):
        return f"""
        <html>
        <head>
            <title>Route Details</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f4f7fb;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    width: 520px;
                    margin: 70px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.12);
                }}
                h2 {{
                    text-align: center;
                    color: #1f3c88;
                    margin-bottom: 25px;
                }}
                label {{
                    display: block;
                    font-weight: bold;
                    margin-top: 12px;
                    margin-bottom: 8px;
                }}
                input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    box-sizing: border-box;
                }}
                .btn-row {{
                    display: flex;
                    gap: 12px;
                    margin-top: 22px;
                }}
                button {{
                    flex: 1;
                    padding: 12px;
                    border: none;
                    border-radius: 6px;
                    color: white;
                    font-size: 15px;
                    font-weight: bold;
                    cursor: pointer;
                }}
                .submit-btn {{
                    background: #007bff;
                }}
                .discount-btn {{
                    background: #28a745;
                }}
                button:hover {{
                    opacity: 0.92;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Enter Route Details</h2>

                <form action="/choose-action" method="post">
                    <input type="hidden" name="name" value="{name}">
                    <input type="hidden" name="phone" value="{phone}">

                    <label for="departure">Departure</label>
                    <input type="text" id="departure" name="departure" required>

                    <label for="arrival">Arrival</label>
                    <input type="text" id="arrival" name="arrival" required>

                    <div class="btn-row">
                        <button type="submit" name="action" value="submit" class="submit-btn">Submit</button>
                        <button type="submit" name="action" value="discount" class="discount-btn">Discount</button>
                    </div>
                </form>
            </div>
        </body>
        </html>
        """

    # -----------------------------------------------------
    # HANDLE SUBMIT / DISCOUNT
    # -----------------------------------------------------
    @app.post("/choose-action", response_class=HTMLResponse)
    def choose_action(
        name: str = Form(...),
        phone: str = Form(...),
        departure: str = Form(...),
        arrival: str = Form(...),
        action: str = Form(...)
    ):
        if action == "submit":
            base_price = get_simple_route_price(arrival=arrival, departure=departure)

            return f"""
            <html>
            <head>
                <title>Ticket Price</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background: #f4f7fb;
                        margin: 0;
                        padding: 0;
                    }}
                    .container {{
                        width: 700px;
                        margin: 60px auto;
                        background: white;
                        padding: 30px;
                        border-radius: 12px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.12);
                    }}
                    h2 {{
                        color: #1f3c88;
                    }}
                    .card {{
                        background: #eef4ff;
                        border-left: 6px solid #007bff;
                        padding: 16px;
                        border-radius: 8px;
                        margin-top: 20px;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 18px;
                        text-decoration: none;
                        color: #333;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Ticket Price</h2>
                    <p><b>Name:</b> {name}</p>
                    <p><b>Phone:</b> {phone}</p>
                    <p><b>Departure:</b> {departure}</p>
                    <p><b>Arrival:</b> {arrival}</p>

                    <div class="card">
                        <p><b>Base Price:</b> €{base_price:.2f}</p>
                    </div>

                    <a href="/">Home</a>
                </div>
            </body>
            </html>
            """

        return f"""
        <html>
        <head>
            <title>Select Date</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f4f7fb;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    width: 520px;
                    margin: 70px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.12);
                }}
                h2 {{
                    text-align: center;
                    color: #1f3c88;
                    margin-bottom: 25px;
                }}
                label {{
                    display: block;
                    font-weight: bold;
                    margin-top: 12px;
                    margin-bottom: 8px;
                }}
                input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    box-sizing: border-box;
                }}
                button {{
                    width: 100%;
                    margin-top: 22px;
                    padding: 12px;
                    border: none;
                    border-radius: 6px;
                    background: #28a745;
                    color: white;
                    font-size: 15px;
                    font-weight: bold;
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Select Travel Date</h2>

                <form action="/predict-html" method="post">
                    <input type="hidden" name="name" value="{name}">
                    <input type="hidden" name="phone" value="{phone}">
                    <input type="hidden" name="departure" value="{departure}">
                    <input type="hidden" name="arrival" value="{arrival}">

                    <label for="travel_date">Travel Date</label>
                    <input type="date" id="travel_date" name="travel_date" required>

                    <button type="submit">Submit</button>
                </form>
            </div>
        </body>
        </html>
        """

    # -----------------------------------------------------
    # JSON API
    # -----------------------------------------------------
    @app.post("/predict")
    def predict_api(
        departure: str = Form(...),
        arrival: str = Form(...),
        travel_date: str = Form(...)
    ):
        try:
            result = predict_for_date_and_route(
                future_date_str=travel_date,
                departure=departure,
                arrival=arrival,
                history_df=app.state.history_df,
                train_stats=app.state.train_stats,
                models=app.state.models,
                top_k=TOP_K_TRAINS
            )

            return JSONResponse({
                "departure": departure,
                "arrival": arrival,
                "travel_date": travel_date,
                "predictions": result.to_dict(orient="records")
            })
        except Exception as e:
            logger.exception("Prediction failed")
            return JSONResponse(status_code=500, content={"error": str(e)})

    # -----------------------------------------------------
    # RESULT PAGE
    # -----------------------------------------------------
    @app.post("/predict-html", response_class=HTMLResponse)
    def predict_html(
        name: str = Form(...),
        phone: str = Form(...),
        departure: str = Form(...),
        arrival: str = Form(...),
        travel_date: str = Form(...)
    ):
        try:
            result = predict_for_date_and_route(
                future_date_str=travel_date,
                departure=departure,
                arrival=arrival,
                history_df=app.state.history_df,
                train_stats=app.state.train_stats,
                models=app.state.models,
                top_k=TOP_K_TRAINS
            )

            if len(result) == 0:
                return f"""
                <html>
                <body style="font-family: Arial; padding: 30px;">
                    <h2>No Route Found</h2>
                    <p>No trains available from <b>{departure}</b> to <b>{arrival}</b>.</p>
                    <a href="/">Go Back</a>
                </body>
                </html>
                """

            cards = ""
            for _, row in result.iterrows():
                cards += f"""
                <div class="card">
                    <h3>{row['train']}</h3>
                    <p><b>Probability:</b> {row['train_probability']:.2f}</p>
                    <p><b>Discount:</b> {row['predicted_discount_percent']:.0f}%</p>
                    <p><b>Price:</b> €{row['predicted_price']:.2f}</p>
                </div>
                """

            return f"""
            <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background: #f4f7fb;
                        margin: 0;
                        padding: 0;
                    }}
                    .container {{
                        width: 800px;
                        margin: 50px auto;
                        background: white;
                        padding: 30px;
                        border-radius: 12px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.12);
                    }}
                    h2 {{
                        color: #1f3c88;
                        margin-bottom: 20px;
                    }}
                    .card {{
                        background: #eaf7ed;
                        border-left: 6px solid #28a745;
                        padding: 16px;
                        border-radius: 8px;
                        margin-bottom: 14px;
                        font-size: 16px;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 16px;
                        text-decoration: none;
                        color: #333;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Predicted Discount List</h2>

                    <p><b>Name:</b> {name}</p>
                    <p><b>Phone:</b> {phone}</p>
                    <p><b>Departure:</b> {departure}</p>
                    <p><b>Arrival:</b> {arrival}</p>
                    <p><b>Travel Date:</b> {travel_date}</p>

                    {cards}

                    <a href="/">Home</a>
                </div>
            </body>
            </html>
            """
        except Exception as e:
            logger.exception("HTML prediction failed")
            return f"""
            <html>
            <body style="font-family: Arial; padding: 30px;">
                <h2>Error</h2>
                <p>{str(e)}</p>
                <a href="/">Go Back</a>
            </body>
            </html>
            """

    return app


app = create_app()