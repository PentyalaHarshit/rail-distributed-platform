import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app import create_app
import numpy as np


class DummyMLB:
    def inverse_transform(self, arr):
        return [("Laptop", "Shoes")]


class DummyClassifier:
    def predict(self, X):
        return np.array([[1, 1]])


class DummyRegressor:
    def predict(self, X):
        return np.array([[20.0, 35.0]])


def test_home_page():
    app = create_app(load_real_models=False)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Rail Distributed Platform" in response.text
    assert "<form" in response.text


def test_predict_page():
    app = create_app(load_real_models=False)

    app.state.models = {
        "xgb_item": DummyClassifier(),
        "xgb_train": DummyClassifier(),
        "xgb_reg_discount": DummyRegressor(),
        "xgb_reg_price": DummyRegressor(),
        "mlb_item": DummyMLB(),
        "mlb_train": DummyMLB(),
    }

    client = TestClient(app)

    response = client.post(
        "/predict",
        data={
            "Name": "Harshit",
            "Age": 25,
            "Phone": "9876543210",
            "Date": "2026-04-04"
        }
    )

    assert response.status_code == 200
    assert "Prediction Result" in response.text
    assert "Harshit" in response.text
    assert "Laptop" in response.text