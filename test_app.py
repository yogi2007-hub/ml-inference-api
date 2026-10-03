from fastapi.testclient import TestClient

from app import app
from train_model import train_model

client = TestClient(app)


def setup_module() -> None:
    train_model()


def test_predict_returns_probabilities() -> None:
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 200
    probabilities = response.json()["probabilities"]
    assert len(probabilities) == 3
    assert abs(sum(probabilities) - 1.0) < 1e-6


def test_predict_rejects_wrong_number_of_features() -> None:
    response = client.post("/predict", json={"features": [1.0, 2.0]})

    assert response.status_code == 422
