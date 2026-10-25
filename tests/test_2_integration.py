"""Integration tests: check that the API and the model work together.

FastAPI's TestClient calls the app directly in memory, so no real server or
network port is needed.
"""
import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    print(f"\nGET /health -> {response.status_code} {response.json()}")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    payload = {"x": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    print(f"\nPOST /predict {payload} -> {response.status_code} {response.json()}")

    assert response.status_code == 200
    assert response.json() == {"prediction": 0}


@pytest.mark.parametrize(
    "payload",
    [
        {"x": [5.1, 3.5]},              # too few values
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0]},  # too many values
        {"x": []},                      # empty list
        {"x": ["a", "b", "c", "d"]},    # not numbers
        {},                             # missing field
    ],
)
def test_predict_rejects_invalid_input(payload):
    response = client.post("/predict", json=payload)
    print(f"\nPOST /predict {payload} -> {response.status_code}")

    # 422 = the request was understood but is invalid. The server must not crash (500).
    assert response.status_code == 422
