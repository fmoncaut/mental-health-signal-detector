import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.schemas import PredictResponse


@pytest.fixture
def client():
    return TestClient(app)


def test_health_model_not_loaded(client):
    with patch("src.api.main.get_model", side_effect=FileNotFoundError("modèle absent")):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is False


def test_predict_missing_model(client):
    with patch("src.api.main.get_model", side_effect=FileNotFoundError("pas de modèle")):
        response = client.post("/predict", json={"text": "I feel hopeless", "model_type": "baseline"})
    assert response.status_code == 503


def test_predict_invalid_model_type(client):
    response = client.post("/predict", json={"text": "test", "model_type": "unknown"})
    assert response.status_code == 422  # validation Pydantic


def test_predict_empty_text(client):
    response = client.post("/predict", json={"text": "", "model_type": "baseline"})
    assert response.status_code == 422


def test_predict_whitespace_text(client):
    response = client.post("/predict", json={"text": "   \n\t  ", "model_type": "baseline"})
    assert response.status_code == 422


def test_predict_accepts_mental_roberta_model_type(client):
    fake_response = PredictResponse(
        label=1,
        score_distress=0.91,
        model="mental_roberta",
        text_preview="I feel hopeless",
        detected_lang="en",
    )
    with patch("src.api.main.get_model", return_value=object()), patch(
        "src.api.main.run_prediction", return_value=fake_response
    ):
        response = client.post("/predict", json={"text": "I feel hopeless", "model_type": "mental_roberta"})
    assert response.status_code == 200
    assert response.json()["model"] == "mental_roberta"
