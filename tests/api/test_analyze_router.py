"""
Tests endpoint POST /analyze

Stratégie : on mocke le client Anthropic pour ne pas dépendre d'une vraie clé API.
Les tests vérifient :
- 503 si clé absente
- 200 + { message } si l'appel Claude réussit (mock)
- 503 sur erreur Anthropic (AuthenticationError, RateLimitError, exception générique)
- Validation Pydantic sur le payload (422)
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.rate_limit import limiter


@pytest.fixture(autouse=True)
def reset_state():
    """Réinitialise le rate limiter et le singleton Anthropic entre chaque test."""
    import src.api.analyze_router as _router
    limiter._storage.reset()
    _router._anthropic_client = None  # force la recréation du client à chaque test
    yield
    _router._anthropic_client = None


@pytest.fixture
def client():
    return TestClient(app)


def base_profile(**overrides):
    profile = {
        "emotionId": "stress",
        "mode": "adult",
        "userText": "je suis stressé par mon travail",
        "selfScore": None,
        "selfReportAnswers": None,
        "mlScore": 0.4,
        "finalScore": 0.4,
        "distressLevel": "elevated",
        "clinicalDimensions": [],
        "clinicalProfile": "adjustment",
    }
    profile.update(overrides)
    return profile


def _mock_anthropic_response(text: str):
    """Construit un mock de message.content[0].text."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    response.usage.output_tokens = 42
    return response


# ─── Sans clé API ─────────────────────────────────────────────────────────────

class TestNoApiKey:
    def test_returns_503_when_key_missing(self, client):
        """Aucune clé configurée → 503 pour dégradation gracieuse."""
        with patch("src.api.analyze_router.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = ""
            resp = client.post("/analyze", json=base_profile())
        assert resp.status_code == 503


# ─── Appel Claude réussi (mock) ───────────────────────────────────────────────

class TestSuccessfulCall:
    def _call_with_mock(self, client, profile_override=None):
        profile = base_profile(**(profile_override or {}))
        with patch("src.api.analyze_router.get_settings") as mock_settings, \
             patch("src.api.analyze_router.anthropic") as mock_anthropic:
            mock_settings.return_value.anthropic_api_key = "sk-test"
            mock_anthropic.Anthropic.return_value.messages.create.return_value = (
                _mock_anthropic_response("Votre stress est tout à fait compréhensible. Vous n'êtes pas seul.")
            )
            mock_anthropic.AuthenticationError = Exception
            mock_anthropic.RateLimitError = Exception
            resp = client.post("/analyze", json=profile)
        return resp

    def test_returns_200(self, client):
        resp = self._call_with_mock(client)
        assert resp.status_code == 200

    def test_returns_message_field(self, client):
        resp = self._call_with_mock(client)
        data = resp.json()
        assert "message" in data
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0

    def test_works_with_kids_mode(self, client):
        resp = self._call_with_mock(client, {"mode": "kids"})
        assert resp.status_code == 200

    def test_works_with_crisis_profile(self, client):
        resp = self._call_with_mock(client, {
            "clinicalProfile": "crisis",
            "distressLevel": "critical",
        })
        assert resp.status_code == 200


# ─── Erreurs Anthropic → 503 ──────────────────────────────────────────────────

class TestAnthropicErrors:
    def _call_with_error(self, client, error_class_name: str):
        with patch("src.api.analyze_router.get_settings") as mock_settings, \
             patch("src.api.analyze_router.anthropic") as mock_anthropic:
            mock_settings.return_value.anthropic_api_key = "sk-test"
            error_class = type(error_class_name, (Exception,), {})
            setattr(mock_anthropic, error_class_name, error_class)
            mock_anthropic.Anthropic.return_value.messages.create.side_effect = error_class("error")
            # L'autre erreur doit aussi être définie
            other = "RateLimitError" if error_class_name == "AuthenticationError" else "AuthenticationError"
            setattr(mock_anthropic, other, type(other, (Exception,), {}))
            resp = client.post("/analyze", json=base_profile())
        return resp

    def test_auth_error_returns_503(self, client):
        resp = self._call_with_error(client, "AuthenticationError")
        assert resp.status_code == 503

    def test_rate_limit_error_returns_503(self, client):
        resp = self._call_with_error(client, "RateLimitError")
        assert resp.status_code == 503

    def test_generic_error_returns_503(self, client):
        with patch("src.api.analyze_router.get_settings") as mock_settings, \
             patch("src.api.analyze_router.anthropic") as mock_anthropic:
            mock_settings.return_value.anthropic_api_key = "sk-test"
            mock_anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
            mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
            mock_anthropic.Anthropic.return_value.messages.create.side_effect = RuntimeError("unexpected")
            resp = client.post("/analyze", json=base_profile())
        assert resp.status_code == 503


# ─── Validation Pydantic ──────────────────────────────────────────────────────

class TestValidation:
    def test_missing_fields_returns_422(self, client):
        resp = client.post("/analyze", json={"emotionId": "stress"})
        assert resp.status_code == 422

    def test_invalid_distress_level_returns_422(self, client):
        resp = client.post("/analyze", json=base_profile(distressLevel="extreme"))
        assert resp.status_code == 422

    def test_empty_text_returns_422(self, client):
        resp = client.post("/analyze", json=base_profile(userText=""))
        assert resp.status_code == 422

    def test_score_above_1_returns_422(self, client):
        resp = client.post("/analyze", json=base_profile(mlScore=1.5))
        assert resp.status_code == 422

    def test_invalid_emotion_id_returns_422(self, client):
        resp = client.post("/analyze", json=base_profile(emotionId="<script>alert(1)</script>"))
        assert resp.status_code == 422

    def test_invalid_clinical_dimension_returns_422(self, client):
        resp = client.post("/analyze", json=base_profile(clinicalDimensions=["invalid_dimension"]))
        assert resp.status_code == 422
