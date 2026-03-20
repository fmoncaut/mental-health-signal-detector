"""Tests unitaires — POST /feedback (collecte anonyme opt-in)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "text": "Je me sens très fatigué et triste depuis plusieurs semaines.",
    "emotion": "sadness",
    "distress_level": 2,
    "score_ml": 0.72,
    "consent": True,
}


class TestFeedbackValidation:
    """Validation Pydantic — payloads invalides."""

    def test_consent_false_rejected(self):
        payload = {**VALID_PAYLOAD, "consent": False}
        res = client.post("/feedback", json=payload)
        assert res.status_code == 422

    def test_text_too_long_rejected(self):
        payload = {**VALID_PAYLOAD, "text": "a" * 5001}
        res = client.post("/feedback", json=payload)
        assert res.status_code == 422

    def test_text_empty_rejected(self):
        payload = {**VALID_PAYLOAD, "text": ""}
        res = client.post("/feedback", json=payload)
        assert res.status_code == 422

    def test_invalid_emotion_rejected(self):
        payload = {**VALID_PAYLOAD, "emotion": "rage"}
        res = client.post("/feedback", json=payload)
        assert res.status_code == 422

    def test_distress_level_out_of_range(self):
        for bad in [-1, 5]:
            payload = {**VALID_PAYLOAD, "distress_level": bad}
            res = client.post("/feedback", json=payload)
            assert res.status_code == 422, f"distress_level={bad} should be rejected"

    def test_score_ml_out_of_range(self):
        for bad in [-0.1, 1.1]:
            payload = {**VALID_PAYLOAD, "score_ml": bad}
            res = client.post("/feedback", json=payload)
            assert res.status_code == 422, f"score_ml={bad} should be rejected"

    def test_score_ml_none_accepted(self):
        """score_ml est optionnel (fallback sans ML)."""
        payload = {**VALID_PAYLOAD, "score_ml": None}
        with patch("src.api.feedback_router._SUPABASE_URL", ""), \
             patch("src.api.feedback_router._SUPABASE_KEY", ""):
            res = client.post("/feedback", json=payload)
        assert res.status_code == 204

    def test_all_emotions_accepted(self):
        emotions = ["joy", "sadness", "anger", "fear", "stress", "calm", "tiredness", "pride"]
        for emotion in emotions:
            payload = {**VALID_PAYLOAD, "emotion": emotion}
            with patch("src.api.feedback_router._SUPABASE_URL", ""), \
                 patch("src.api.feedback_router._SUPABASE_KEY", ""):
                res = client.post("/feedback", json=payload)
            assert res.status_code == 204, f"emotion={emotion} should be accepted"


class TestFeedbackNoSupabase:
    """Dégradation gracieuse si Supabase n'est pas configuré."""

    def test_returns_204_when_supabase_not_configured(self):
        with patch("src.api.feedback_router._SUPABASE_URL", ""), \
             patch("src.api.feedback_router._SUPABASE_KEY", ""):
            res = client.post("/feedback", json=VALID_PAYLOAD)
        assert res.status_code == 204

    def test_returns_204_when_httpx_unavailable(self):
        with patch("src.api.feedback_router._SUPABASE_URL", "https://example.supabase.co"), \
             patch("src.api.feedback_router._SUPABASE_KEY", "fake-key"), \
             patch("src.api.feedback_router._HTTPX_AVAILABLE", False):
            res = client.post("/feedback", json=VALID_PAYLOAD)
        assert res.status_code == 204


class TestFeedbackSupabaseIntegration:
    """Appel Supabase mocké."""

    def test_success_returns_204(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.api.feedback_router._SUPABASE_URL", "https://example.supabase.co"), \
             patch("src.api.feedback_router._SUPABASE_KEY", "fake-key"), \
             patch("src.api.feedback_router._HTTPX_AVAILABLE", True), \
             patch("httpx.AsyncClient", return_value=mock_client):
            res = client.post("/feedback", json=VALID_PAYLOAD)

        assert res.status_code == 204

    def test_supabase_http_error_returns_503(self):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)
        )

        with patch("src.api.feedback_router._SUPABASE_URL", "https://example.supabase.co"), \
             patch("src.api.feedback_router._SUPABASE_KEY", "fake-key"), \
             patch("src.api.feedback_router._HTTPX_AVAILABLE", True), \
             patch("httpx.AsyncClient", return_value=mock_client):
            res = client.post("/feedback", json=VALID_PAYLOAD)

        assert res.status_code == 503

    def test_supabase_network_error_returns_503(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch("src.api.feedback_router._SUPABASE_URL", "https://example.supabase.co"), \
             patch("src.api.feedback_router._SUPABASE_KEY", "fake-key"), \
             patch("src.api.feedback_router._HTTPX_AVAILABLE", True), \
             patch("httpx.AsyncClient", return_value=mock_client):
            res = client.post("/feedback", json=VALID_PAYLOAD)

        assert res.status_code == 503
