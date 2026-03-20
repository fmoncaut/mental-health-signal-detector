"""Tests unitaires — POST /feedback (collecte anonyme opt-in)."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.common.config import Settings

client = TestClient(app)

VALID_PAYLOAD = {
    "text": "Je me sens très fatigué et triste depuis plusieurs semaines.",
    "emotion": "sadness",
    "distress_level": 2,
    "score_ml": 0.72,
    "consent": True,
}

_SETTINGS_NO_SUPABASE = Settings(supabase_url="", supabase_service_key="")
_SETTINGS_WITH_SUPABASE = Settings(supabase_url="https://example.supabase.co", supabase_service_key="fake-key")


class TestSupabaseUrlValidation:
    def test_accepts_https_supabase_url(self):
        from src.api.feedback_router import _is_valid_supabase_url

        assert _is_valid_supabase_url("https://example.supabase.co") is True

    def test_rejects_http_scheme(self):
        from src.api.feedback_router import _is_valid_supabase_url

        assert _is_valid_supabase_url("http://example.supabase.co") is False

    def test_rejects_url_with_credentials(self):
        from src.api.feedback_router import _is_valid_supabase_url

        assert _is_valid_supabase_url("https://user:pass@example.supabase.co") is False

    def test_rejects_url_with_query_or_fragment(self):
        from src.api.feedback_router import _is_valid_supabase_url

        assert _is_valid_supabase_url("https://example.supabase.co?x=1") is False
        assert _is_valid_supabase_url("https://example.supabase.co#frag") is False


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

    def test_text_whitespace_only_rejected(self):
        payload = {**VALID_PAYLOAD, "text": "   \n\t  "}
        res = client.post("/feedback", json=payload)
        assert res.status_code == 422

    def test_invalid_emotion_rejected(self):
        payload = {**VALID_PAYLOAD, "emotion": "rage"}
        res = client.post("/feedback", json=payload)
        assert res.status_code == 422

    def test_emotion_is_normalized(self):
        payload = {**VALID_PAYLOAD, "emotion": "  Sadness  "}
        with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_NO_SUPABASE):
            res = client.post("/feedback", json=payload)
        assert res.status_code == 204

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
        with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_NO_SUPABASE):
            res = client.post("/feedback", json=payload)
        assert res.status_code == 204

    def test_all_emotions_accepted(self):
        emotions = ["joy", "sadness", "anger", "fear", "stress", "calm", "tiredness", "pride"]
        for emotion in emotions:
            payload = {**VALID_PAYLOAD, "emotion": emotion}
            with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_NO_SUPABASE):
                res = client.post("/feedback", json=payload)
            assert res.status_code == 204, f"emotion={emotion} should be accepted"


class TestFeedbackNoSupabase:
    """Dégradation gracieuse si Supabase n'est pas configuré."""

    def test_returns_204_when_supabase_not_configured(self):
        with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_NO_SUPABASE):
            res = client.post("/feedback", json=VALID_PAYLOAD)
        assert res.status_code == 204

    def test_returns_204_when_httpx_unavailable(self):
        with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_WITH_SUPABASE), \
             patch("src.api.feedback_router._HTTPX_AVAILABLE", False):
            res = client.post("/feedback", json=VALID_PAYLOAD)
        assert res.status_code == 204

    def test_returns_204_when_supabase_url_is_insecure(self):
        bad_settings = Settings(supabase_url="http://example.supabase.co", supabase_service_key="fake-key")
        with patch("src.api.feedback_router.get_settings", return_value=bad_settings):
            res = client.post("/feedback", json=VALID_PAYLOAD)
        assert res.status_code == 204

    def test_returns_204_when_supabase_url_contains_query(self):
        bad_settings = Settings(
            supabase_url="https://example.supabase.co?redirect=https://evil.example",
            supabase_service_key="fake-key",
        )
        with patch("src.api.feedback_router.get_settings", return_value=bad_settings):
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

        with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_WITH_SUPABASE), \
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

        with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_WITH_SUPABASE), \
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

        with patch("src.api.feedback_router.get_settings", return_value=_SETTINGS_WITH_SUPABASE), \
             patch("src.api.feedback_router._HTTPX_AVAILABLE", True), \
             patch("httpx.AsyncClient", return_value=mock_client):
            res = client.post("/feedback", json=VALID_PAYLOAD)

        assert res.status_code == 503
