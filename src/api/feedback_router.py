"""Endpoint de collecte anonyme de données utilisateur (opt-in explicite RGPD Art. 9).

Flux :
  1. L'utilisateur coche la case de consentement dans Expression.tsx
  2. Après navigation vers /support, un POST fire-and-forget est envoyé ici
  3. Le texte est transmis tel quel à Supabase (consentement explicite obtenu)
  4. Aucune donnée identifiante n'est transmise (pas d'IP, pas d'userId)

Supabase REST API est appelé via httpx — pas de dépendance ORM lourde.
Variables d'environnement requises :
  - SUPABASE_URL          ex: https://xxxx.supabase.co
  - SUPABASE_SERVICE_KEY  clé service_role (lecture/écriture, garder secrète)
"""

from __future__ import annotations
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from loguru import logger
from src.common.config import get_settings

router = APIRouter(prefix="/feedback", tags=["feedback"])

_TABLE = "anonymous_feedback"
_SUPABASE_DOMAIN = ".supabase.co"


def _is_valid_supabase_url(raw_url: str) -> bool:
    """Validate Supabase endpoint URL used for feedback persistence.

    Accepts only HTTPS URLs targeting a hostname under ``*.supabase.co``.
    Disallows credentials and query fragments to reduce configuration abuse
    and SSRF-style redirection vectors.
    """
    parsed = urlparse(raw_url.strip())
    host = parsed.hostname or ""
    if parsed.scheme != "https":
        return False
    if not host.endswith(_SUPABASE_DOMAIN):
        return False
    if parsed.username or parsed.password:
        return False
    if parsed.fragment or parsed.query:
        return False
    return True


class FeedbackPayload(BaseModel):
    """Données envoyées par le frontend lors d'un consentement opt-in."""

    text: str = Field(..., min_length=1, max_length=5000)
    emotion: str = Field(..., min_length=1, max_length=50)
    distress_level: int = Field(..., ge=0, le=4)
    score_ml: float | None = Field(None, ge=0.0, le=1.0)
    consent: bool

    @field_validator("consent")
    @classmethod
    def consent_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError("consent must be True to store data")
        return v

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        text = v.strip()
        if not text:
            raise ValueError("text cannot be blank")
        return text

    @field_validator("emotion")
    @classmethod
    def emotion_allowlist(cls, v: str) -> str:
        allowed = {"joy", "sadness", "anger", "fear", "stress", "calm", "tiredness", "pride"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"emotion must be one of {allowed}")
        return normalized


@router.post("", status_code=status.HTTP_204_NO_CONTENT)
async def save_feedback(payload: FeedbackPayload) -> None:
    """Persiste un retour anonyme dans Supabase (opt-in explicite uniquement).

    Retourne 204 No Content en cas de succès.
    Retourne 503 si Supabase est indisponible — ne bloque pas l'expérience utilisateur.
    """
    settings = get_settings()
    supabase_url = settings.supabase_url
    supabase_key = settings.supabase_service_key

    if not supabase_url or not supabase_key:
        logger.warning("Supabase non configuré (SUPABASE_URL/SUPABASE_SERVICE_KEY manquants) — feedback ignoré")
        return  # Dégradation gracieuse : pas d'erreur côté utilisateur

    # Validation domaine Supabase — anti-SSRF configuration injection
    if not _is_valid_supabase_url(supabase_url):
        logger.error("SUPABASE_URL invalide — doit se terminer par {}", _SUPABASE_DOMAIN)
        return

    if not _HTTPX_AVAILABLE:
        logger.warning("httpx non installé — feedback ignoré")
        return

    row = {
        "text": payload.text,
        "emotion": payload.emotion,
        "distress_level": payload.distress_level,
        "score_ml": payload.score_ml,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{supabase_url}/rest/v1/{_TABLE}",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=row,
            )
            resp.raise_for_status()
            logger.info("Feedback anonyme enregistré — emotion={} distress_level={}", payload.emotion, payload.distress_level)
    except httpx.HTTPStatusError as exc:
        # Ne pas loguer response.text (peut contenir des infos sensibles)
        logger.error("Supabase HTTP {} — feedback non persisté", exc.response.status_code)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Supabase indisponible") from exc
    except httpx.RequestError as exc:
        logger.error("Supabase connexion échouée — feedback non persisté")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Supabase indisponible") from exc
