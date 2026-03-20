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

import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from loguru import logger

router = APIRouter(prefix="/feedback", tags=["feedback"])

_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
_TABLE = "anonymous_feedback"


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

    @field_validator("emotion")
    @classmethod
    def emotion_allowlist(cls, v: str) -> str:
        allowed = {"joy", "sadness", "anger", "fear", "stress", "calm", "tiredness", "pride"}
        if v not in allowed:
            raise ValueError(f"emotion must be one of {allowed}")
        return v


@router.post("", status_code=status.HTTP_204_NO_CONTENT)
async def save_feedback(payload: FeedbackPayload) -> None:
    """Persiste un retour anonyme dans Supabase (opt-in explicite uniquement).

    Retourne 204 No Content en cas de succès.
    Retourne 503 si Supabase est indisponible — ne bloque pas l'expérience utilisateur.
    """
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        logger.warning("Supabase non configuré (SUPABASE_URL/SUPABASE_SERVICE_KEY manquants) — feedback ignoré")
        return  # Dégradation gracieuse : pas d'erreur côté utilisateur

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
                f"{_SUPABASE_URL}/rest/v1/{_TABLE}",
                headers={
                    "apikey": _SUPABASE_KEY,
                    "Authorization": f"Bearer {_SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=row,
            )
            resp.raise_for_status()
            logger.info("Feedback anonyme enregistré — emotion={} distress_level={}", payload.emotion, payload.distress_level)
    except httpx.HTTPStatusError as exc:
        logger.error("Supabase HTTP {} — {}", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Supabase indisponible")
    except httpx.RequestError as exc:
        logger.error("Supabase connexion échouée — {}", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Supabase indisponible")
