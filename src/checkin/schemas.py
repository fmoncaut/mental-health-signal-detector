from typing import Literal

from pydantic import BaseModel, Field


class CheckInRequest(BaseModel):
    emoji: str | None = Field(
        default=None,
        pattern=r"^(😄|🙂|😐|😔|😢)$",
        description="Emoji choisi parmi 😄🙂😐😔😢",
    )
    text: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description="Texte libre optionnel",
    )
    step: int = Field(default=1, ge=1, le=2, description="1=check-in initial, 2=réponse à la question de suivi")


class ResourceItem(BaseModel):
    title: str
    description: str
    action: str
    url: str | None = None


class CheckInResponse(BaseModel):
    level: str = Field(..., description="green | yellow | red")
    score: float = Field(..., ge=0.0, le=1.0)
    greeting: str
    message: str
    tip: str | None = None
    follow_up: str | None = Field(default=None, description="Question de suivi si pertinent")
    resources: list[ResourceItem] = Field(default_factory=list)
    detected_lang: str = "fr"


# ─── Rappels de suivi ─────────────────────────────────────────────────────────

class ReminderRequest(BaseModel):
    offset: Literal["1h", "4h", "tomorrow"] = Field(
        ..., description="Délai avant le rappel : 1 heure, 4 heures, ou demain matin"
    )
    mode: Literal["kids", "adult"] = "adult"
    emotion_id: str | None = Field(default=None, description="Émotion principale du flow")
    distress_level: str | None = Field(default=None, description="Niveau de détresse (light/elevated/critical)")


class ReminderResponse(BaseModel):
    id: str = Field(..., description="Identifiant unique du rappel")
    offset: str
    scheduled_at: str = Field(..., description="Heure programmée (ISO 8601)")
    scheduled_label: str = Field(..., description="Libellé lisible, ex. 'aujourd\\'hui à 15h30'")
    message: str = Field(..., description="Message de confirmation adapté au mode")
