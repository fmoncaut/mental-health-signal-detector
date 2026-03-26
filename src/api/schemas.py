from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., min_length=1, max_length=5000, description="Texte à analyser")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        text = v.strip()
        if not text:
            raise ValueError("text cannot be blank")
        return text

    model_type: Literal["baseline", "distilbert", "mental_bert_v3", "mental_roberta"] = "baseline"


class PredictResponse(BaseModel):
    label: int = Field(..., description="0 = non-détresse, 1 = détresse")
    score_distress: float = Field(..., ge=0.0, le=1.0, description="Score de risque entre 0 et 1")
    model: str
    text_preview: str = Field(..., description="Début du texte analysé")
    detected_lang: str = Field(default="en", description="Langue détectée (en/fr)")


class ExplainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., min_length=1, max_length=5000)
    n_features: int = Field(default=15, ge=3, le=30)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        text = v.strip()
        if not text:
            raise ValueError("text cannot be blank")
        return text


class FeatureContribution(BaseModel):
    word: str
    shap_value: float


class ExplainResponse(BaseModel):
    label: int
    score_distress: float
    features: list[FeatureContribution]
    detected_lang: str = "en"


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
