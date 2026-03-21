from functools import lru_cache
from loguru import logger

# Liste blanche explicite — protège contre l'injection de chemins
_ALLOWED_MODEL_TYPES: frozenset[str] = frozenset({"baseline", "distilbert", "mental_bert_v3", "mental_roberta"})


@lru_cache(maxsize=4)
def get_model(model_type: str = "baseline"):
    """Charge chaque type de modèle une seule fois (cache par model_type).

    maxsize=4 pour conserver baseline, distilbert, mental_bert_v3 et mental_roberta simultanément.
    """
    if model_type not in _ALLOWED_MODEL_TYPES:
        raise ValueError(f"model_type invalide : {model_type!r}. Valeurs autorisées : {_ALLOWED_MODEL_TYPES}")
    from src.training.predict import load_model
    logger.info(f"Chargement modèle : {model_type}")
    return load_model(model_type)
