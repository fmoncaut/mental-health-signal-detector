"""
Router FastAPI — POST /analyze

Génère un message empathique personnalisé via Claude API (claude-haiku-4-5-20251001)
à partir du DiagnosticProfile calculé côté frontend.

Dégradation gracieuse :
- Clé API absente → 503 (le frontend conserve le message local)
- Appel Anthropic échoue → 503 (idem)

Contraintes de sécurité clinique inscrites dans le prompt système :
- Jamais de méthodes ou moyens de se faire du mal
- Jamais de diagnostic médical
- 3114 mentionné discrètement si niveau >= 3
"""

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.api.rate_limit import limiter
from src.solutions.schemas import DiagnosticProfileRequest
from src.common.config import get_settings

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Prompt système — invariant clinique et éthique
_SYSTEM_PROMPT = """Tu es un assistant bienveillant spécialisé en soutien émotionnel, \
intégré dans l'application "Comment vas-tu ?" (Grande Cause Nationale Santé Mentale 2026). \
Tu réponds UNIQUEMENT en français. Tu n'évoques JAMAIS de méthodes ou moyens de se faire du mal. \
Tu ne poses JAMAIS de diagnostic médical. Tu restes chaleureux, empathique et non-alarmiste. \
Tu génères des messages courts (2-3 phrases maximum), fluides et naturels. \
Tu n'utilises jamais de bullet points ni de listes. \
En mode enfant (kids), tu utilises un langage simple, chaleureux et encourageant. \
En mode adulte (adult), tu utilises un langage nuancé et rassurant."""


def _build_user_prompt(profile: DiagnosticProfileRequest) -> str:
    """Construit le prompt utilisateur à partir du profil diagnostique."""

    dim_labels = {
        "burnout": "épuisement",
        "anxiety": "anxiété",
        "depression_masked": "humeur dépressive",
        "dysregulation": "dysrégulation émotionnelle",
    }
    dims_str = (
        ", ".join(dim_labels.get(d, d) for d in profile.clinicalDimensions)
        if profile.clinicalDimensions
        else "aucune dimension spécifique détectée"
    )

    mode_label = "enfant ou adolescent" if profile.mode == "kids" else "adulte"
    score_str = f"{int(profile.finalScore * 100)}%" if profile.finalScore is not None else "non calculé"

    crisis_note = (
        "\nNote de sécurité : le niveau de détresse est élevé (>= 3/4). "
        "Inclure discrètement une mention du 3114 (numéro national prévention suicide, 24h/24, gratuit)."
        if profile.distressLevel in ("critical", "elevated") and profile.clinicalProfile in ("crisis", "depression", "burnout")
        else ""
    )

    return (
        f"Génère un message d'introduction empathique pour une personne qui ressent : "
        f"{profile.emotionId} (profil clinique : {profile.clinicalProfile}).\n"
        f"Mode : {mode_label}.\n"
        f"Niveau de triage : {profile.distressLevel}.\n"
        f"Dimensions cliniques : {dims_str}.\n"
        f"Score de détresse : {score_str}.{crisis_note}\n\n"
        f"Le message doit valider l'émotion sans dramatiser, offrir un signal d'espoir adapté, "
        f"et donner envie de consulter les ressources proposées. "
        f"2-3 phrases maximum, ton {mode_label}."
    )


@router.post("")
@limiter.limit("5/minute")
def analyze_endpoint(request: Request, profile: DiagnosticProfileRequest) -> dict[str, str]:
    """
    Génère un message empathique personnalisé via Claude API.

    - Retourne { message: str } si la clé API est configurée et l'appel réussit.
    - Retourne 503 si la clé est absente ou si l'appel échoue — le frontend conserve le message local.
    """
    settings = get_settings()

    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="Service de personnalisation non configuré (ANTHROPIC_API_KEY manquante).",
        )

    if not _ANTHROPIC_AVAILABLE or anthropic is None:
        logger.warning("Package 'anthropic' absent — endpoint /analyze indisponible.")
        raise HTTPException(status_code=503, detail="Service de personnalisation indisponible.")

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(profile)}],
        )
        text = message.content[0].text.strip() if message.content else ""
        if not text:
            raise ValueError("Réponse Claude vide.")

        logger.info(
            f"[analyze] emotion={profile.emotionId} mode={profile.mode} "
            f"level={profile.distressLevel} tokens={message.usage.output_tokens}"
        )
        return {"message": text}

    except anthropic.AuthenticationError:
        logger.error("[analyze] Clé API Anthropic invalide.")
        raise HTTPException(status_code=503, detail="Clé API invalide — service indisponible.")
    except anthropic.RateLimitError:
        logger.warning("[analyze] Rate limit Anthropic atteint.")
        raise HTTPException(status_code=503, detail="Service temporairement indisponible.")
    except Exception as e:
        logger.error(f"[analyze] Erreur inattendue : {e}")
        raise HTTPException(status_code=503, detail="Service de personnalisation indisponible.")
