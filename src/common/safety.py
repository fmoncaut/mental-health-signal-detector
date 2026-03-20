"""
Source de vérité unique pour la détection d'idéation suicidaire.

Utilisé par :
- src/training/predict.py  (inférence ML)
- src/checkin/content.py   (listes de contenu)
- src/checkin/engine.py    (moteur conversationnel)

Toutes les comparaisons se font sur du texte normalisé (minuscules, sans
accents, sans variantes d'apostrophes).  La normalisation est appliquée
une seule fois à l'import sur les mots-clés, et à chaque appel sur le
texte utilisateur via :func:`normalize_text`.
"""

import re
import unicodedata

# Regex compilée une seule fois — couvre toutes les variantes d'apostrophes
_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u201c\u201d'`]")


def normalize_text(text: str) -> str:
    """Normalise un texte pour la détection critique.

    Applique successivement :
    - Mise en minuscules
    - Suppression des diacritiques (NFD + filtrage Mn)
    - Suppression des variantes d'apostrophes (\u2018 \u2019 \u201c \u201d ' `)
      via une regex compilée à l'import (O(n) garanti).

    Args:
        text: Texte brut à normaliser.

    Returns:
        Texte normalisé, sans accents ni apostrophes variantes.
    """
    lower = unicodedata.normalize("NFD", text.lower())
    no_accent = "".join(c for c in lower if unicodedata.category(c) != "Mn")
    return _APOSTROPHE_RE.sub("", no_accent)


# ---------------------------------------------------------------------------
# Mots-clés d'idéation suicidaire — stockés en clair, normalisés à l'import
# ---------------------------------------------------------------------------

CRITICAL_KEYWORDS_FR: list[str] = [
    # Idéation directe
    "je veux mourir", "envie de mourir", "j'ai envie de mourir",
    "je voudrais mourir", "j'aimerais mourir",
    "je veux en finir", "en finir avec tout", "en finir avec la vie",
    "je ne veux plus vivre", "plus envie de vivre", "je veux disparaitre",
    "me suicider", "suicide", "me tuer", "me faire du mal",
    "me supprimer", "me blesser",
    # Pensées de fardeau / d'inutilité
    "je suis un fardeau", "ca serait mieux sans moi", "tout irait mieux sans moi",
    "tout le monde serait mieux sans moi",
    "si je disparaissais personne s'en apercevrait",
    "personne ne remarquerait si je mourais",
    "j'ai besoin de disparaitre",
    # Désespoir
    "plus de raison de vivre", "aucune raison de vivre",
    "je ne sers a rien a personne", "a quoi bon vivre",
    "plus de raison de continuer", "je ne veux plus etre la",
    "je ne veux plus rien",
    # Idéation voilée — sévérité 4 (lexique signaux de crise)
    "je voudrais ne pas me reveiller", "plus envie de me reveiller",
    "j'aimerais ne pas me reveiller", "j'espere ne pas me reveiller",
    "fatigue de vivre", "lasse de vivre", "las de vivre",
    "je ne merite pas de vivre", "je ne merite pas detre la",
    "je suis de trop", "je suis de trop dans ce monde",
    "il ny a plus dissue", "il ny a plus de solution pour moi",
    "personne ne peut maider",
]

CRITICAL_KEYWORDS_EN: list[str] = [
    # Direct ideation
    "i want to die", "want to die", "i want to end it",
    "i dont want to live", "dont want to live", "i want to disappear",
    "kill myself", "i want to kill myself", "want to kill myself",
    "wanna kill myself", "i wanna die", "wanna die",
    # Typo variants — "myself" omitted (e.g. "I want to kill me")
    "i want to kill me", "want to kill me", "wanna kill me", "gonna kill me",
    "suicide", "suicidal", "hurt myself", "want to hurt myself",
    "cut myself", "end my life", "want to end my life",
    "take my life",
    # Burden / hopelessness
    "no reason to live", "no point in living", "no point living",
    "better off without me", "everyone would be better without me",
    "world would be better without me", "no one would miss me",
    "nothing to live for", "cant go on anymore", "cant go on",
    "dont want to be here anymore",
    # Veiled ideation — severity 4 (lexique signaux de crise)
    "wish i was dead", "wish i were dead", "i wish i was dead",
    "tired of living", "tired of being alive",
    "better off dead", "i would be better off dead",
    "dont want to wake up", "hope i dont wake up", "wish i wouldnt wake up",
    "i dont deserve to live", "dont deserve to live",
    "giving up on life", "i give up on life",
    "no hope left", "theres no way out", "there is no way out",
]

# Liste unifiée — source pour les exports vers content.py / predict.py
CRITICAL_KEYWORDS: list[str] = CRITICAL_KEYWORDS_FR + CRITICAL_KEYWORDS_EN

# Pré-normalisés à l'import pour comparaison O(n) sans re-normaliser
_CRITICAL_KEYWORDS_NORMALIZED: list[str] = [
    normalize_text(kw) for kw in CRITICAL_KEYWORDS
]


def check_critical(text: str | None) -> bool:
    """Détecte l'idéation suicidaire par mots-clés, avant tout scoring ML.

    Normalise accents et apostrophes pour couvrir toutes les variantes de
    saisie.  Appelée en priorité absolue, avant NLP et avant scoring emoji.

    Args:
        text: Texte utilisateur brut.  ``None`` ou chaîne vide → ``False``.

    Returns:
        ``True`` si au moins un mot-clé critique est détecté.
    """
    if not text:
        return False
    normalized = normalize_text(text)
    return any(kw in normalized for kw in _CRITICAL_KEYWORDS_NORMALIZED)
