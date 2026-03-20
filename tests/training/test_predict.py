"""
Tests unitaires — src/training/predict.py + src/common/safety.py

Couvre :
- check_critical : détection mots-clés critiques (FR/EN, accents, apostrophes)
- _safe_load_joblib : path traversal bloqué
- load_model : type inconnu → ValueError
"""

import pytest
from pathlib import Path

from src.common.safety import check_critical, normalize_text


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("BONJOUR") == "bonjour"

    def test_accent_removal(self):
        assert normalize_text("épuisé") == "epuise"
        assert normalize_text("à quoi ça sert") == "a quoi ca sert"
        assert normalize_text("disparaître") == "disparaitre"

    def test_apostrophe_variants(self):
        # Apostrophe droite
        assert normalize_text("j'ai") == "jai"
        # Apostrophe courbe unicode \u2019
        assert normalize_text("j\u2019ai") == "jai"
        # Apostrophe ouvrante \u2018
        assert normalize_text("j\u2018ai") == "jai"
        # Backtick
        assert normalize_text("j`ai") == "jai"

    def test_combined(self):
        assert normalize_text("J'AI ENVIE DE MOURIR") == "jai envie de mourir"
        assert normalize_text("À QUOI BON VIVRE") == "a quoi bon vivre"


# ---------------------------------------------------------------------------
# check_critical — filet de sécurité absolu
# ---------------------------------------------------------------------------

class TestCheckCritical:
    # Cas positifs — idéation directe FR
    def test_fr_direct_mourir(self):
        assert check_critical("je veux mourir")

    def test_fr_direct_suicider(self):
        assert check_critical("je veux me suicider")

    def test_fr_direct_tuer(self):
        assert check_critical("j'ai envie de me tuer")

    def test_fr_direct_en_finir(self):
        assert check_critical("je veux en finir avec tout")

    def test_fr_with_accent_apostrophe(self):
        """Accents et apostrophes variantes doivent être normalisés."""
        assert check_critical("j\u2019ai envie de mourir")
        assert check_critical("J'AI ENVIE DE MOURIR")
        assert check_critical("à quoi bon vivre")
        assert check_critical("A quoi bon vivre")

    # Cas positifs — idéation indirecte FR
    def test_fr_indirect_fardeau(self):
        assert check_critical("je suis un fardeau pour tout le monde")

    def test_fr_indirect_disparaitre(self):
        assert check_critical("j'ai besoin de disparaître")

    # Cas positifs — EN direct
    def test_en_kill_myself(self):
        assert check_critical("I want to kill myself")

    def test_en_wanna_die(self):
        assert check_critical("I wanna die")

    def test_en_end_my_life(self):
        assert check_critical("I want to end my life")

    def test_en_suicidal(self):
        assert check_critical("I've been feeling suicidal lately")

    # Cas positifs — EN indirect
    def test_en_no_reason_to_live(self):
        assert check_critical("there's no reason to live anymore")

    def test_en_better_off_without_me(self):
        assert check_critical("everyone would be better off without me")

    # Cas négatifs — texte en détresse mais pas critique
    def test_negative_sadness(self):
        assert not check_critical("je me sens triste aujourd'hui")

    def test_negative_stress(self):
        assert not check_critical("I'm very stressed about work")

    def test_negative_empty(self):
        assert not check_critical("")

    def test_negative_none(self):
        assert not check_critical(None)

    def test_negative_positive_emotion(self):
        assert not check_critical("je suis tellement heureux aujourd'hui")


# ---------------------------------------------------------------------------
# _safe_load_joblib — path traversal
# ---------------------------------------------------------------------------

class TestSafeLoadJoblib:
    def test_path_traversal_blocked(self, tmp_path):
        """Un chemin hors du répertoire models/ doit lever ValueError."""
        from src.training.predict import _safe_load_joblib
        evil_path = tmp_path / "evil.joblib"
        evil_path.write_text("not a real model")
        with pytest.raises(ValueError, match="non autorisé"):
            _safe_load_joblib(evil_path)

    def test_missing_file_raises(self):
        """Un fichier inexistant dans models/ doit lever FileNotFoundError."""
        from src.training.predict import _safe_load_joblib, _MODELS_DIR
        missing = _MODELS_DIR / "does_not_exist_xyz.joblib"
        with pytest.raises(FileNotFoundError):
            _safe_load_joblib(missing)


# ---------------------------------------------------------------------------
# load_model — type inconnu
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_unknown_model_type_raises(self):
        from src.training.predict import load_model
        with pytest.raises(ValueError, match="model_type inconnu"):
            load_model("unknown_model_xyz")
