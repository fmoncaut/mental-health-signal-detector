# Mental Health Signal Detector

Système de détection de signaux de détresse mentale — projet final Artefact School of Data.

Combine un pipeline NLP (TF-IDF + DistilBERT), une API REST FastAPI et une web app mobile-first React pour orienter les utilisateurs en détresse vers des ressources adaptées.

> **Avertissement clinique :** Cette application ne constitue pas un dispositif médical et ne remplace en aucun cas un avis médical, un diagnostic ou un traitement par un professionnel de santé.

---

## Table des matières

- [Architecture](#architecture)
- [Datasets](#datasets)
- [Modèles](#modèles)
- [Performances](#performances)
- [Installation](#installation)
- [Entraînement](#entraînement)
- [API](#api)
- [Web App](#web-app)
- [Dashboard](#dashboard)
- [Docker & Déploiement](#docker--déploiement)
- [Tests & CI](#tests--ci)
- [Sécurité](#sécurité)
- [Structure du projet](#structure-du-projet)

---

## Architecture

```
Texte (FR ou EN)
    │
    ▼
Détection langue (langdetect)
    │
    ▼ si FR
Traduction FR→EN (deep-translator, timeout 5s)
    │
    ▼
Nettoyage texte (clean_text)
    │
    ├──► Baseline : TF-IDF + Logistic Regression  ──► score_distress [0–1]
    │         └──► Explication SHAP (contributions par mot)
    │
    └──► Avancé   : DistilBERT fine-tuned          ──► score_distress [0–1]
                          │
                          ▼
                 Pipeline clinique (scoringEngine.ts)
                          │
                ┌─────────┴──────────┐
                │                    │
         emotionFloor          DIMENSION_KEYWORDS
         (plancher/émotion)    (burnout/anxiety/
                                depression_masked/
                                dysregulation)
                │                    │
                └─────────┬──────────┘
                          │
                   DiagnosticProfile
                          │
                          ▼
                  Solution Engine (triage 0–4)
                          │
                   SolutionResponse
              (message + micro-actions + ressources)
```

| Couche | Technologie |
|--------|-------------|
| ML / NLP | scikit-learn, HuggingFace Transformers |
| Sérialisation modèle | joblib (format compact — artefacts ML internes) |
| API REST | FastAPI + Uvicorn + CORS middleware |
| Solution engine | `src/solutions/` — triage 0–4, ressources France |
| Check-in backend | `src/checkin/` — rappels, planification |
| Web App | React 18 + TypeScript + Tailwind v4 + Vite |
| Dashboard | Streamlit + visualisation SHAP |
| Traduction | deep-translator + timeout via `concurrent.futures` |
| Infrastructure | Docker multi-stage + GitHub Actions CI |
| Déploiement | Render (API slim) + Vercel (frontend) |

---

## Datasets

Trois sources publiques combinées — **170 000 exemples** après nettoyage :

| Dataset | Source | Taille | Label détresse |
|---------|--------|--------|---------------|
| Reddit Depression | [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) | 2,47M posts → 100K sous-échantillonnés | `label=1` |
| DAIR-AI/emotion | [HuggingFace](https://huggingface.co/datasets/dair-ai/emotion) | 18K phrases | sadness + fear → 1 |
| GoEmotions (Google) | [HuggingFace](https://huggingface.co/datasets/google-research-datasets/go_emotions) | 54K commentaires | sadness, grief, fear, nervousness… → 1 |

**Distribution finale :** 69% non-détresse / 31% détresse

---

## Modèles

### Baseline — TF-IDF + Logistic Regression

- TF-IDF : 50 000 features, unigrammes + bigrammes, `sublinear_tf=True`
- Logistic Regression : `C=1.0`, `class_weight="balanced"`, `max_iter=1000`
- Sérialisé via **joblib** dans `models/baseline.joblib`

### Avancé — DistilBERT fine-tuned

- Fine-tuning de `distilbert-base-uncased` sur HuggingFace Trainer
- 3 epochs, batch 16, `max_length=128` — Colab T4 GPU (~8 min)
- Sauvegardé dans `models/fine_tuned/` (format HuggingFace safetensors)

> Fine-tuning CPU estimé à ~36h — utiliser `notebooks/distilbert_finetune_colab.ipynb`

---

## Performances

| Modèle | Dataset | Accuracy | F1 weighted | F1 macro |
|--------|---------|----------|-------------|----------|
| Baseline TF-IDF+LR | 170K exemples | **90.9%** | **91.0%** | **89.4%** |
| DistilBERT fine-tuned | DAIR-AI (16K) | **96.8%** | — | — |

---

## Installation

**Prérequis :** Python 3.11+

```bash
git clone https://github.com/fmoncaut/mental-health-signal-detector.git
cd mental-health-signal-detector

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.example .env
```

---

## Entraînement

```bash
# Baseline (CPU, ~2 min)
python -m src.training.train --model baseline

# Full (Kaggle + DAIR-AI + GoEmotions — recommandé)
python -m src.training.train --model baseline \
    --kaggle-path data/raw/reddit_depression_dataset.csv \
    --go-emotions \
    --kaggle-samples 100000
```

---

## API

### Démarrage

```bash
TRANSFORMERS_NO_TF=1 uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

#### `GET /health`
```json
{"status": "ok", "model_loaded": true}
```

#### `POST /predict`

Prédit le niveau de détresse d'un texte (FR ou EN). Retourne `503` si le modèle est absent (déploiement slim).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Je me sens tellement triste", "model_type": "baseline"}'
```

#### `POST /checkin`

Check-in conversationnel — combine emoji de humeur + texte libre.

```bash
curl -X POST http://localhost:8000/checkin \
  -H "Content-Type: application/json" \
  -d '{"emoji": "😔", "text": "Fatigué depuis quelques jours", "step": 1}'
```

**Plancher de sécurité :**

| Emoji | Niveau minimum |
|-------|---------------|
| 😢 | `red` — toujours |
| 😔 | `yellow` — minimum garanti |

#### `POST /checkin/reminder`

Planifie un rappel (1h / 4h / demain). Stockage en mémoire (`deque`, maxlen=1000).

```bash
curl -X POST http://localhost:8000/checkin/reminder \
  -H "Content-Type: application/json" \
  -d '{"offset": "1h", "mode": "adult"}'
```

#### `POST /solutions`

Calcule une `SolutionResponse` depuis un `DiagnosticProfile` — triage 0–4, micro-actions, ressources.

```bash
curl -X POST http://localhost:8000/solutions \
  -H "Content-Type: application/json" \
  -d '{"emotionId": "sadness", "mode": "adult", "userText": "...", "distressLevel": "elevated", "clinicalProfile": "depression", "clinicalDimensions": []}'
```

---

## Web App

Application mobile-first "Comment vas-tu ?" — interface de check-in émotionnel avec moteur de recommandation clinique.

### Démo en ligne

- **Frontend** : [https://mental-health-signal-detector.vercel.app](https://mental-health-signal-detector.vercel.app)
- **API** : [https://mental-health-signal-detector.onrender.com](https://mental-health-signal-detector.onrender.com)

> L'instance Render free tier se met en veille après inactivité (cold start ~50s).

### Lancement local

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
```

### Workflow utilisateur

```
Welcome → EmotionSelection → Expression → SupportResponse → Solutions → CheckIn
```

### Pipeline clinique (scoringEngine.ts)

**3 correctifs sécurité (v2) :**

| Fix | Problème corrigé | Solution |
|-----|-----------------|----------|
| Fix 1 — Fallback | Sans ML : selfScore et texte ignorés | Fallback utilise `max(emotionFloor, selfScore)` + `DISTRESS_TEXT_SIGNALS` |
| Fix 2 — Règle du max | Émotion positive tire le score vers le bas | `finalScore = max(blended, floor, mlAdjusted)` ; masking déclenché dès mlScore > 0.25 (+0.20) |
| Fix 3 — Dimensions avant guard | Dimensions cliniques ignorées sans ML | Vérification dimensions **avant** le null-guard `mlScore` |

**Filet de sécurité absolu — 33 keywords critiques :**

Idéation directe (`suicide`, `me tuer`, `je veux mourir`…) + idéation indirecte (`ça serait mieux sans moi`, `plus de raison de vivre`, `je suis un fardeau`…) + EN (`better off without me`, `no reason to live`…).

→ Force `critical` immédiatement, sans tenir compte du score ML.

**4 dimensions cliniques (enrichies — recommandations clinicien) :**

| Dimension | Axes couverts |
|-----------|--------------|
| `burnout` | Épuisement + cynisme/désengagement + inefficacité (Maslach tri-dimensionnel) |
| `anxiety` | Activation physiologique + anticipation catastrophiste + hypervigilance |
| `depression_masked` | Triade dépressive (humeur ↓ + énergie ↓ + plaisir ↓) + isolement + cognitions négatives |
| `dysregulation` | Passage à l'acte, auto-agression, perte de contrôle comportemental |

**6 profils cliniques :** `crisis` · `depression` · `burnout` · `anxiety` · `adjustment` · `wellbeing`

### Moteur de recommandation clinique (Solutions)

| Niveau | État | Protocole |
|--------|------|-----------|
| 0 | Bien-être | Renforcement positif, ancrage |
| 1 | Inconfort léger | Auto-régulation (CBT activation) |
| 2 | Détresse modérée | Structuration + soutien (CBT/ACT) |
| 3 | Alerte clinique | Orientation professionnelle |
| 4 | Urgence critique | 3114 + SAMU — escalade immédiate |

**Contraintes non-négociables :**
- Niveau 4 → 3114 toujours visible, jamais d'écran vide
- Mode enfants → aucun score numérique affiché
- L'app n'émet jamais de diagnostic médical

Voir [frontend/README.md](frontend/README.md) pour la documentation complète.

---

## Dashboard

```bash
streamlit run src/dashboard/app.py
# http://localhost:8501
```

Score de risque, label, langue détectée, graphique SHAP horizontal (baseline).

---

## Docker & Déploiement

### Local (full stack avec ML)

```bash
cd docker/
docker-compose up --build
```

| Service | Dockerfile | Port |
|---------|-----------|------|
| `api` | `Dockerfile.api` | 8000 |
| `frontend` | `Dockerfile.frontend` | 3000 |
| `dashboard` | `Dockerfile.dashboard` | 8501 |

### Slim (sans modèle ML — déploiement free tier)

```bash
docker build -f docker/Dockerfile.api.slim -t mhsd-api-slim .
docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://... mhsd-api-slim
```

- Pas de `torch` / `transformers` / `sklearn` (~150MB vs ~2.6GB)
- `/predict` et `/explain` retournent `503` gracieusement
- `/solutions`, `/checkin`, `/checkin/reminder`, `/health` : 100% fonctionnels

### Render + Vercel (recommandé — $0/mois)

**API sur Render :**
- Language : Docker
- Dockerfile : `docker/Dockerfile.api.slim`
- Env vars : `ENV=production`, `ALLOWED_ORIGINS=https://your-app.vercel.app`

**Frontend sur Vercel :**
- Root Directory : `mental-health-signal-detector/frontend`
- Env vars : `VITE_API_URL=https://your-api.onrender.com`

---

## Tests & CI

```bash
pip install -r requirements-dev.txt
ruff check src/ tests/
pytest tests/ --cov=src --cov-report=term-missing
```

**Backend (78 tests) :**

| Fichier | Couverture |
|---------|-----------|
| `tests/api/test_health.py` | `/health`, `/predict` sans modèle |
| `tests/api/test_checkin.py` | `/checkin` : 422, emoji, NLP fallback, planchers sécurité |
| `tests/api/test_reminder.py` | `/checkin/reminder` : validation, scheduling, UUID, labels FR |
| `tests/api/test_solutions.py` | `/solutions` : triage 0–4, ressources, sanitisation, 3114 |
| `tests/checkin/test_engine.py` | Score, niveaux, planchers 😢→RED 😔→YELLOW |
| `tests/training/test_train.py` | Preprocessing, entraînement, scoring |

**Frontend (71 tests — Vitest) :**

| Fichier | Couverture |
|---------|-----------|
| `scoringEngine.test.ts` | `sanitizeMlScore`, `computeFinalScore`, `detectClinicalDimensions`, `getDistressLevel` (Fix 1/2/3), `deriveClinicalProfile` |
| `solutionEngine.test.ts` | Triage 0–4, ressources, briques thérapeutiques, mode kids/adult, 3114 |

```bash
cd frontend
npm test              # vitest run
npm run test:coverage # couverture v8
```

**CI GitHub Actions** — déclenché sur push `main`, `Fabrice`, `Stan`, `Thomas`, `aimen` :
```
ruff → pytest --cov → vitest (à venir)
```

---

## Sécurité

| Mesure | Détail |
|--------|--------|
| **CORS** | `ALLOWED_ORIGINS` env var (comma-separated) — `*` en dev, origines restreintes en prod |
| **Sérialisation** | joblib pour artefacts ML internes — non exposé au réseau |
| **Erreurs API** | Messages génériques côté client, détails loggés côté serveur uniquement |
| **Données sensibles** | Textes jamais loggués (longueur uniquement) |
| **Traduction externe** | Timeout 5s via `concurrent.futures` + fallback silencieux |
| **Validation URL** | `API_URL` normalisée + validée regex (protection SSRF) |
| **Docker** | Conteneurs non-root (`appuser`, uid=1000) |
| **Secrets** | `.env` dans `.gitignore` — jamais commité |

---

## Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `ENV` | `development` \| `production` | `development` |
| `LOG_LEVEL` | Niveau de log | `INFO` |
| `ALLOWED_ORIGINS` | Origines CORS autorisées (comma-separated) | `*` |
| `MODEL_NAME` | Modèle HuggingFace de base | `distilbert-base-uncased` |
| `MODEL_PATH` | Chemin DistilBERT fine-tuned | `./models/fine_tuned` |

---

## Structure du projet

```
mental-health-signal-detector/
├── frontend/                     # Web app React (Vite + React 18 + Tailwind v4)
│   ├── src/
│   │   ├── screens/              # Welcome, EmotionSelection, Expression,
│   │   │                         # SupportResponse, Solutions, CheckIn
│   │   ├── lib/
│   │   │   ├── scoringEngine.ts  # Pipeline clinique pur (100% testé)
│   │   │   ├── solutionEngine.ts # Moteur recommandation (triage 0–4)
│   │   │   └── api.ts            # Client API avec VITE_API_URL
│   │   ├── types/
│   │   │   ├── diagnostic.ts     # DiagnosticProfile, ClinicalProfile…
│   │   │   └── solutions.ts      # SolutionResponse, MicroAction, Resource…
│   │   ├── data/
│   │   │   └── solutions.ts      # Bibliothèque thérapeutique complète
│   │   └── __tests__/lib/        # 71 tests Vitest (scoringEngine + solutionEngine)
│   ├── vercel.json               # SPA routing Vercel
│   ├── vitest.config.ts          # Config tests frontend
│   └── vite.config.ts            # Proxy dev → :8000
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI, CORS, imports ML conditionnels
│   │   ├── checkin_router.py     # /checkin + /checkin/reminder
│   │   ├── solutions_router.py   # POST /solutions
│   │   └── dependencies.py       # get_model() avec lru_cache
│   ├── checkin/
│   │   ├── engine.py             # compute_score, get_level, compute_reminder
│   │   ├── content.py            # Réponses/tips/ressources VERT/JAUNE/ROUGE
│   │   └── schemas.py            # CheckInRequest/Response, ReminderRequest/Response
│   ├── solutions/
│   │   ├── engine.py             # map_to_triage_level, compute_solution
│   │   ├── data.py               # Port Python de solutions.ts
│   │   └── schemas.py            # DiagnosticProfileRequest, SolutionResponse
│   ├── common/
│   │   ├── config.py             # Settings pydantic-settings (+ allowed_origins)
│   │   ├── language.py           # Détection langue + traduction FR→EN
│   │   └── logging.py            # Loguru
│   └── training/
│       ├── preprocess.py         # Chargement datasets
│       ├── train.py              # Baseline + DistilBERT
│       ├── evaluate.py           # Métriques + SHAP
│       └── predict.py            # Inférence
├── docker/
│   ├── Dockerfile.api            # Full (avec ML)
│   ├── Dockerfile.api.slim       # Slim sans ML (~150MB) — Render free tier
│   ├── Dockerfile.frontend       # nginx:alpine multi-stage
│   ├── nginx.conf                # SPA routing + proxy API + security headers
│   ├── docker-compose.yml        # api + frontend + dashboard
│   └── requirements.slim.txt     # Sans torch/transformers/sklearn
├── tests/
│   ├── api/                      # test_health, test_checkin, test_reminder, test_solutions
│   ├── checkin/                  # test_engine
│   └── training/                 # test_train
├── docs/
│   └── phase3-roadmap.md         # Roadmap phases 1–3 + Sprint 4
├── .env.example
└── render.yaml                   # One-click deploy Render
```

---

## Auteurs

Projet réalisé dans le cadre du bootcamp **Data Science** à l'[Artefact School of Data](https://www.artefact.com/data-consulting-transformation/artefact-school-of-data/).

- Fabrice Moncaut
- Stanislas Grinchenko
- Thomas
- Aimen
