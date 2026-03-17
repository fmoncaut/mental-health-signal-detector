# Roadmap — Compagnon émotionnel intelligent & triage clinique digital

**Principe directeur :** ne jamais remplacer l'humain en cas de risque — toujours orienter.

---

## Architecture globale

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                      │
│  Émotions sélectionnées (multi) + Texte libre               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  PROCESSING — scoringEngine.ts (Phase 2, v2)                │
│                                                             │
│  DistilBERT → score_distress [0–1]                          │
│  + Fix 1 : fallback selfScore + DISTRESS_TEXT_SIGNALS       │
│  + Fix 2 : règle du maximum (mlAdjusted ne peut pas baisser)│
│  + Fix 3 : dimensions cliniques avant guard ML              │
│  + 33 keywords critiques (idéation directe + indirecte)     │
│  + 4 dimensions enrichies (Maslach, triade dépressive…)     │
│  → DiagnosticProfile { distressLevel, clinicalProfile, ... }│
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  SOLUTION ENGINE — solutionEngine.ts / src/solutions/       │
│                                                             │
│  DiagnosticProfile → niveau 0–4                             │
│  → message empathique (kids / adult)                        │
│  → micro-actions (CBT / ACT / mindfulness)                  │
│  → ressources (3114, SAMU, Mon Soutien Psy…)                │
│  → escalade si niveau ≥ 4                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Modèle de triage en 5 niveaux

| Niveau | État clinique | distressLevel | clinicalProfile | Objectif |
|--------|--------------|---------------|-----------------|----------|
| **0** | Bien-être | light | wellbeing | Renforcement positif |
| **1** | Inconfort léger | light | adjustment | Auto-régulation |
| **2** | Détresse modérée | elevated | burnout / anxiety / depression | Structuration |
| **3** | Alerte clinique | critical | depression / burnout | Orientation professionnelle |
| **4** | Urgence critique | critical | crisis | Protection immédiate — 3114 |

---

## Phase 1 — NLP Pipeline ✅ TERMINÉE

- [x] Datasets : Kaggle Reddit + DAIR-AI + GoEmotions (170K exemples)
- [x] Baseline TF-IDF + Logistic Regression (90.9% accuracy)
- [x] DistilBERT fine-tuned (96.8% accuracy)
- [x] API FastAPI (`/predict`, `/explain`, `/health`)
- [x] Dashboard Streamlit + SHAP
- [x] CI GitHub Actions (ruff + pytest)

---

## Phase 2 — Web App React ✅ TERMINÉE

- [x] 6 écrans : Welcome, EmotionSelection, Expression, SupportResponse, Solutions, CheckIn
- [x] Pipeline clinique `scoringEngine.ts` (planchers, masking, dimensions, profils)
- [x] Moteur de recommandation `solutionEngine.ts` (triage 0–4, CBT/ACT/mindfulness)
- [x] Bibliothèque thérapeutique `data/solutions.ts` (20 micro-actions, ressources France)
- [x] 17 keywords critiques → `critical` immédiat

---

## Phase 3 — Backend & Déploiement ✅ TERMINÉE

### Sprint 1 — POST /solutions ✅
- [x] `src/solutions/schemas.py` — `DiagnosticProfileRequest`, `SolutionResponse`
- [x] `src/solutions/data.py` — port Python de `data/solutions.ts`
- [x] `src/solutions/engine.py` — `map_to_triage_level`, `compute_solution`
- [x] `src/api/solutions_router.py` — `POST /solutions` avec sanitisation emotionId

### Sprint 2 — Multi-sélection émotions ✅
- [x] `EmotionSelection.tsx` — multi-select toggle (max 2 kids / 3 adult)
- [x] `CLINICAL_PRIORITY` map — émotion primaire cliniquement dominante
- [x] Propagation `emotionIds[]` + `emotionLabels[]` dans tout le flow
- [x] Badges émotions secondaires dans `Expression.tsx`

### Sprint 3 — CheckIn backend ✅
- [x] `src/checkin/engine.py` — `compute_reminder()` (1h / 4h / tomorrow)
- [x] `src/checkin/schemas.py` — `ReminderRequest`, `ReminderResponse`
- [x] `src/api/checkin_router.py` — `POST /checkin/reminder`, `GET /checkin/reminders`
- [x] `CheckIn.tsx` — appel réel API + `scheduled_label` affiché + fallback gracieux

### Sprint 4 — Tests unitaires frontend ✅
- [x] `vitest.config.ts` — config Vitest (node env, coverage v8)
- [x] `scoringEngine.test.ts` — 46 tests (sanitize, computeFinalScore, dimensions, distressLevel, profil)
- [x] `solutionEngine.test.ts` — 25 tests (triage 0–4, ressources, briques, kids/adult, 3114)
- [x] Scripts `npm test` / `npm run test:coverage`

### Sprint 5 — Déploiement slim ✅
- [x] `Dockerfile.api.slim` — sans torch/transformers/sklearn (~150MB)
- [x] `requirements.slim.txt` — dépendances minimales
- [x] Imports ML conditionnels dans `main.py` (`_ML_AVAILABLE` flag)
- [x] `ALLOWED_ORIGINS` env var (CORS configurable)
- [x] `Dockerfile.frontend` — nginx:alpine multi-stage
- [x] `docker/nginx.conf` — SPA routing + proxy + security headers
- [x] `render.yaml` — one-click deploy
- [x] `vercel.json` — SPA routing
- [x] Déployé : Render (API) + Vercel (frontend) — $0/mois

---

## Phase 4 — Correctifs sécurité clinique ✅ TERMINÉE

### Biais critique corrigé : incohérence émotion/texte

Problème identifié : sélection d'une émotion positive (joy) + texte exprimant une détresse → message positif renvoyé. Risque clinique critique.

- [x] **Fix 1** — Fallback sans ML : `max(emotionFloor, selfScore)` + `DISTRESS_TEXT_SIGNALS` (25 phrases FR/EN)
- [x] **Fix 2** — Règle du maximum : `finalScore = max(blended, floor, mlAdjusted)` ; masking dès `mlScore > 0.25` (+0.20)
- [x] **Fix 3** — Dimensions cliniques avant null-guard ML

### Enrichissement DIMENSION_KEYWORDS (recommandations clinicien)

- [x] **CRITICAL_KEYWORDS** enrichis : 33 keywords (directe + indirecte : fardeau, disparition perçue bénéfique, EN)
- [x] **burnout** : + cynisme/désengagement + inefficacité (modèle Maslach tri-dimensionnel)
- [x] **anxiety** : + anticipation catastrophiste + hypervigilance somatique
- [x] **depression_masked** : + anhédonie + fatigue morale + ralentissement psychomoteur + isolement
- [x] **dysregulation** : + impulsivité/fuite (note : `"j'ai besoin de disparaître"` → CRITICAL)
- [x] **DISTRESS_TEXT_SIGNALS** : 25 signaux généraux pour fallback sans ML

### Disclaimer médical
- [x] Phrase légale ajoutée sur la homepage : _"Cette application ne constitue pas un dispositif médical et ne remplace en aucun cas un avis médical, un diagnostic ou un traitement par un professionnel de santé."_

---

## Phase 5 — Backlog (prochaine itération)

### Amélioration du modèle clinique

- [ ] Détection co-occurrence (patterns multi-dimensions → triage plus précis)
- [ ] Modifiers fréquence/intensité (`"souvent"`, `"tout le temps"`, `"très"`, `"plus du tout"`)
- [ ] Axe temporel : durée des symptômes (`"depuis des semaines"`)
- [ ] Test utilisateurs → recalibration EMOTION_FLOOR et seuils

### Expérience utilisateur

- [ ] Mode hors-ligne (PWA — fonctionnel sans réseau)
- [ ] Historique des check-ins (localStorage ou backend persistant)
- [ ] Notifications réelles via service worker (rappels planifiés)
- [ ] Accessibilité (WCAG 2.1 AA — screen reader, contraste, focus)

### Backend & infrastructure

- [ ] Base de données persistante (sessions anonymisées — opt-in)
- [ ] Rate limiting sur les endpoints (protection abus)
- [ ] Logs structurés + monitoring (Sentry ou Datadog free tier)
- [ ] Tests d'intégration E2E (Playwright)
- [ ] Modèle NLP FR natif (CamemBERT) — remplace traduction FR→EN

### Conformité

- [ ] Analyse RGPD / DPIA (données de santé en clair)
- [ ] Conformité ANSSI / HDS si hébergement données de santé
- [ ] Audit de sécurité externe avant ouverture publique large

---

## Contraintes non-négociables (permanentes)

- Niveau 4 → jamais d'écran vide, toujours le 3114 visible
- Mode enfants → jamais afficher un score numérique de détresse
- L'app ne pose jamais de diagnostic médical — elle oriente
- Tout signal de crise → escalade humaine immédiate
