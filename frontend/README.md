# Frontend — "Comment vas-tu ?" Web App

Application mobile-first React/TypeScript — check-in émotionnel avec pipeline clinique et moteur de recommandation (Phases 2 & 3).

> **Avertissement :** Cette application ne constitue pas un dispositif médical et ne remplace en aucun cas un avis médical, un diagnostic ou un traitement par un professionnel de santé.

---

## Stack

| Technologie | Version |
|-------------|---------|
| Vite | 6.3.5 |
| React | 18 |
| TypeScript | 5.9 |
| Tailwind CSS | v4 (via `@tailwindcss/vite`) |
| motion | v12 (`import from "motion/react"`) |
| React Router | v7 |
| Vitest | latest |
| lucide-react | 0.487 |

---

## Structure

```
frontend/
├── src/
│   ├── screens/
│   │   ├── Welcome.tsx          # Accueil — choix mode Enfant / Adulte
│   │   ├── EmotionSelection.tsx # Multi-sélection (max 2 kids / 3 adult)
│   │   ├── Expression.tsx       # Textarea + POST /predict (distilbert)
│   │   ├── SupportResponse.tsx  # Consomme scoringEngine → DiagnosticProfile
│   │   ├── Solutions.tsx        # Moteur de recommandation (triage 0–4)
│   │   └── CheckIn.tsx          # Planification rappel — POST /checkin/reminder
│   ├── lib/
│   │   ├── scoringEngine.ts     # Pipeline clinique pur — 100% exporté et testé
│   │   ├── solutionEngine.ts    # DiagnosticProfile → SolutionResponse
│   │   └── api.ts               # Client API (VITE_API_URL ou relatif)
│   ├── types/
│   │   ├── diagnostic.ts        # DiagnosticProfile, ClinicalProfile, ClinicalDimension
│   │   └── solutions.ts         # SolutionResponse, MicroAction, Resource, TriageLevel
│   ├── data/
│   │   └── solutions.ts         # Bibliothèque thérapeutique complète
│   ├── __tests__/lib/
│   │   ├── scoringEngine.test.ts  # 46 tests Vitest
│   │   └── solutionEngine.test.ts # 25 tests Vitest
│   ├── components/figma/
│   │   └── ImageWithFallback.tsx
│   ├── App.tsx                  # Mobile frame + RouterProvider
│   ├── routes.ts                # createBrowserRouter — 6 routes
│   └── index.css / tailwind.css / theme.css
├── vercel.json                  # SPA routing Vercel
├── vitest.config.ts             # Config tests (node env, coverage v8)
└── vite.config.ts               # Proxy dev → :8000 + VITE_API_URL
```

---

## Lancement

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
```

Le backend FastAPI doit tourner sur `:8000` :

```bash
TRANSFORMERS_NO_TF=1 uvicorn src.api.main:app --reload --port 8000
```

---

## Tests

```bash
npm test              # vitest run (71 tests)
npm run test:watch    # mode watch
npm run test:coverage # couverture v8
```

---

## Build

```bash
npm run build     # tsc -b && vite build → dist/
```

---

## Workflow utilisateur

```
Welcome
  ↓ Mode Enfant ou Adulte
EmotionSelection
  ↓ Multi-sélection (max 2 kids / 3 adult) + CLINICAL_PRIORITY
Expression
  ↓ Texte libre → POST /predict (distilbert) — spinner + AbortController
SupportResponse   [Pipeline clinique scoringEngine.ts]
  ↓ Fix 1/2/3 → DiagnosticProfile
Solutions         [Moteur de recommandation solutionEngine.ts]
  ↓ SolutionResponse (triage 0–4) + micro-actions + ressources
CheckIn
  ↓ POST /checkin/reminder (1h / 4h / demain) → scheduled_label
```

---

## Pipeline clinique — scoringEngine.ts

### 3 correctifs sécurité (v2) — biais émotion/texte

Le biais identifié : si l'utilisateur sélectionne une émotion positive (joy/calm/pride) mais saisit un texte exprimant une détresse, l'ancienne version retournait un message positif — risque clinique critique.

| Fix | Problème | Solution implémentée |
|-----|----------|---------------------|
| **Fix 1** — Fallback sans ML | Sans modèle : `selfScore` et texte ignorés, tout dépend du plancher émotionnel | Fallback = `max(emotionFloor, selfScore)` + détection `DISTRESS_TEXT_SIGNALS` (25 phrases FR/EN) |
| **Fix 2** — Règle du maximum | L'émotion positive pouvait tirer le score vers le bas malgré un texte alarmant | `finalScore = max(blended, floor, mlAdjusted)` ; masking déclenché dès `mlScore > 0.25` (bonus +0.20) |
| **Fix 3** — Dimensions avant guard ML | Dimensions cliniques ignorées si `mlScore = null` | Vérification dimensions **avant** le null-guard → s'appliquent même sans API |

### Filet de sécurité absolu — 33 keywords critiques

**Idéation directe :** `suicide`, `suicider`, `me tuer`, `mourir`, `plus envie de vivre`, `disparaître`, `en finir`, `me supprimer`, `j'ai envie de mourir`, `pensées suicidaires`, `je veux mourir`, `je n'en peux plus`, `je veux disparaître`, `je ne veux plus être là`, `personne ne m'aime`, `personne ne peut m'aider`, `je ne veux plus vivre`, `j'en ai marre de tout`, `tout seul au monde`

**Idéation indirecte (rec. clinicien) :** `ça serait mieux sans moi`, `tout serait plus simple si je n'étais plus là`, `plus de raison de vivre`, `à quoi bon vivre`, `je suis un fardeau`, `personne ne remarquerait si je disparaissais`

**EN :** `better off without me`, `no reason to live`, `can't go on anymore`

→ Force `critical` immédiatement, indépendamment du score ML.

### Planchers émotionnels recalibrés

| Émotion | Floor | Justification clinique |
|---------|-------|----------------------|
| sadness, fear | **0.35** | Signal majeur — `elevated` minimum garanti |
| anger, tiredness | **0.30** | Dépression masquée, burn-out |
| stress | **0.25** | Anxiété chronique |
| joy, calm, pride | **0.0** | Pas de plancher — texte domine |

### Score final

```
isMasking  = (emotionFloor < 0.2) AND (mlScore > 0.25)
mlAdjusted = mlScore + (isMasking ? 0.20 : 0)
blended    = selfScore ? selfScore × 0.45 + mlAdjusted × 0.55 : mlAdjusted
finalScore = max(blended, emotionFloor, mlAdjusted)   ← règle du maximum
```

### 4 dimensions cliniques (enrichies — recommandations clinicien)

| Dimension | Axes couverts | Mots-clés types |
|-----------|--------------|----------------|
| `burnout` | Épuisement + cynisme/désengagement + inefficacité (Maslach) | `j'en peux plus`, `je m'en fiche de tout`, `je suis dépassé` |
| `anxiety` | Activation physiologique + anticipation + hypervigilance | `boule au ventre`, `je pense au pire`, `je suis tendu` |
| `depression_masked` | Triade : humeur ↓ + énergie ↓ + plaisir ↓ (anhédonie, fatigue morale, ralentissement) | `plus envie de rien`, `tout me coûte`, `je suis inutile` |
| `dysregulation` | Passage à l'acte, auto-agression, impulsivité | `je me fais du mal`, `j'ai envie de tout lâcher` |

Présence d'au moins une dimension → niveau minimum `elevated` (Fix 3 : s'applique sans ML).

### 6 profils cliniques

| Profil | Condition |
|--------|-----------|
| `crisis` | Keywords critiques ou dysrégulation |
| `depression` | sadness/fear + dépression masquée |
| `burnout` | tiredness/anger + dimension burnout |
| `anxiety` | fear/stress + dimension anxiety |
| `adjustment` | Détresse légère sans dimension |
| `wellbeing` | joy/calm/pride + score bas |

---

## Moteur de recommandation — solutionEngine.ts

### Triage 5 niveaux

| Niveau | État | Protocole | Ressources |
|--------|------|-----------|-----------|
| 0 | Bien-être | Renforcement positif, ancrage | — |
| 1 | Inconfort léger | CBT activation comportementale | — |
| 2 | Détresse modérée | CBT/ACT structuration | Mon Soutien Psy, Psycom |
| 3 | Alerte clinique | Orientation professionnelle | Médecin traitant, Psycom |
| 4 | Urgence critique | Escalade immédiate | **3114**, SAMU (15/112), Fil Santé Jeunes |

### Briques thérapeutiques

`cbt_activation` · `cbt_restructuring` · `act` · `mindfulness` · `psychoeducation` · `social_support` · `professional` · `crisis`

### Contraintes non-négociables

- Niveau 4 → 3114 toujours visible, jamais d'écran vide
- Mode enfants → aucun score numérique de détresse affiché
- L'app n'émet jamais de diagnostic médical — elle oriente uniquement

---

## Déploiement (Vercel)

Le frontend est déployé sur Vercel. Les variables d'environnement à configurer :

| Variable | Description |
|----------|-------------|
| `VITE_API_URL` | URL de l'API Render (ex: `https://mental-health-signal-detector.onrender.com`) |

Le `vercel.json` gère le SPA routing (toutes les routes → `index.html`).
