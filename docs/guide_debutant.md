# Guide débutant — Comprendre le Mental Health Signal Detector pas à pas

**Artefact School of Data · Bootcamp Data Science · Mars 2026**

> Ce document explique le projet de A à Z, sans supposer de connaissances avancées.
> Chaque concept est introduit simplement avant d'aller dans le détail technique.

---

## Sommaire

1. [Le problème qu'on résout](#1-le-problème-quon-résout)
2. [La solution en une image](#2-la-solution-en-une-image)
3. [Le parcours utilisateur — ce qui se passe à l'écran](#3-le-parcours-utilisateur)
4. [Les données — d'où vient l'intelligence ?](#4-les-données)
5. [Le machine learning — comment la machine apprend ?](#5-le-machine-learning)
6. [Le backend — le cerveau de l'application](#6-le-backend)
7. [Le frontend — ce que l'utilisateur voit](#7-le-frontend)
8. [Le filet de sécurité — la partie la plus critique](#8-le-filet-de-sécurité)
9. [Le moteur de recommandation](#9-le-moteur-de-recommandation)
10. [Les tests — comment on sait que ça marche ?](#10-les-tests)
11. [Le déploiement — comment ça tourne en production ?](#11-le-déploiement)
12. [Les arbitrages techniques — pourquoi ces choix ?](#12-les-arbitrages-techniques)
13. [Glossaire](#13-glossaire)

---

## 1. Le problème qu'on résout

### Contexte

La santé mentale est une crise silencieuse. En France :
- **1 personne sur 5** souffre d'un trouble mental chaque année
- Moins de **50%** des personnes en détresse consultent un professionnel
- Le délai moyen entre l'apparition des symptômes et une première consultation est de **10 ans**

### Pourquoi ce délai ?

Trois raisons principales :
1. **La honte** — parler de sa santé mentale reste tabou
2. **Le manque d'orientation** — on ne sait pas à qui s'adresser
3. **L'accès aux soins** — délais de rendez-vous, coût, géographie

### Ce qu'on construit

Un **détecteur de signaux de détresse** : une application qui permet à une personne d'exprimer comment elle se sent, puis lui propose une orientation adaptée — de la simple respiration guidée jusqu'à l'appel au 3114 (numéro national de prévention du suicide) en cas de crise.

> **Analogie** : c'est comme un thermomètre émotionnel. Il ne remplace pas le médecin — il aide à comprendre si tu as 37°C ou 40°C et oriente vers la bonne ressource.

### Ce que ce projet N'EST PAS

- Ce n'est pas un outil de diagnostic médical
- Ce n'est pas un chatbot thérapeutique
- Ce n'est pas un remplacement des professionnels de santé

---

## 2. La solution en une image

```
┌─────────────────────────────────────────────────────┐
│                  UTILISATEUR                         │
│  "Je me sens mal... je suis épuisé de tout"         │
└───────────────────────┬─────────────────────────────┘
                        │ tape son ressenti
                        ▼
┌─────────────────────────────────────────────────────┐
│                 APPLICATION WEB                      │
│                                                      │
│  1. Détection de crise (mots-clés critiques)        │
│     → "suicide", "mourir" → appel 3114 immédiat     │
│                                                      │
│  2. Machine Learning (analyse du texte)             │
│     → score de détresse 0.0 → 1.0                  │
│                                                      │
│  3. Fusion avec l'émotion déclarée + questionnaire  │
│     → score final ajusté                            │
│                                                      │
│  4. Moteur de recommandation                        │
│     → niveau 0 (bien-être) à niveau 4 (crise)       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              RESSOURCE PROPOSÉE                      │
│  Niveau 0 : exercice de respiration                 │
│  Niveau 1 : podcast, méditation                     │
│  Niveau 2 : psychologue en ligne                    │
│  Niveau 3 : consultation urgente                    │
│  Niveau 4 : 3114 — tout de suite                   │
└─────────────────────────────────────────────────────┘
```

---

## 3. Le parcours utilisateur

L'application guide l'utilisateur en **6 étapes** :

### Étape 1 — Accueil (Welcome)

L'utilisateur choisit son mode : **adulte** ou **enfant** (mode simplifié avec un langage adapté).

```
[ Mode adulte ]  [ Mode enfant ]
```

### Étape 2 — Sélection d'émotion (EmotionSelection)

8 émotions proposées avec des couleurs et icônes :
`😊 Joyeux · 😌 Calme · 😰 Anxieux · 😢 Triste · 😤 En colère · 😔 Découragé · 😩 Épuisé · 😶 Engourdi`

> **Pourquoi demander l'émotion d'abord ?**
> Parce qu'on sait que les gens sous-estiment ou sur-estiment leur état. Une personne qui dit "je suis joyeux" mais écrit "je veux en finir" envoie un signal contradictoire qui mérite une attention particulière.

### Étape 3 — Questionnaire rapide (QuickCheck)

4 questions inspirées des outils cliniques PHQ-9 et GAD-7 :
- Depuis combien de temps tu te sens ainsi ?
- Quel impact sur ta vie quotidienne ?
- Intensité de 1 à 10 ?
- Tu as eu des pensées difficiles ?

### Étape 4 — Expression libre (Expression)

L'utilisateur écrit librement comment il se sent.

**Ce qui se passe en coulisses :**
- Si le texte est très court (< 30 caractères) → nudge orange : *"Un peu plus de détails nous aide à mieux vous accompagner"*
- Pendant que l'utilisateur écrit, rien n'est envoyé — tout est local dans le navigateur
- À l'envoi : le texte part vers l'API

### Étape 5 — Analyse (Support)

L'application affiche un indicateur animé "Je vous écoute…" pendant le calcul.

### Étape 6 — Résultat (Solutions)

La page de résultat affiche :
- Un message empathique personnalisé (généré par Claude d'Anthropic si disponible)
- Le niveau de triage (0 à 4)
- Des ressources adaptées (numéros, applications, professionnels)
- En mode adulte : un panneau d'analyse avec le score, les dimensions cliniques détectées

---

## 4. Les données

### D'où vient l'intelligence ?

Un modèle de machine learning apprend à partir d'**exemples**. On lui montre des milliers de textes déjà classés par des humains : "ce texte exprime de la détresse" / "ce texte est neutre".

### Les datasets utilisés

| Dataset | Taille | Source | Particularité |
|---|---|---|---|
| **Kaggle Reddit Depression** | 388 000 posts | Reddit (r/depression vs r/casual) | Grand volume, en anglais |
| **DAIR-AI Emotion** | 16 000 phrases | Textes anglais labellisés | 6 émotions |
| **GoEmotions** | 58 000 commentaires | Google / Reddit | 27 émotions fine-grained |
| **eRisk25 (CLEF 2025)** | 75 700 posts | Cliniciens | Labels validés par des experts médicaux |

> **Pourquoi plusieurs datasets ?**
> Un seul dataset peut avoir des biais. Par exemple, Reddit parle principalement anglais et américain. En combinant des sources, on rend le modèle plus robuste.

### Préparation des données — les étapes

```
Données brutes
    │
    ▼
1. Nettoyage
   - Supprimer les URLs, les emojis parasites
   - Normaliser les accents et apostrophes (é→e, l'→l )
   - Mettre en minuscules

    │
    ▼
2. Équilibrage
   - Si 80% des textes sont "pas en détresse" → le modèle apprend à dire "non" par défaut
   - Solution : sous-échantillonner la classe majoritaire

    │
    ▼
3. Tokenisation
   - Découper le texte en "tokens" (morceaux)
   - Exemple : "Je suis épuisé" → ["je", "suis", "ép", "##uisé"]
   - Le modèle travaille avec ces tokens, pas le texte brut

    │
    ▼
4. Encodage
   - Transformer chaque token en un vecteur de nombres
   - "épuisé" → [0.12, -0.34, 0.87, ...] (768 dimensions)
```

---

## 5. Le machine learning

### Intuition simple

Le machine learning, c'est chercher un **pattern** dans des données.

> **Analogie** : imagine que tu dois apprendre à reconnaître si un email est un spam. Tu regardes 10 000 emails. Tu remarques : "spam" apparaît souvent avec "gagner de l'argent" et "cliquez ici". Tu n'as pas eu besoin qu'on te l'explique — tu l'as déduit des exemples.

C'est exactement ce que fait un modèle ML.

### Les trois modèles du projet

#### Modèle 1 — Baseline TF-IDF + Régression Logistique

**C'est quoi TF-IDF ?**
- *TF* (Term Frequency) : combien de fois un mot apparaît dans le texte
- *IDF* (Inverse Document Frequency) : à quel point ce mot est rare dans l'ensemble des textes
- Un mot rare qui apparaît souvent → très informatif

**C'est quoi la Régression Logistique ?**
- Un algorithme qui prédit une probabilité entre 0 et 1
- "Ce texte a 87% de chance d'exprimer de la détresse"

**Résultats :** 78% de précision, répond en < 1 milliseconde

#### Modèle 2 — DistilBERT v2

**C'est quoi BERT ?**
- Un modèle de langage pré-entraîné par Google sur des milliards de textes
- Il comprend le **contexte** : "je suis dans un état" ≠ "l'état français"
- "Distil" = version allégée (40% plus rapide, 60% de la taille)

**Comment on l'a spécialisé (fine-tuning) ?**
- On part d'un modèle qui "comprend" déjà le langage
- On lui ré-apprend une tâche spécifique avec nos données
- Comme un médecin généraliste qui fait une spécialisation en psychiatrie

**Résultats :** 88.4% de précision, F1 Macro 85.9%, entraîné en ~3h sur GPU Colab

```
Texte → Tokenisation → BERT layers (compréhension) → Couche de classification → Score 0.0-1.0
```

#### Modèle 3 — Mental-BERT v3

- Basé sur `mental/mental-bert-base-uncased` (BERT pré-entraîné sur des textes cliniques)
- Entraîné sur 99 000 posts Reddit cliniquement labellisés
- **Résultats :** Accuracy 92.7%, Sensitivité 95.9%, AUC-ROC 98.2%
- Limitation : nécessite un GPU (non déployé en prod, trop coûteux sur free tier)

### Comment on évalue un modèle ?

Trois métriques importantes :

| Métrique | Définition simple | Importance ici |
|---|---|---|
| **Accuracy** | % de bonnes réponses | Vue d'ensemble |
| **Recall (Sensitivité)** | % de vrais positifs détectés | **CRITIQUE** — rater une crise = danger |
| **F1 Macro** | Équilibre précision + recall | Bonne métrique globale |

> **Pourquoi le recall est plus important que l'accuracy ?**
> Dans ce contexte : mieux vaut 10 fausses alertes que de rater 1 vraie crise.

### L'interprétabilité — SHAP

**SHAP** (SHapley Additive exPlanations) permet de répondre à la question : "pourquoi le modèle a dit ça ?"

```
Score de détresse : 0.82
Contribution des mots :
  "épuisé"   → +0.31  ████████
  "plus"     → +0.18  █████
  "envie"    → +0.12  ███
  "joyeux"   → -0.08  ██ (négatif = fait baisser le score)
```

---

## 6. Le backend

### C'est quoi un backend ?

Le backend, c'est la **partie invisible** d'une application. Quand tu cliques sur "Envoyer", ton texte part vers un serveur qui :
1. Analyse le texte avec le modèle ML
2. Calcule un score
3. Renvoie le résultat

### Notre backend : FastAPI (Python)

FastAPI est un framework Python pour créer des **APIs** (interfaces de communication entre le frontend et les modèles).

**Structure du code :**

```
src/
├── api/
│   ├── main.py              ← Point d'entrée — démarre le serveur
│   ├── checkin_router.py    ← Gère les check-ins (parcours utilisateur)
│   ├── analyze_router.py    ← Appelle l'API Claude (Anthropic) pour messages personnalisés
│   ├── feedback_router.py   ← Enregistre les feedbacks utilisateur (Supabase)
│   ├── rate_limit.py        ← Limite les abus (max 10 requêtes/min)
│   └── security_headers.py ← Ajoute les en-têtes de sécurité HTTP
│
├── checkin/
│   ├── engine.py            ← Calcule le score final (fusion ML + émotion + masquage)
│   ├── content.py           ← Textes et ressources (messages, numéros d'urgence)
│   └── schemas.py           ← Structure des données (validation)
│
├── common/
│   └── safety.py            ← SOURCE DE VÉRITÉ des mots-clés critiques
│
└── training/
    └── predict.py           ← Charge et exécute le modèle ML
```

### Comment fonctionne un appel API ?

```
Navigateur                    Serveur FastAPI
    │                               │
    │  POST /checkin/analyze         │
    │  { text: "je suis épuisé",    │
    │    emotion: "sad" }           │
    │ ─────────────────────────────►│
    │                               │ 1. Vérifie les mots-clés critiques
    │                               │ 2. Appelle le modèle ML
    │                               │ 3. Calcule le score final
    │                               │ 4. Choisit les ressources
    │                               │
    │  { distressLevel: 2,          │
    │    score: 0.67,               │
    │    resources: [...] }         │
    │ ◄─────────────────────────────│
```

### La sécurité du backend

**Pourquoi c'est important ?** Une API accessible sur Internet reçoit des attaques automatisées. On a implémenté :

| Mesure | Pourquoi |
|---|---|
| **Rate limiting** | Empêche 10 000 requêtes/seconde d'un bot |
| **CORS strict** | Seul notre frontend peut appeler l'API |
| **Validation des données** | Un hacker ne peut pas envoyer n'importe quoi |
| **Headers de sécurité** | Content-Security-Policy, HSTS, X-Frame-Options |
| **Anti-SSRF** | Empêche l'API de faire des requêtes internes non autorisées |

---

## 7. Le frontend

### C'est quoi un frontend ?

Le frontend, c'est tout ce que l'utilisateur **voit et touche** : les boutons, les couleurs, les animations.

### Notre frontend : React + TypeScript

**React** est une bibliothèque JavaScript pour créer des interfaces. L'idée centrale : l'interface est composée de **composants** réutilisables.

> **Analogie** : React c'est comme des LEGO. Chaque composant est une brique. Tu assembles les briques pour construire l'application.

**TypeScript** = JavaScript + types. Ça permet d'attraper les erreurs avant d'exécuter le code.

**Structure du frontend :**

```
frontend/src/
├── screens/          ← Les 6 écrans (Welcome, EmotionSelection, etc.)
├── lib/
│   ├── scoringEngine.ts    ← Calcul du score (ENTIÈREMENT LOCAL)
│   └── solutionEngine.ts   ← Moteur de recommandation (LOCAL)
├── components/       ← Briques réutilisables (boutons, cartes, etc.)
└── App.tsx           ← Racine de l'application, gère la navigation
```

### Pourquoi le scoring est local ?

Le `scoringEngine.ts` tourne **dans le navigateur**, pas sur le serveur.

**Avantages :**
- Zéro latence (pas de réseau)
- Confidentialité maximale (les données ne quittent pas le navigateur)
- Fonctionne hors ligne

**Comment ça marche ?**

```typescript
// Exemple simplifié de scoringEngine.ts
function computeDistressScore(text, emotion, selfReportScore) {
  // 1. Le ML donne un score brut (reçu de l'API)
  let score = mlScore;

  // 2. On ajuste selon l'émotion déclarée
  if (emotion === "sad") score += 0.15;
  if (emotion === "joy") score -= 0.10;  // mais pas trop : masquage possible

  // 3. On détecte le masquage : joie déclarée + texte très négatif
  if (emotion === "joy" && mlScore > 0.15) {
    score += 0.20;  // signal contradictoire → on remonte le score
  }

  // 4. On ajoute le questionnaire clinique
  score += selfReportScore * 0.3;

  return Math.min(score, 1.0);  // jamais au-dessus de 1.0
}
```

---

## 8. Le filet de sécurité

### La règle absolue

**Avant tout calcul ML, avant toute logique de scoring, il y a un filet de sécurité absolu.**

Si le texte contient un mot-clé critique → réponse CRITIQUE immédiate, peu importe le reste.

```
Texte reçu : "I want to kill me"
    │
    ▼
check_critical("I want to kill me")
    │
    ▼
Normalisation : "i want to kill me" (accents retirés, minuscules)
    │
    ▼
Comparaison avec liste de 50+ mots-clés critiques (FR + EN) :
  ✓ "want to kill me" → MATCH
    │
    ▼
→ Score forcé à 1.0 (CRITIQUE)
→ Niveau 4 → 3114 affiché immédiatement
→ Le ML n'est même pas appelé
```

### Pourquoi ne pas faire confiance qu'au ML ?

Le ML peut se tromper. Sur 100 textes suicidaires, même le meilleur modèle en rate quelques-uns.

Pour ce cas précis, l'erreur est inacceptable. Donc on applique le principe : **règles explicites > probabilités**.

### Le fichier `src/common/safety.py`

C'est la **source de vérité unique** pour tous les mots-clés critiques. Un seul endroit, utilisé partout :

```python
# Exemples de mots-clés FR
CRITICAL_KEYWORDS_FR = [
    "je veux mourir",
    "envie de mourir",
    "je veux en finir",
    "me suicider",
    "je suis un fardeau",
    "voudrais ne pas me reveiller",    # idéation voilée
    "fatigue de vivre",                # idéation voilée
    ...
]

# Exemples EN (avec variantes typo)
CRITICAL_KEYWORDS_EN = [
    "kill myself",
    "want to kill me",      # variante sans "my"
    "wanna kill me",        # variante argot
    "wish i was dead",      # idéation voilée
    "tired of living",      # idéation voilée
    ...
]
```

### La normalisation — pourquoi c'est crucial

```python
def normalize_text(text: str) -> str:
    # "j'ai envie de mourir" → "j ai envie de mourir"
    # "épuisé" → "epuise"
    # "l'envie" → "l envie"
    text = unicodedata.normalize("NFD", text)  # décompose les accents
    text = "".join(c for c in text if not unicodedata.combining(c))  # retire les accents
    text = text.lower()
    text = re.sub(r"['\u2018\u2019\u02bc]", " ", text)  # toutes les variantes d'apostrophe
    return text
```

Sans normalisation, "je veux mourir" et "je veux mourir" (avec accent différent) seraient considérés différents.

---

## 9. Le moteur de recommandation

### Le modèle de soin par paliers (stepped-care NICE)

Le moteur suit le **modèle NICE** (National Institute for Care Excellence, UK), une référence clinique internationale :

```
Niveau 4 — CRISE           → 3114, urgences, contact immédiat
Niveau 3 — SÉVÈRE          → Consultation psychiatrique urgente
Niveau 2 — MODÉRÉ          → Psychologue, plateforme en ligne
Niveau 1 — LÉGER           → Self-help, méditation, applis
Niveau 0 — BIEN-ÊTRE       → Prévention, exercices, podcasts
```

### Comment on passe du score au niveau ?

```
Score 0.0 → 0.2  :  Niveau 0  (bien-être)
Score 0.2 → 0.4  :  Niveau 1  (léger)
Score 0.4 → 0.6  :  Niveau 2  (modéré)
Score 0.6 → 0.8  :  Niveau 3  (sévère)
Score 0.8 → 1.0  :  Niveau 4  (crise)
```

### 80 profils de réponse

Le moteur adapte la réponse selon **3 dimensions** :

- **8 émotions** (joyeux, calme, anxieux, triste, en colère, découragé, épuisé, engourdi)
- **5 niveaux** (0 à 4)
- **2 modes** (adulte / enfant)

= 80 combinaisons de messages possibles, toutes écrites dans `solutionEngine.ts`.

### Les dimensions cliniques

En plus du score global, le système détecte des **dimensions cliniques** dans le texte :

| Dimension | Mots-clés déclencheurs |
|---|---|
| Burnout | "épuisé", "plus d'énergie", "vide", "automne" |
| Anxiété | "peur", "inquiet", "angoisse", "stress" |
| Dépression masquée | "rien ne va", "à quoi bon", "inutile" |
| Dysrégulation | "explosé", "crise", "débordé" |

Ces dimensions permettent d'affiner les ressources proposées.

---

## 10. Les tests

### Pourquoi tester ?

> **Analogie** : avant de mettre en service un pont, on teste sa résistance. Avant de déployer une application médicale, on vérifie que chaque composant fait ce qu'il est censé faire.

### Les 3 types de tests

#### Tests unitaires Python (pytest) — 188 tests

Ils testent une fonction isolée :

```python
def test_check_critical_typo_variant():
    """'I want to kill me' (typo sans 'myself') doit être détecté."""
    assert check_critical("I want to kill me") is True

def test_check_critical_happy_emotion_short_text():
    """Texte court + émotion positive ne doit pas bypasser la détection."""
    assert check_critical("want to kill me") is True
```

#### Tests unitaires JavaScript (Vitest) — 180 tests

Ils testent le scoring dans le navigateur :

```typescript
test("masking detection: joy + high ML score triggers boost", () => {
  const score = computeDistressScore({
    text: "je veux en finir",
    emotion: "joy",
    mlScore: 0.80
  });
  expect(score).toBeGreaterThan(0.80); // le masquage booste le score
});
```

#### Tests end-to-end (Playwright) — 18 tests

Ils simulent un vrai utilisateur dans un vrai navigateur :

```typescript
test("crisis flow: text with critical keyword → level 4 → 3114", async ({ page }) => {
  await page.goto("/");
  await page.click("text=Adulte");
  await page.click("[data-emotion='joy']");   // émotion positive — le masquage compte
  await page.fill("textarea", "I want to kill me");
  await page.click("text=Envoyer");
  await expect(page.locator("text=3114")).toBeVisible();
});
```

**Total : 386 tests** (188 + 180 + 18)

### Le CI/CD — les tests qui tournent automatiquement

À chaque push de code sur GitHub, un **pipeline automatique** lance :

```
Push du code
    │
    ▼
GitHub Actions
    ├── ruff (vérification style Python)
    ├── pytest (188 tests Python)
    ├── pip-audit (vulnérabilités dans les dépendances)
    ├── bandit (failles de sécurité dans le code)
    ├── trivy (vulnérabilités Docker)
    └── gitleaks (secrets/passwords dans le code)
    │
    ▼
✅ Tout vert → déploiement autorisé
❌ Échec → bloqué, email d'alerte
```

---

## 11. Le déploiement

### Architecture de production

```
Utilisateur (smartphone)
        │
        ▼
   Vercel (CDN mondial)           → frontend React (dist/)
        │ requêtes API
        ▼
   Render.com (serveur Docker)    → FastAPI + modèle baseline
        │
        ▼
   HuggingFace Hub                → modèle DistilBERT v2 (stockage)
        │
        ▼
   Anthropic API                  → Claude Haiku (messages personnalisés)
```

### Pourquoi Docker ?

**Docker** crée une "boîte" (conteneur) qui contient l'application et toutes ses dépendances. La boîte fonctionne identiquement sur ton ordinateur, sur le serveur de Render, et sur n'importe quelle machine.

> **Analogie** : c'est comme un appartement meublé. Tu déménages avec tout, rien ne manque, rien n'est incompatible.

### Les 3 variantes Docker

```
Dockerfile.api.slim       → Production actuelle
  - Python + FastAPI + modèle baseline (LR)
  - Taille : ~200MB
  - Coût : gratuit (Render free tier)
  - Vitesse : < 1ms par prédiction

Dockerfile.api.distilbert → Prête à déployer si budget
  - Ajoute PyTorch CPU + DistilBERT
  - Taille : ~2GB
  - Coût : $7/mois (Render Starter)
  - Vitesse : ~3-5s par prédiction (CPU sans GPU)

Dockerfile.api (full)     → Développement local seulement
  - Mental-BERT v3 complet
  - Taille : ~3.2GB
  - Nécessite un GPU
```

---

## 12. Les arbitrages techniques

### Pourquoi pas le meilleur modèle en production ?

Mental-BERT v3 a 92.7% d'accuracy contre 78% pour le baseline. Pourquoi ne pas l'utiliser ?

| Critère | Baseline (LR) | DistilBERT v2 | Mental-BERT v3 |
|---|---|---|---|
| Accuracy | 78% | 88.4% | 92.7% |
| Vitesse | < 1ms | ~3-5s | ~10-20s (GPU requis) |
| Coût infra | Gratuit | $7/mois | $50+/mois |
| Taille | ~5MB | ~268MB | ~500MB+ |

**Décision :** baseline en prod pour démontrer le principe, DistilBERT documenté et prêt à déployer quand le budget le permet.

> **Règle d'or en ML prod :** un bon modèle qui répond en < 1ms vaut mieux qu'un excellent modèle qui répond en 30 secondes.

### Pourquoi le scoring est dans le frontend (pas l'API) ?

Deux raisons :
1. **Latence** : le score est calculé instantanément, sans appel réseau
2. **Confidentialité** : le texte complet n'est pas envoyé si le check de mots-clés le permet

Mais : le ML tourne côté serveur (le modèle ne peut pas tenir dans le navigateur — 268MB).

### Pourquoi Python pour le backend ?

- L'écosystème data science est en Python (scikit-learn, transformers, numpy)
- Les modèles ML sont entraînés en Python → même langage pour la prédiction
- FastAPI est aussi rapide que Node.js pour des APIs

### Pourquoi React pour le frontend ?

- TypeScript permet d'attraper les bugs avant l'exécution
- Composants réutilisables → développement rapide
- Vite = bundler ultra-rapide (start en < 500ms)
- Tailwind CSS = styles utilitaires, cohérence visuelle garantie

### Pourquoi les mots-clés critiques avant le ML ?

C'est une **décision clinique**, pas technique.

Le ML est probabiliste : il peut se tromper. Pour un système de santé mentale, le coût d'une erreur type "faux négatif" (rater une crise) est inacceptable. Les règles explicites sont déterministes : si le mot est là, la réponse est toujours la même.

---

## 13. Glossaire

| Terme | Explication simple |
|---|---|
| **API** | Interface de communication entre deux logiciels (comme un guichet) |
| **Accuracy** | % de prédictions correctes sur toutes les prédictions |
| **Recall / Sensitivité** | % de vrais cas positifs correctement identifiés |
| **F1 Macro** | Moyenne équilibrée entre précision et recall |
| **Token** | Morceau de texte que le modèle traite (souvent < 1 mot) |
| **Fine-tuning** | Spécialiser un modèle pré-entraîné sur une tâche spécifique |
| **Embedding** | Représentation numérique d'un mot ou d'un texte |
| **BERT** | Modèle de langage de Google, comprend le contexte bidirectionnel |
| **TF-IDF** | Méthode statistique pour mesurer l'importance d'un mot dans un texte |
| **Régression logistique** | Algorithme ML qui prédit une probabilité (0 à 1) |
| **SHAP** | Méthode pour expliquer les prédictions d'un modèle ML |
| **Docker** | Technologie pour empaqueter une application et ses dépendances |
| **CI/CD** | Pipeline automatique qui teste et déploie le code à chaque push |
| **Stepped-care** | Modèle clinique d'orientation par paliers de soin |
| **CORS** | Mécanisme de sécurité du navigateur pour les requêtes cross-domaine |
| **Rate limiting** | Limite le nombre de requêtes par minute pour éviter les abus |
| **Masquage** | Phénomène où quelqu'un déclare aller bien mais exprime le contraire |
| **WCAG** | Standard d'accessibilité web (Web Content Accessibility Guidelines) |
| **PHQ-9 / GAD-7** | Questionnaires cliniques validés pour dépression et anxiété |
| **eRisk25** | Dataset de recherche clinique CLEF 2025 (labels validés par cliniciens) |
| **HuggingFace Hub** | Plateforme de partage de modèles ML (comme GitHub pour les modèles) |
| **3114** | Numéro national de prévention du suicide en France |

---

*Document généré le 20 mars 2026 · Mental Health Signal Detector v1.0*
*Artefact School of Data — Fabrice Moncaut*
