# Mental Health Signal Detector — Document de synthèse

**Artefact School of Data · Bootcamp Data Science · Mars 2026**

---

## 1. Contexte et problématique

### Pourquoi ce projet existe

Selon le baromètre Santé publique France 2024, **1 adulte sur 6 a vécu un épisode dépressif** — et **1 sur 2 n'a pas consulté de professionnel**. Les freins principaux : coût, stigmatisation, et méconnaissance des ressources disponibles. Ce projet s'inscrit dans la **Grande Cause Nationale 2025-2026 « Parlons santé mentale ! »**.

### Quels besoins il adresse

- **Dépistage précoce** : identifier les signaux de détresse avant qu'ils s'aggravent
- **Orientation** : guider vers les ressources adaptées (auto-soin, professionnel, urgence)
- **Accessibilité** : outil gratuit, anonyme, utilisable depuis un smartphone

### Dans quel cadre

Grand public (adolescents et adultes), potentiellement intégrable dans un contexte scolaire ou de médecine de premier recours. Ce n'est **pas** un outil de diagnostic médical — il est explicitement conçu comme un premier niveau d'orientation.

---

## 2. Vision globale du projet

### Ce que fait l'application

L'application web **« Comment vas-tu ce matin ? »** guide l'utilisateur en 6 écrans pour évaluer son état émotionnel, analyser son texte libre par intelligence artificielle, et lui proposer des ressources adaptées à son niveau de détresse.

**Version non-tech :** C'est une application mobile qui pose quelques questions simples sur comment tu te sens, analyse ce que tu écris, et te propose ensuite des conseils ou des ressources d'aide — de la respiration guidée jusqu'au numéro d'urgence 3114.

### Parcours utilisateur typique

1. **Accueil** — choix du mode (enfant / adulte)
2. **Sélection d'émotion** — 8 émotions proposées (joie, tristesse, peur, stress, colère, fatigue, calme, fierté)
3. **QuickCheck** *(émotions négatives uniquement)* — 3 micro-questions adaptées, inspirées des échelles cliniques PHQ-9 / GAD-7 : intensité, durée, impact sur le quotidien
4. **Expression libre** — l'utilisateur écrit ce qu'il ressent ; le texte est analysé par le modèle ML
5. **Réponse de soutien** — message empathique + niveau de triage + ressources si nécessaire
6. **Solutions** — micro-actions personnalisées (respiration, restructuration cognitive, orientation professionnelle)

### Valeur ajoutée

Contrairement à un simple questionnaire, l'application **combine trois sources d'information** pour construire un profil clinique : l'émotion choisie, le questionnaire déclaratif, et l'analyse automatique du texte libre. Elle détecte aussi des situations de **masquage** (quand quelqu'un dit aller bien mais écrit le contraire) et des **mots-clés critiques** déclenchant une alerte immédiate vers le 3114.

---

## 3. Architecture fonctionnelle et technique

### Les grands blocs du système

```
[Utilisateur smartphone]
        ↓
[Interface React — Vercel]       ← 6 écrans, modes enfant/adulte
        ↓
[API FastAPI — Render]           ← reçoit le texte, appelle le modèle ML
        ↓
[Modèle ML]                      ← prédit un score de détresse 0→1
        ↓
[Moteur de scoring]              ← fusionne ML + émotion + questionnaire
        ↓
[Claude Haiku (Anthropic)]       ← génère un message empathique personnalisé
        ↓
[Moteur de solutions]            ← sélectionne les ressources selon le triage
```

### Rôle de chaque bloc

| Bloc | Rôle |
|---|---|
| **Frontend React** | Interface utilisateur mobile, 6 écrans, entièrement en français |
| **API FastAPI** | Reçoit les textes, orchestre les appels ML, sécurise les accès |
| **Modèle ML** | Analyse le texte et produit un score de risque (0 = pas de détresse, 1 = détresse élevée) |
| **Moteur de scoring** | Combine le score ML + l'émotion déclarée + le questionnaire en un score final |
| **Claude Haiku** | Rédige un message de soutien personnalisé adapté au profil |
| **Moteur de solutions** | Applique la logique de soins progressifs (stepped-care NICE) pour choisir les ressources |

### Structure du code

```
mental-health-signal-detector/
├── frontend/          Application web React (interface utilisateur)
├── src/api/           API REST FastAPI (point d'entrée backend)
├── src/training/      Entraînement et évaluation des modèles ML
├── src/solutions/     Logique de recommandation de ressources
├── src/common/        Détection de langue, traduction FR→EN, configuration
├── notebooks/         Expériences Jupyter (comparaison modèles, fine-tuning Colab)
├── models/            Modèles entraînés sauvegardés sur disque
└── tests/             315 tests automatisés (Python + TypeScript)
```

---

## 4. Données et modèles de machine learning

**Version non-tech :** Pour apprendre à reconnaître la détresse, le système a été entraîné sur des centaines de milliers de messages réels issus de Reddit et d'études cliniques, avec des étiquettes indiquant si l'auteur traversait ou non une dépression.

### Datasets utilisés

| Source | Volume effectif | Équilibre classes | Type |
|---|---|---|---|
| Kaggle Reddit Depression | 100 000 messages | 80% / 20% | Posts Reddit étiquetés dépression/non-dépression |
| DAIR-AI/emotion | 18 000 phrases | Variable | Phrases avec 6 émotions → transformées en binaire |
| GoEmotions (Google) | 53 000 commentaires | Variable | 28 émotions → transformées en binaire |
| **eRisk25 (CLEF 2025)** | **75 700 posts** | **47% / 53%** | **Données cliniques validées, dépression précoce** |
| **Total entraînement v2** | **246 170 exemples** | 71% / 29% | |

> **Note :** eRisk25 contient 909 sujets (~217K posts bruts), mais seuls 75 700 posts ont un label clinique valide (`target ≠ null`) — ce sont ces données qui ont servi à l'entraînement. C'est la source la plus précieuse : labels validés cliniquement par des psychologues dans le cadre de la compétition CLEF 2025.

### Préparation des données

- Nettoyage : suppression URLs, mentions, caractères spéciaux
- Détection de langue automatique + traduction FR→EN avant analyse
- Équilibrage des classes via **pondération des erreurs** à l'entraînement (class weights : 0.71 pour classe 0, 1.71 pour classe 1) — pénalise davantage les erreurs sur les cas de détresse, cliniquement prioritaires

### Modèles comparés — résultats mesurés

| Modèle | Dataset entraînement | Accuracy | F1 Macro | Recall détresse | Statut |
|---|---|---|---|---|---|
| TF-IDF + LR | Kaggle 254K | 78% | — | 73% | Ancienne prod |
| DistilBERT v1 | Kaggle 100K | 60% | — | 31% | Écarté (distribution shift) |
| **DistilBERT v2** *(prod)* | **Kaggle + DAIR-AI + GoEmotions + eRisk25 (246K)** | **88.4%** | **85.9%** | **~85%\*** | **✅ Déployé** |
| Mental-BERT v3 | — | 92.7% | — | 95.9% | GPU requis |

*\* Recall estimé sur validation set mixte. Seuil de décision ajusté à 0.65 (voir section 6).*

**TF-IDF** (Term Frequency-Inverse Document Frequency) est une méthode statistique qui mesure l'importance des mots. **Transformer / BERT** sont des architectures de deep learning qui comprennent le contexte et les nuances du langage.

---

## 5. Méthodologie d'entraînement et d'évaluation

### Entraînement

- Les données sont divisées en **80% entraînement / 20% test** (le modèle ne voit jamais les données de test)
- DistilBERT est fine-tuné sur Google Colab (GPU T4, ~30-45 min)
- Un mécanisme d'**arrêt précoce** (early stopping) interrompt l'entraînement quand les performances cessent de s'améliorer — évite le surapprentissage

### Métriques clés

- **Recall (Sensitivité)** : capacité à détecter tous les vrais cas de détresse — **métrique prioritaire** en santé mentale (rater un cas est plus grave que faire un faux positif)
- **Precision** : parmi les cas signalés, combien sont réellement en détresse
- **F1 Macro** : équilibre entre les deux classes (détresse / non-détresse)
- **AUC-ROC** : qualité globale du classifieur sur tous les seuils possibles

### Comment on évite le surapprentissage

- Séparation stricte train/test (90% / 10%)
- Early stopping sur F1 Macro (patience = 1 époque)
- `load_best_model_at_end=True` : on conserve le checkpoint de la meilleure époque, pas la dernière
- Pour DistilBERT v2 : la validation loss remonte à l'époque 3 (0.33 → 0.43) → arrêt automatique après l'époque 4. La meilleure époque est l'**époque 3** (checkpoint-20772)

---

## 6. Choix et arbitrages techniques

### Stack choisie

| Composant | Choix | Raison probable |
|---|---|---|
| Backend | Python / FastAPI | Standard data science, performances API |
| Frontend | React + TypeScript + Tailwind | Écosystème mature, mobile-first rapide |
| ML production | TF-IDF + LR | Ultra-léger (989 KB), CPU, gratuit sur Render |
| ML avancé | DistilBERT / Mental-BERT | Meilleure compréhension du langage naturel |
| LLM messages | Claude Haiku | Génération empathique contrôlée, coût faible |
| Déploiement | Render (backend) + Vercel (frontend) | Gratuit / freemium, déploiement simple |

### Arbitrages observables

- **Performance vs coût** : le modèle le plus puissant (Mental-BERT v3, AUC-ROC 98.2%) nécessite un GPU — trop coûteux pour la prod actuelle. DistilBERT v2 tourne sur CPU (~3-5s/requête) et a été retenu comme meilleur compromis (+10 points vs LR, sans GPU).
- **Seuil de décision à 0.65** (au lieu du standard 0.5) : DistilBERT v2 a été entraîné sur un mix où 29% des exemples sont "détresse", mais en prod les textes peuvent avoir une distribution différente (≈20% déprimés sur Kaggle). Le seuil 0.65 corrige la sur-prédiction observée sans sacrifier le Recall cliniquement prioritaire.
- **Explainability vs complexité** : le baseline LR est interprétable (on peut voir quels mots ont pesé) ; DistilBERT est une boîte noire. Un dashboard SHAP existe pour le LR.
- **Vitesse vs richesse** : le moteur de solutions tourne **localement dans le navigateur** (pas d'appel API) pour affichage instantané. L'appel Claude se fait en arrière-plan.
- **Sécurité clinique avant tout** : les mots-clés critiques (idéation suicidaire) déclenchent le niveau 4 **avant** tout scoring ML — la sécurité ne dépend pas du modèle.

---

## 7. Enjeux éthiques, réglementaires et limites

### Risques spécifiques à la santé mentale

- **Faux négatifs** (manquer une détresse réelle) : risque le plus grave → le système impose des planchers de score par émotion et des mots-clés critiques non contournables
- **Faux positifs** (sur-alarmer) : irritant mais moins dangereux → le score ML est masqué en mode enfant et en niveau critique pour éviter l'anxiété
- **Masquage** : une personne peut exprimer une émotion positive tout en écrivant un texte préoccupant → détecté par comparaison émotion / score ML

### Contraintes RGPD

- **Aucune donnée personnelle stockée** côté serveur : les textes ne sont pas loggués (hashés avant toute trace)
- **Données sensibles Art. 9** (`emotion_id`, `distress_level`) non persistées dans la base de données
- `GET /checkin/reminders` supprimé (risque de fuite inter-utilisateurs)
- Historique stocké uniquement en **localStorage** (navigateur local, 30 jours max)
- Feedback micro-actions stocké localement, **aucune transmission serveur**

### Limites actuelles

*Basé sur le code (certain) :*
- Le modèle de production (LR baseline) a un recall de ~73% sur données mixtes — il manque ~1 cas sur 4
- Pas d'authentification : l'application est anonyme, aucune continuité de suivi inter-sessions possible (sauf localStorage)
- Support linguistique limité FR/EN

*Déduit :*
- L'application n'est pas certifiée dispositif médical (CE) — limite son usage clinique officiel
- Le dataset eRisk25 impose des contraintes d'usage (compétition CLEF, accès réglementé)

---

## 8. Roadmap et perspectives

### Ce qui est observable dans le code

- **DistilBERT v2** ✅ déployé en production (Accuracy 88.4%, F1 Macro 85.9%, seuil 0.65) — entraîné sur 246K exemples dont eRisk25 clinique
- **Mental-BERT v3** (Accuracy 92.7%, AUC-ROC 98.2%) prêt mais réservé aux déploiements GPU — prochain candidat production
- Infrastructure Docker complète pour déploiement avec modèles en volume
- Endpoint `/analyze` (Claude Haiku) : fondation pour personnalisation LLM plus poussée

### Évolutions naturelles identifiées

- **Basculer Mental-BERT v3 en production** dès disponibilité d'une instance GPU (Recall 95.9% vs ~85% actuel DistilBERT v2)
- **Suivi longitudinal** enrichi : l'historique 30 jours existe, mais pas encore exploité pour adapter les recommandations dans le temps
- **Intégration clinique** : connecter à un système de référencement vers des professionnels (Mon Soutien Psy, médecin traitant)
- **Certification** : envisager le marquage CE dispositif médical de classe I pour un usage en milieu scolaire ou hospitalier
- **Multilinguisme** étendu : au-delà du FR/EN, notamment pour les populations migrantes

### Posture technique actuelle

**315 tests automatisés** (117 Python + 180 TypeScript + 18 Playwright), CI GitHub Actions, 3 revues de sécurité documentées, conformité WCAG 2.1 AA — base solide pour une montée en charge.

---

*Certitudes : basées sur le code source analysé. Hypothèses signalées par "Déduit". Inconnues : métriques de performance en production réelle, volume d'utilisateurs actifs, validation clinique externe.*

*Document généré le 2026-03-19. Mis à jour le 2026-03-19 : résultats DistilBERT v2 validés, déploiement effectué, seuil prod ajusté à 0.65.*
