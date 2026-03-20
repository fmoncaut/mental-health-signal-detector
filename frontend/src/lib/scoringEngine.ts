/**
 * scoringEngine — Fonctions pures d'analyse clinique
 *
 * Correctifs sécurité (v2) :
 *   Fix 1 — Fallback sans ML : utilise selfScore + DISTRESS_TEXT_SIGNALS
 *   Fix 2 — Règle du maximum : seuil de masquage abaissé à > 0.25 (bonus +0.20)
 *            le texte ne peut jamais abaisser le triage sous son propre signal
 *   Fix 3 — Dimensions cliniques vérifiées avant le null-guard ML
 *            → s'appliquent même si l'API est indisponible
 */

import type { ClinicalDimension, ClinicalProfile, DiagnosticProfile } from "../types/diagnostic";

export type DistressLevel = "light" | "elevated" | "critical";

// ─── Normalisation robuste ────────────────────────────────────────────────────
// Supprime accents, apostrophes variantes (courbe/droite/unicode), casse.
// Garantit que "j'ai envie de mourir" = "jai envie de mourir" = "J'AI ENVIE DE MOURIR".
export function normalizeText(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")   // accents : é→e, ç→c, à→a
    .replace(/[\u2018\u2019\u201c\u201d'`]/g, "")        // apostrophes → supprimées
    .replace(/\s+/g, " ")
    .trim();
}

// ─── Filet de sécurité absolu ────────────────────────────────────────────────
// Stockés normalisés (sans accents ni apostrophes) — comparés via normalizeText().
// Idéation suicidaire explicite + variantes indirectes (fardeau, disparition).
// Toute correspondance → "critical" immédiat, indépendant du score ML.
export const CRITICAL_KEYWORDS = [
  // FR — idéation directe
  "suicide", "suicider", "me suicider", "me tuer", "mourir",
  "plus envie de vivre", "envie de mourir", "jai envie de mourir",
  "pensees suicidaires", "je veux mourir", "je veux en finir",
  "en finir avec tout", "en finir avec la vie", "en finir",
  "me supprimer", "disparaitre", "je veux disparaitre",
  "je ne veux plus etre la", "je ne veux plus vivre",
  "me faire du mal", "je vais me faire du mal",
  // FR — idéation indirecte (fardeau, absence bénéfique)
  "ca serait mieux sans moi", "tout irait mieux sans moi",
  "tout le monde serait mieux sans moi",
  "plus de raison de vivre", "aucune raison de vivre",
  "plus de raison de continuer", "a quoi bon vivre",
  "je suis un fardeau", "je ne sers a rien a personne",
  "si je disparaissais personne sen apercevrait",
  "personne ne remarquerait si je disparaissais",
  "jai besoin de disparaitre",
  // EN — direct suicidal ideation
  "kill myself", "i want to kill myself", "want to kill myself",
  "wanna kill myself", "gonna kill myself",
  "end my life", "want to end my life", "i want to end my life",
  "i want to end it", "want to end it all",
  "take my life", "take my own life",
  "i want to die", "want to die", "i wanna die", "wanna die",
  "hurt myself", "want to hurt myself", "i want to hurt myself",
  "cut myself", "harm myself",
  "suicide", "suicidal",
  // EN — indirect suicidal ideation
  "better off without me", "everyone would be better without me",
  "world would be better without me", "no one would miss me",
  "no reason to live", "nothing to live for",
  "cant go on", "cant go on anymore",
  "no point in living", "no point living",
  "dont want to live", "i dont want to live",
  "dont want to be here anymore",
];

// ─── Seuils de triage clinique ───────────────────────────────────────────────
export const SCORE_CRITICAL = 0.65;
export const SCORE_ELEVATED = 0.35;

// ─── Planchers cliniques recalibrés ─────────────────────────────────────────
export const EMOTION_FLOOR: Record<string, number> = {
  sadness: 0.35, fear: 0.35,
  anger: 0.30, stress: 0.25, tiredness: 0.30,
  joy: 0.0, calm: 0.0, pride: 0.0,
};

// ─── Signaux texte de détresse générale (Fix 1 & 3) ─────────────────────────
// Utilisés dans le fallback (API ML indisponible) pour détecter une incohérence
// entre l'émotion sélectionnée (positive) et le texte saisi (négatif).
// Stockés normalisés (sans accents ni apostrophes) — comparés via normalizeText().
export const DISTRESS_TEXT_SIGNALS = [
  // FR
  "je me sens mal", "ca ne va pas", "je souffre", "jai du mal",
  "tout va mal", "je suis epuise", "je pleure", "je suis triste",
  "jen ai marre", "je suis perdu", "je suis seul", "je me sens seul",
  "jai peur", "je suis anxieux", "je suis stresse", "ca fait mal",
  "je ne suis pas bien", "je vais pas bien", "jai besoin daide",
  // EN
  "i feel bad", "im struggling", "im suffering", "i feel terrible",
  "im not okay", "im not ok",
  "i want to die", "kill myself", "hurt myself", "end my life",
];

// ─── Détection de dimensions cliniques dans le texte ────────────────────────
export const DIMENSION_KEYWORDS: Record<ClinicalDimension, string[]> = {
  // Axe : épuisement motivationnel / énergétique / professionnel
  // Tri-dimensionnel : épuisement + cynisme/désengagement + inefficacité
  burnout: [
    // Épuisement (existant)
    "j'en peux plus", "à bout", "plus la force", "épuisé depuis",
    "plus d'énergie", "je tiens plus", "i can't go on", "burned out",
    "exhausted for weeks",
    // Épuisement complémentaire
    "je suis à plat", "plus aucune motivation", "running on empty",
    "no motivation left", "je suis vidé",
    // Cynisme / désengagement (rec. clinicien — discriminant burnout vs dépression)
    "je m'en fiche de tout", "plus de sens au travail", "je suis désengagé",
    "rien ne sert à rien", "i don't care anymore",
    // Inefficacité (rec. clinicien — 3e dimension Maslach)
    "je suis dépassé", "complètement dépassé", "je suis submergé", "je n'y arrive plus au travail",
    "overwhelmed", "je suis incompétent",
  ],
  // Axe : activation anxieuse — projection future + activation physiologique
  anxiety: [
    // Contrôle / fréquence (existant)
    "je ne contrôle plus", "tout le temps peur", "ça ne s'arrête pas",
    "j'angoisse", "attaque de panique", "je panique", "peur de tout",
    "anxiété", "rumination", "can't stop worrying", "panic attack",
    // Anticipation catastrophiste (rec. clinicien — projection future)
    "je m'inquiète tout le temps", "je pense au pire", "et si ça tourne mal",
    "j'anticipe le pire",
    // Hypervigilance / tension (rec. clinicien)
    "je suis tendu", "je n'arrive pas à me détendre", "always on edge",
    "je suis sur les nerfs",
    // Somatique (rec. clinicien — activation physiologique)
    "je tremble", "boule au ventre", "je n'arrive pas à dormir",
    "i can't sleep", "heart racing", "j'ai du mal à respirer",
  ],
  // Axe : registre cognitif / existentiel — dépression masquée
  // Triade classique : humeur ↓ + énergie ↓ + plaisir ↓
  depression_masked: [
    // Existentiel / sens (existant)
    "à quoi ça sert", "plus de plaisir", "je suis nul", "rien ne va",
    "plus rien ne m'intéresse", "je me sens vide", "sans espoir",
    // Anhédonie (rec. clinicien — plaisir ↓)
    "plus envie de rien", "rien ne me fait plaisir", "plus rien ne m'amuse",
    // Fatigue morale / charge (rec. clinicien — énergie ↓)
    "tout me coûte", "c'est trop lourd", "je suis épuisé moralement",
    // Ralentissement psychomoteur (rec. clinicien)
    "je n'avance pas", "je suis ralenti", "je n'arrive pas à me lever",
    // Isolement / invisibilité (rec. clinicien)
    "je me sens seul", "personne ne comprend", "je suis invisible",
    "je ne sers à rien", "je suis inutile",
    // EN (existant + nouveau)
    "worthless", "hopeless", "nothing matters", "no point",
    "i feel empty", "i feel alone",
  ],
  // Axe : impulsivité / perte de contrôle comportemental
  dysregulation: [
    // Passage à l'acte / auto-agression (existant)
    "j'explose", "je casse tout", "je me suis blessé", "je perds le contrôle",
    "je me fais du mal", "self-harm", "je me coupe", "envie de frapper",
    // Impulsivité / fuite (rec. clinicien)
    "j'ai envie de tout lâcher", "je veux tout casser", "je n'en peux plus de moi",
    "out of control",
  ],
};

// Pré-normalisés à l'import pour comparaison robuste (sans accents ni apostrophes)
const _DIMENSION_KEYWORDS_NORMALIZED: Record<ClinicalDimension, string[]> = Object.fromEntries(
  (Object.keys(DIMENSION_KEYWORDS) as ClinicalDimension[]).map(
    (dim) => [dim, DIMENSION_KEYWORDS[dim].map(normalizeText)]
  )
) as Record<ClinicalDimension, string[]>;

// ─── Sanitisation du score ML ────────────────────────────────────────────────
export function sanitizeMlScore(raw: unknown): number | null {
  if (typeof raw !== "number" || !isFinite(raw)) return null;
  return Math.min(1, Math.max(0, raw));
}

// ─── Détection de dimensions cliniques ──────────────────────────────────────
export function detectClinicalDimensions(text: string): ClinicalDimension[] {
  const normalized = normalizeText(text);
  return (Object.keys(_DIMENSION_KEYWORDS_NORMALIZED) as ClinicalDimension[]).filter(
    (dim) => _DIMENSION_KEYWORDS_NORMALIZED[dim].some((kw) => normalized.includes(kw))
  );
}

// ─── Score final fusionné (Fix 2 — règle du maximum) ────────────────────────
// Seuil de masquage abaissé : mlScore > 0.25 (vs 0.50 avant) pour les émotions
// positives (floor < 0.2 : joy/calm/pride). Bonus porté à +0.20 (vs +0.15).
// Garantit que "joy + texte modérément négatif" n'est jamais sous-estimé.
export function computeFinalScore(
  mlScore: number,
  emotionId: string,
  selfScore: number | null
): number {
  const floor = EMOTION_FLOOR[emotionId] ?? 0.0;

  // Détection de masquage émotion/texte : émotion positive + texte distressant
  const isPositiveEmotion = floor < 0.2;
  const isMasking = isPositiveEmotion && mlScore > 0.25;
  const mlAdjusted = Math.min(1.0, mlScore + (isMasking ? 0.20 : 0));

  const blended = selfScore !== null
    ? selfScore * 0.45 + mlAdjusted * 0.55
    : mlAdjusted;

  // Le texte ne peut pas abaisser le triage en dessous de son propre signal
  return Math.min(1.0, Math.max(blended, floor, mlAdjusted));
}

// ─── Niveau de détresse ──────────────────────────────────────────────────────
export function getDistressLevel(
  mlScore: number | null,
  userText: string,
  emotionId: string,
  dimensions: ClinicalDimension[],
  selfScore: number | null
): DistressLevel {
  const normalizedText = normalizeText(userText);

  // 1. Sécurité absolue : keywords critiques (normalisés — accents + apostrophes)
  if (CRITICAL_KEYWORDS.some((kw) => normalizedText.includes(kw))) return "critical";

  // 2. Dysrégulation → toujours au moins elevated
  if (dimensions.includes("dysregulation")) return "elevated";

  // 3. Fix 3 — Dimensions cliniques : signal indépendant du score ML
  //    S'applique même sans API (slim deployment) — évite l'incohérence
  //    émotion positive + texte angoissant sans modèle disponible.
  if (dimensions.length > 0) return "elevated";

  // 4. Fusion ML + self-report + émotion (Fix 2)
  if (mlScore !== null) {
    const finalScore = computeFinalScore(mlScore, emotionId, selfScore);
    if (finalScore >= SCORE_CRITICAL) return "critical";
    if (finalScore >= SCORE_ELEVATED) return "elevated";
    return "light";
  }

  // 5. Fix 1 — Fallback si API indisponible
  //    Ordre de priorité : selfScore > plancher émotionnel > signaux texte
  const floor = EMOTION_FLOOR[emotionId] ?? 0.0;
  const fallbackScore = selfScore !== null ? Math.max(floor, selfScore) : floor;
  if (fallbackScore >= SCORE_CRITICAL) return "critical";
  if (fallbackScore >= SCORE_ELEVATED) return "elevated";
  // Signaux texte généraux : détecte "joy + je me sens mal" sans ML
  if (DISTRESS_TEXT_SIGNALS.some((kw) => normalizedText.includes(kw))) return "elevated";
  return "light";
}

// ─── Profil clinique synthétique ────────────────────────────────────────────
export function deriveClinicalProfile(
  distressLevel: DistressLevel,
  emotionId: string,
  dimensions: ClinicalDimension[]
): ClinicalProfile {
  if (distressLevel === "critical") return "crisis";
  if (dimensions.includes("dysregulation")) return "crisis";
  if (
    dimensions.includes("burnout") ||
    (emotionId === "tiredness" && dimensions.includes("depression_masked"))
  ) return "burnout";
  if (dimensions.includes("anxiety") || emotionId === "fear") return "anxiety";
  if (dimensions.includes("depression_masked") || emotionId === "sadness") return "depression";
  if (distressLevel === "elevated") return "adjustment";
  return "wellbeing";
}

// ─── Score self-report pondéré ───────────────────────────────────────────────
// Duration et impact/énergie sont de meilleurs prédicteurs cliniques
// que l'intensité seule — ils reflètent la chronicité et l'altération fonctionnelle.
//
// Pondération :  intensity × 1.0 · duration × 1.5 · impact/energy × 1.5
// (aligné sur les critères DSM-5 : persistance + altération fonctionnelle > intensité subjective)
export function computeSelfScore(answers: number[]): number {
  if (!answers || answers.length === 0) return 0;
  if (answers.length === 1) return answers[0] / 3;

  const weights = [1.0, 1.5, 1.5];
  let weighted = 0;
  let maxWeighted = 0;
  for (let i = 0; i < Math.min(answers.length, weights.length); i++) {
    weighted    += answers[i] * weights[i];
    maxWeighted += 3 * weights[i]; // valeur max par réponse = 3
  }
  return Math.min(1.0, weighted / maxWeighted);
}

// ─── Dimensions cliniques depuis le self-report ──────────────────────────────
//
// Les questions QuickCheck codent des signaux cliniques structurés :
//   answers[0] = intensité     (0–3)
//   answers[1] = durée         (0=auj. · 1=qqs jours · 2=>1 sem. · 3=plusieurs sem.)
//   answers[2] = impact quotidien (sadness/fear/stress)
//             OU niveau d'énergie (anger/tiredness)  (0–3)
//
// Logique clinique (références DSM-5 / ICD-11) :
//   durée ≥ 2  = persistance → signal de chronicité (critère A de durée)
//   impact ≥ 2 = altération fonctionnelle → critère B clinique
//   énergie ≥ 2 = épuisement → axe burnout/dépression
export function detectDimensionsFromSelfReport(
  answers: number[],
  emotionId: string,
): ClinicalDimension[] {
  if (!answers || answers.length < 2) return [];

  const dims = new Set<ClinicalDimension>();
  const intensity = answers[0] ?? 0;
  const duration  = answers[1] ?? 0;
  const third     = answers[2] ?? 0;

  const isPersistent   = duration  >= 2; // > 1 semaine
  const isIntense      = intensity >= 2; // fort ou très fort
  const hasThirdSignal = third     >= 2; // impact ou énergie significatif

  switch (emotionId) {
    case "stress":
      // Stress chronique (durée) → burnout ; persistant + intense → anxiété
      if (isPersistent)              dims.add("burnout");
      if (hasThirdSignal)            dims.add("burnout");   // impact fonctionnel
      if (isIntense && isPersistent) dims.add("anxiety");
      break;

    case "tiredness":
      // Fatigue persistante ou intense → burnout
      if (isPersistent || isIntense) dims.add("burnout");
      if (hasThirdSignal)            dims.add("burnout");   // énergie très basse
      // Fatigue multi-semaines + énergie nulle → dépression masquée
      if (duration >= 3 && third >= 2) dims.add("depression_masked");
      break;

    case "sadness":
      // Tristesse persistante → dépression masquée (critère durée DSM-5)
      if (isPersistent)  dims.add("depression_masked");
      // Impact fonctionnel → dépression masquée confirmée
      if (hasThirdSignal) dims.add("depression_masked");
      break;

    case "fear":
      // Peur persistante → anxiété chronique
      if (isPersistent)  dims.add("anxiety");
      // Impact sur quotidien → anxiété clinique (critère B GAD-7)
      if (hasThirdSignal) dims.add("anxiety");
      break;

    case "anger":
      // Colère intense ET persistante → dysrégulation émotionnelle
      if (isIntense && isPersistent)    dims.add("dysregulation");
      // Énergie très basse + persistance → dysrégulation émotionnelle chronique
      if (hasThirdSignal && isPersistent) dims.add("dysregulation");
      break;
  }

  return Array.from(dims);
}

// ─── Pipeline complet ────────────────────────────────────────────────────────
export function buildDiagnosticProfile(params: {
  emotionId: string;
  mode: "kids" | "adult";
  userText: string;
  mlScore: number | null;
  selfScore: number | null;
  selfReportAnswers: number[] | null;
}): DiagnosticProfile {
  const { emotionId, mode, userText, mlScore, selfScore, selfReportAnswers } = params;

  const textDimensions       = detectClinicalDimensions(userText);
  const selfReportDimensions = selfReportAnswers
    ? detectDimensionsFromSelfReport(selfReportAnswers, emotionId)
    : [];
  // Merge dédupliqué — texte + self-report (les deux sources sont complémentaires)
  const clinicalDimensions = [
    ...new Set([...textDimensions, ...selfReportDimensions]),
  ] as ClinicalDimension[];

  const distressLevel  = getDistressLevel(mlScore, userText, emotionId, clinicalDimensions, selfScore);
  const finalScore     = mlScore !== null ? computeFinalScore(mlScore, emotionId, selfScore) : null;
  const clinicalProfile = deriveClinicalProfile(distressLevel, emotionId, clinicalDimensions);

  return {
    emotionId,
    mode,
    userText,
    selfScore,
    selfReportAnswers,
    mlScore,
    finalScore,
    distressLevel,
    clinicalDimensions,
    clinicalProfile,
  };
}
