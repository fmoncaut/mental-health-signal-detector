/**
 * QuickCheck — Micro-questionnaire déclaratif (self-report)
 *
 * 3 questions adaptatives selon l'émotion, inspirées PHQ-9 / GAD-7 / PSS
 * Score normalisé [0–1] → fusionné dans SupportResponse avec le score ML
 *
 * Flow :
 *  - Émotions positives (joy, calm, pride) → redirect immédiat vers /expression
 *  - Émotions négatives → 3 questions one-at-a-time avec auto-avancement
 *  - "Passer" disponible à tout moment → selfScore: null
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { computeSelfScore } from "../lib/scoringEngine";

// ─── Émotions positives — bypass direct ──────────────────────────────────────

const POSITIVE_EMOTIONS = new Set(["joy", "calm", "pride"]);

// ─── Définition des questions ────────────────────────────────────────────────

interface QuickQuestion {
  id: string;
  kids: string;
  adult: string;
  options: { value: number; kids: string; adult: string }[];
}

const Q_INTENSITY: QuickQuestion = {
  id: "intensity",
  kids: "À quel point tu te sens comme ça ?",
  adult: "À quelle intensité ressentez-vous cela ?",
  options: [
    { value: 0, kids: "Un peu",     adult: "Légèrement" },
    { value: 1, kids: "Assez",      adult: "Modérément" },
    { value: 2, kids: "Beaucoup",   adult: "Fortement" },
    { value: 3, kids: "Très fort",  adult: "Très fortement" },
  ],
};

const Q_DURATION: QuickQuestion = {
  id: "duration",
  kids: "Depuis combien de temps ?",
  adult: "Depuis combien de temps ?",
  options: [
    { value: 0, kids: "Aujourd'hui",         adult: "Aujourd'hui" },
    { value: 1, kids: "Quelques jours",       adult: "Quelques jours" },
    { value: 2, kids: "Plus d'une semaine",   adult: "Plus d'une semaine" },
    { value: 3, kids: "Depuis longtemps",     adult: "Plusieurs semaines" },
  ],
};

const Q_IMPACT: QuickQuestion = {
  id: "impact",
  kids: "Est-ce que ça t'empêche de faire tes choses habituelles ?",
  adult: "Est-ce que cela impacte votre quotidien ?",
  options: [
    { value: 0, kids: "Pas vraiment",    adult: "Pas vraiment" },
    { value: 1, kids: "Un peu",          adult: "Un peu" },
    { value: 2, kids: "Oui, assez",      adult: "Assez souvent" },
    { value: 3, kids: "Oui, beaucoup",   adult: "Beaucoup" },
  ],
};

const Q_ENERGY: QuickQuestion = {
  id: "energy",
  kids: "Ton énergie est comment en ce moment ?",
  adult: "Comment est votre niveau d'énergie ?",
  options: [
    { value: 0, kids: "Normale",      adult: "Normale" },
    { value: 1, kids: "Un peu basse", adult: "Un peu basse" },
    { value: 2, kids: "Fatigué",      adult: "Fatigué(e)" },
    { value: 3, kids: "Épuisé",       adult: "Épuisé(e)" },
  ],
};

// ─── Mapping émotion → questions ─────────────────────────────────────────────

const EMOTION_QUESTIONS: Record<string, QuickQuestion[]> = {
  sadness:   [Q_INTENSITY, Q_DURATION, Q_IMPACT],
  fear:      [Q_INTENSITY, Q_DURATION, Q_IMPACT],
  stress:    [Q_INTENSITY, Q_DURATION, Q_IMPACT],
  anger:     [Q_INTENSITY, Q_DURATION, Q_ENERGY],
  tiredness: [Q_INTENSITY, Q_DURATION, Q_ENERGY],
};

// ─── Couleurs d'accent par émotion ───────────────────────────────────────────

const EMOTION_ACCENT: Record<string, string> = {
  sadness:   "from-blue-300 to-blue-400",
  fear:      "from-purple-300 to-violet-400",
  stress:    "from-orange-300 to-orange-400",
  anger:     "from-red-300 to-rose-400",
  tiredness: "from-slate-300 to-slate-400",
};

// ─── Composant principal ──────────────────────────────────────────────────────

export default function QuickCheck() {
  const navigate = useNavigate();
  const location = useLocation();
  const isMountedRef = useRef(true);

  const emotionId: string = location.state?.emotionId ?? "";
  const emotionLabel: string = location.state?.emotionLabel ?? "";
  const emotionColor: string = location.state?.emotionColor ?? "";
  const mode: "kids" | "adult" = location.state?.mode === "adult" ? "adult" : "kids";
  // Émotions secondaires (multi-sélection Palier 2) — optionnelles
  const emotionIds: string[] = Array.isArray(location.state?.emotionIds) ? location.state.emotionIds : [];
  const emotionLabels: string[] = Array.isArray(location.state?.emotionLabels) ? location.state.emotionLabels : [];

  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<number[]>([]);
  const [direction, setDirection] = useState(1); // 1 = forward

  // Guard
  useEffect(() => {
    if (!location.state) navigate("/", { replace: true });
  }, [location.state, navigate]);

  // Cleanup
  useEffect(() => {
    isMountedRef.current = true;
    return () => { isMountedRef.current = false; };
  }, []);

  // Émotions positives → bypass immédiat
  useEffect(() => {
    if (POSITIVE_EMOTIONS.has(emotionId)) {
      navigate("/expression", {
        replace: true,
        state: { emotionId, emotionLabel, emotionColor, emotionIds, emotionLabels, mode, selfScore: null, selfReportAnswers: null },
      });
    }
  }, [emotionId, emotionLabel, emotionColor, emotionIds, emotionLabels, mode, navigate]);

  if (!location.state || POSITIVE_EMOTIONS.has(emotionId)) return null;

  const questions = EMOTION_QUESTIONS[emotionId] ?? [Q_INTENSITY, Q_DURATION, Q_IMPACT];
  const total = questions.length;
  const question = questions[currentIndex];
  const accent = EMOTION_ACCENT[emotionId] ?? "from-teal-300 to-cyan-400";

  const goToExpression = (allAnswers: number[]) => {
    if (!isMountedRef.current) return;
    // Pondération durée × 1.5 + impact/énergie × 1.5 > intensité × 1.0
    // → persistance et altération fonctionnelle pèsent plus que l'intensité subjective
    const selfScore = allAnswers.length > 0 ? computeSelfScore(allAnswers) : null;
    navigate("/expression", {
      state: { emotionId, emotionLabel, emotionColor, emotionIds, emotionLabels, mode, selfScore, selfReportAnswers: allAnswers },
    });
  };

  const handleAnswer = (value: number) => {
    const newAnswers = [...answers, value];
    if (currentIndex + 1 < total) {
      setAnswers(newAnswers);
      setDirection(1);
      setCurrentIndex(currentIndex + 1);
    } else {
      goToExpression(newAnswers);
    }
  };

  const handleSkip = () => goToExpression([]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-sky-50 via-teal-50 to-slate-50 flex flex-col">

      {/* Header */}
      <div className="px-6 pt-12 pb-2">
        <button
          onClick={() => navigate(-1)}
          className="text-gray-400 hover:text-gray-600 transition-colors mb-4"
        >
          ← Retour
        </button>

        {/* Barre de progression */}
        <div className="flex gap-1.5 mb-6">
          {questions.map((_, i) => (
            <div
              key={i}
              className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                i < currentIndex ? `bg-gradient-to-r ${accent}` :
                i === currentIndex ? `bg-gradient-to-r ${accent} opacity-60` :
                "bg-gray-200"
              }`}
            />
          ))}
        </div>

        {/* Émotion(s) sélectionnée(s) */}
        <div className="flex items-center gap-2 mb-6 flex-wrap">
          <div className={`w-6 h-6 rounded-full bg-gradient-to-br ${accent} flex-shrink-0`} />
          <span className="text-sm text-gray-500">
            {mode === "kids" ? "Tu te sens " : "Vous vous sentez "}
            <span className="font-medium text-gray-700">{emotionLabel?.toLowerCase()}</span>
            {emotionLabels.length > 1 && emotionLabels
              .filter((_, i) => emotionIds[i] !== emotionId)
              .map((lbl) => (
                <span key={lbl} className="text-gray-500"> et <span className="font-medium text-gray-700">{lbl.toLowerCase()}</span></span>
              ))
            }
          </span>
        </div>
      </div>

      {/* Question */}
      <div className="flex-1 px-6 flex flex-col justify-between pb-10">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={currentIndex}
            initial={{ opacity: 0, x: direction * 32 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -direction * 32 }}
            transition={{ duration: 0.22 }}
            className="space-y-4"
          >
            {/* Numéro */}
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Question {currentIndex + 1} / {total}
            </p>

            {/* Texte de la question */}
            <h2 className="text-xl font-medium leading-snug" style={{ color: "#2A5F7D" }}>
              {mode === "kids" ? question.kids : question.adult}
            </h2>

            {/* Options */}
            <div className="space-y-2 pt-2">
              {question.options.map((opt) => (
                <motion.button
                  key={opt.value}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handleAnswer(opt.value)}
                  className="w-full text-left bg-white/80 backdrop-blur-sm rounded-2xl px-5 py-4 shadow-sm border border-white hover:shadow-md hover:border-teal-200 transition-all flex items-center justify-between group"
                >
                  <span className="text-gray-700 text-sm font-medium">
                    {mode === "kids" ? opt.kids : opt.adult}
                  </span>
                  <div className={`w-4 h-4 rounded-full border-2 border-gray-300 group-hover:border-teal-400 group-hover:bg-teal-400 transition-all flex-shrink-0`} />
                </motion.button>
              ))}
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Passer */}
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          onClick={handleSkip}
          className="text-gray-400 text-sm hover:text-gray-600 transition-colors text-center pt-6"
        >
          {mode === "kids" ? "Passer cette étape" : "Passer cette étape"}
        </motion.button>
      </div>
    </div>
  );
}
