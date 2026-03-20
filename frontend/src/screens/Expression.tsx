import { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router";
import { API_BASE, MODEL_TYPE } from "../lib/api";

const VALID_EMOTIONS = new Set(["joy", "sadness", "anger", "fear", "stress", "calm", "tiredness", "pride"]);

// Whitelist des gradients Tailwind autorisés — empêche l'injection CSS via router state
const VALID_EMOTION_COLORS = new Set([
  "from-yellow-300 to-amber-400",
  "from-blue-300 to-blue-400",
  "from-red-300 to-rose-400",
  "from-purple-300 to-violet-400",
  "from-orange-300 to-orange-400",
  "from-emerald-300 to-teal-400",
  "from-slate-300 to-slate-400",
  "from-pink-300 to-pink-400",
]);
import { motion } from "motion/react";
import { Heart, Send, Smile, Frown, Angry, CloudRain, Zap, Cloud, Moon, Trophy, Loader2 } from "lucide-react";

const getEmotionIcon = (emotionId: string) => {
  const icons: Record<string, React.ComponentType<any>> = {
    joy: Smile,
    sadness: Frown,
    anger: Angry,
    fear: CloudRain,
    stress: Zap,
    calm: Cloud,
    tiredness: Moon,
    pride: Trophy,
  };
  const Icon = icons[emotionId] || Smile;
  return <Icon className="w-16 h-16" strokeWidth={1.5} />;
};

export default function Expression() {
  const navigate = useNavigate();
  const location = useLocation();
  const rawEmotionId = location.state?.emotionId;
  const emotionId: string = VALID_EMOTIONS.has(rawEmotionId) ? rawEmotionId : "";
  const emotionLabel: string = typeof location.state?.emotionLabel === "string" ? location.state.emotionLabel : "";
  const rawColor: string = location.state?.emotionColor ?? "";
  const emotionColor: string = VALID_EMOTION_COLORS.has(rawColor) ? rawColor : "";
  const mode: "kids" | "adult" = location.state?.mode === "adult" ? "adult" : "kids";
  const selfScore: number | null = typeof location.state?.selfScore === "number" ? location.state.selfScore : null;
  const selfReportAnswers: number[] | null = Array.isArray(location.state?.selfReportAnswers) ? location.state.selfReportAnswers : null;
  // Émotions secondaires (multi-sélection Palier 2) — optionnelles
  const emotionIds: string[] = Array.isArray(location.state?.emotionIds) ? location.state.emotionIds : [];
  const emotionLabels: string[] = Array.isArray(location.state?.emotionLabels) ? location.state.emotionLabels : [];
  const secondaryLabels = emotionLabels.filter((_, i) => emotionIds[i] !== emotionId);
  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [consent, setConsent] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);

  // Guard : accès direct sans flow → redirection
  useEffect(() => {
    if (!location.state) navigate("/", { replace: true });
  }, [location.state, navigate]);

  // Annule le fetch + marque le composant comme démonté
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  const handleSubmit = async () => {
    if (!text.trim() || isLoading) return;
    setIsLoading(true);

    abortRef.current = new AbortController();

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), model_type: MODEL_TYPE }),
        signal: abortRef.current.signal,
      });

      if (!isMountedRef.current) return;

      if (res.ok) {
        const data = await res.json();
        // Fire-and-forget feedback anonyme si consentement donné
        if (consent) {
          fetch(`${API_BASE}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text: text.trim(),
              emotion: emotionId,
              distress_level: data.distress_level ?? 0,
              score_ml: data.score_distress ?? null,
              consent: true,
            }),
          }).catch(() => {/* silencieux — ne bloque pas l'expérience */});
        }
        navigate("/support", {
          state: {
            emotionId, emotionLabel, emotionColor, emotionIds, emotionLabels,
            userText: text, mode,
            mlScore: data.score_distress, mlLabel: data.label,
            selfScore, selfReportAnswers,
          },
        });
      } else {
        navigate("/support", {
          state: { emotionId, emotionLabel, emotionColor, emotionIds, emotionLabels, userText: text, mode, mlScore: null, selfScore, selfReportAnswers },
        });
      }
    } catch (err) {
      // Abort volontaire (démontage du composant) → ne pas naviguer
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (!isMountedRef.current) return;
      // Erreur réseau → fallback sans ML
      navigate("/support", {
        state: { emotionId, emotionLabel, emotionColor, emotionIds, emotionLabels, userText: text, mode, mlScore: null, selfScore, selfReportAnswers },
      });
    }
  };

  const emotionVerbs: Record<string, { kids: string; adult: string }> = {
    joy: { kids: "super joyeux", adult: "joyeux" },
    sadness: { kids: "un peu triste", adult: "triste" },
    anger: { kids: "fâché", adult: "en colère" },
    fear: { kids: "inquiet", adult: "inquiet" },
    stress: { kids: "stressé", adult: "stressé" },
    calm: { kids: "calme", adult: "calme" },
    tiredness: { kids: "fatigué", adult: "fatigué" },
    pride: { kids: "fier", adult: "fier" },
  };

  if (!location.state) return null;

  const emotionEntry = emotionId ? emotionVerbs[emotionId as keyof typeof emotionVerbs] : undefined;
  const emotionVerb = emotionEntry
    ? emotionEntry[mode as "kids" | "adult"] || emotionLabel?.toLowerCase()
    : "comme ça";

  return (
    <div className="min-h-screen bg-gradient-to-b from-sky-50 via-teal-50 to-peach-50 flex flex-col">
      {/* Header */}
      <div className="px-6 pt-12 pb-4">
        <button
          onClick={() => navigate("/emotions", { state: { mode } })}
          disabled={isLoading}
          className="text-gray-500 mb-6 hover:text-gray-700 transition-colors disabled:opacity-40"
        >
          ← Retour
        </button>

        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          {emotionId && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", delay: 0.2 }}
              className={`inline-block bg-gradient-to-br ${emotionColor} rounded-3xl p-5 mb-6 shadow-xl`}
            >
              <div className="text-white drop-shadow-lg">{getEmotionIcon(emotionId)}</div>
            </motion.div>
          )}

          {secondaryLabels.length > 0 && (
            <div className="flex justify-center gap-2 mb-4 flex-wrap">
              {secondaryLabels.map((lbl) => (
                <span key={lbl} className="text-xs text-gray-500 bg-white/70 rounded-full px-3 py-1 backdrop-blur-sm">
                  {lbl}
                </span>
              ))}
            </div>
          )}

          <h1 className="text-2xl mb-3 px-4" style={{ color: "#2A5F7D" }}>
            {mode === "kids"
              ? `Tu te sens ${emotionVerb} aujourd'hui. Tu veux me dire pourquoi ?`
              : `Vous vous sentez ${emotionVerb} aujourd'hui. Voulez-vous me dire pourquoi ?`}
          </h1>
          <p className="text-gray-500 text-sm px-4">
            {mode === "kids"
              ? "Tu peux écrire ce que tu veux. Même quelques mots."
              : "Prenez le temps d'exprimer ce que vous ressentez."}
          </p>
        </motion.div>
      </div>

      {/* Main Content */}
      <div className="flex-1 px-6 pb-6 flex flex-col">
        {/* Listening / Analysing indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="flex items-center justify-center gap-2 mb-6"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 text-teal-400 animate-spin" />
              <span className="text-sm text-teal-600 font-medium">
                {mode === "kids" ? "J'analyse tes mots..." : "Analyse en cours..."}
              </span>
            </>
          ) : (
            <>
              <div className="flex gap-1.5">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-2.5 h-2.5 bg-teal-400 rounded-full"
                    animate={{ scale: [1, 1.3, 1], opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2, ease: "easeInOut" }}
                  />
                ))}
              </div>
              <span className="text-sm text-gray-500">
                {mode === "kids" ? "J'écoute..." : "Je vous écoute..."}
              </span>
            </>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-6"
        >
          {/* Nudge texte court — au-dessus du champ, visible dès la saisie */}
          <div className="h-8 flex items-center px-1 mb-2">
            {text.length > 0 && text.length < 30 ? (
              <motion.p
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs text-amber-500 font-medium"
              >
                {mode === "kids"
                  ? "Tu peux en dire un peu plus — ça m'aide à mieux te comprendre 🙏"
                  : "Un peu plus de détails nous aide à mieux vous accompagner."}
              </motion.p>
            ) : (
              <span className="text-xs text-gray-400 ml-auto">
                {text.length > 0 && `${text.length} caractères`}
              </span>
            )}
          </div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={isLoading}
            maxLength={5000}
            placeholder={
              mode === "kids"
                ? "Écris ce que tu ressens ici... Ce qui s'est passé, ce qui te tracasse, ou juste ce que tu penses."
                : "Exprimez ce que vous ressentez... Les événements, vos pensées, vos préoccupations."
            }
            className="w-full min-h-[180px] bg-white/80 backdrop-blur-sm rounded-3xl p-6 shadow-lg resize-none focus:outline-none focus:ring-2 focus:ring-teal-300 transition-all text-gray-700 placeholder:text-gray-400 leading-relaxed disabled:opacity-60"
            autoFocus
          />
        </motion.div>

        {/* Consentement RGPD opt-in — visible uniquement si du texte a été saisi */}
        {text.trim().length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-4 p-4 bg-blue-50 rounded-2xl border border-blue-100"
          >
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
                disabled={isLoading}
                className="mt-0.5 w-4 h-4 accent-teal-500 flex-shrink-0"
              />
              <span className="text-xs text-slate-600 leading-relaxed">
                <strong className="text-slate-700">Aider d'autres personnes</strong>
                {" "}— En cochant cette case, vous acceptez que le texte ci-dessus soit conservé
                de façon <strong>anonyme</strong> pour améliorer notre capacité à détecter
                les signaux de détresse et mieux accompagner les personnes qui en ont besoin.
                {" "}<em>Seul le texte saisi sera conservé</em>, sans aucune information permettant de vous identifier.
                Si vous préférez ne pas partager, vos données seront supprimées à la fin de cette session.
              </span>
            </label>
          </motion.div>
        )}

        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          whileHover={{ scale: text.trim() && !isLoading ? 1.02 : 1 }}
          whileTap={{ scale: text.trim() && !isLoading ? 0.98 : 1 }}
          onClick={handleSubmit}
          disabled={!text.trim() || isLoading}
          className={`w-full py-5 rounded-3xl shadow-lg transition-all flex items-center justify-center gap-3 ${
            text.trim() && !isLoading
              ? "bg-gradient-to-r from-teal-400 via-cyan-400 to-blue-400 text-white hover:shadow-xl"
              : "bg-gray-200 text-gray-400 cursor-not-allowed"
          }`}
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
          <span className="font-medium">
            {isLoading
              ? mode === "kids" ? "Analyse en cours..." : "Analyse en cours..."
              : mode === "kids" ? "Partager mes ressentis" : "Envoyer"}
          </span>
        </motion.button>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="mt-6 flex items-center justify-center gap-2 text-sm text-gray-500"
        >
          <Heart className="w-4 h-4 text-pink-400" fill="currentColor" />
          <span>
            {mode === "kids"
              ? "Ton message reste confidentiel et sécurisé"
              : "Votre message reste confidentiel"}
          </span>
        </motion.div>
      </div>

      <div className="h-8"></div>
    </div>
  );
}
