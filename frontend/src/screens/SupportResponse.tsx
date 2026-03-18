import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router";
import { motion } from "motion/react";
import {
  Heart,
  Phone,
  MessageCircle,
  Sparkles,
  Shield,
  Smile,
  Frown,
  Angry,
  CloudRain,
  Zap,
  Cloud,
  Moon,
  Trophy,
  ArrowRight,
} from "lucide-react";
import type { DiagnosticProfile } from "../types/diagnostic";
import {
  sanitizeMlScore,
  detectClinicalDimensions,
  detectDimensionsFromSelfReport,
  computeFinalScore,
  getDistressLevel,
  deriveClinicalProfile,
} from "../lib/scoringEngine";

const getEmotionIcon = (emotionId: string, size: string = "w-16 h-16") => {
  const icons: Record<string, React.ComponentType<any>> = {
    joy: Smile, sadness: Frown, anger: Angry, fear: CloudRain,
    stress: Zap, calm: Cloud, tiredness: Moon, pride: Trophy,
  };
  const Icon = icons[emotionId] || Smile;
  return <Icon className={size} strokeWidth={1.5} />;
};

const VALID_EMOTIONS = new Set(["joy", "sadness", "anger", "fear", "stress", "calm", "tiredness", "pride"]);

export default function SupportResponse() {
  const navigate = useNavigate();
  const location = useLocation();

  // Valeurs brutes depuis le router state
  const rawEmotionId = location.state?.emotionId;
  const rawMode = location.state?.mode;

  // Validation et sanitisation
  const emotionId: string = VALID_EMOTIONS.has(rawEmotionId) ? rawEmotionId : "";
  const emotionColor: string = location.state?.emotionColor ?? "";
  const userText: string = typeof location.state?.userText === "string" ? location.state.userText : "";
  const mode: "kids" | "adult" = rawMode === "adult" ? "adult" : "kids";
  const mlScore: number | null = sanitizeMlScore(location.state?.mlScore);
  // Self-report (QuickCheck) — null si émotion positive ou étape passée
  const selfScore: number | null = typeof location.state?.selfScore === "number"
    ? Math.min(1, Math.max(0, location.state.selfScore))
    : null;
  const selfReportAnswers: number[] | null = Array.isArray(location.state?.selfReportAnswers)
    ? location.state.selfReportAnswers
    : null;

  // Guard : accès direct sans flow → redirection vers l'accueil
  useEffect(() => {
    if (!location.state) {
      navigate("/", { replace: true });
    }
  }, [location.state, navigate]);

  if (!location.state) return null;

  // ─── Pipeline d'analyse clinique ────────────────────────────────────────
  const textDimensions       = detectClinicalDimensions(userText);
  const selfReportDimensions = selfReportAnswers
    ? detectDimensionsFromSelfReport(selfReportAnswers, emotionId)
    : [];
  // Merge dédupliqué — self-report enrichit les dimensions sans écraser le texte
  const clinicalDimensions = [
    ...new Set([...textDimensions, ...selfReportDimensions]),
  ] as typeof textDimensions;
  const distressLevel = getDistressLevel(mlScore, userText, emotionId, clinicalDimensions, selfScore);
  const finalScore = mlScore !== null ? computeFinalScore(mlScore, emotionId, selfScore) : null;
  const clinicalProfile = deriveClinicalProfile(distressLevel, emotionId, clinicalDimensions);

  // DiagnosticProfile complet — passé à Phase 3 via le router state
  const diagnosticProfile: DiagnosticProfile = {
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

  const getSupportMessage = () => {
    if (distressLevel === "critical") {
      return mode === "kids"
        ? "Je vois que tu traverses un moment vraiment difficile. Tu es très courageux d'en parler. Ces sentiments sont lourds à porter, mais tu n'es pas seul. Il y a des personnes qui peuvent t'aider et qui sont là pour toi, 24h/24."
        : "Je comprends que vous traversez un moment particulièrement difficile. Il faut beaucoup de courage pour exprimer ce que vous ressentez. Vous n'êtes pas seul(e), et il existe des personnes formées qui peuvent vous accompagner immédiatement.";
    }

    if (distressLevel === "elevated") {
      switch (emotionId) {
        case "sadness":
          return mode === "kids"
            ? "Je comprends que tu te sentes triste. C'est difficile d'avoir le cœur lourd. Ces émotions sont normales et elles vont passer. Tu as le droit de te sentir comme ça, et c'est bien d'en parler."
            : "La tristesse que vous ressentez est légitime. Il est important de reconnaître ces émotions difficiles. N'hésitez pas à parler à quelqu'un de confiance ou à un professionnel si ces sentiments persistent.";
        case "fear":
          return mode === "kids"
            ? "C'est normal d'avoir peur parfois. Tu n'es pas seul avec cette inquiétude. Parler de ce qui te fait peur peut aider à se sentir mieux. Tu peux aussi en parler avec quelqu'un en qui tu as confiance."
            : "L'inquiétude que vous ressentez mérite d'être entendue. Il est important de ne pas rester seul(e) avec ces préoccupations. Parler à quelqu'un peut vraiment aider.";
        default:
          return mode === "kids"
            ? "Ce que tu ressens est important et valide. Merci de m'avoir fait confiance. Si ça continue ou si ça devient trop difficile, n'hésite pas à en parler à un adulte de confiance."
            : "Vos émotions sont légitimes et méritent d'être exprimées. Si vous sentez que cela devient difficile à gérer, n'hésitez pas à chercher du soutien auprès de proches ou d'un professionnel.";
      }
    }

    // light
    switch (emotionId) {
      case "joy":
        return mode === "kids"
          ? "C'est génial que tu te sentes joyeux ! Ces moments de bonheur sont précieux. Continue à faire ce qui te rend heureux, et n'hésite pas à partager ta joie avec les personnes que tu aimes."
          : "C'est merveilleux de vous sentir joyeux ! Ces moments positifs sont précieux. Continuez à cultiver ce qui vous fait du bien.";
      case "calm":
        return mode === "kids"
          ? "Bravo pour ce moment de calme ! C'est important de se sentir zen. Continue à prendre soin de toi et à trouver ces moments de paix."
          : "C'est excellent de ressentir cette sérénité. Profitez de ce moment de calme intérieur, c'est un cadeau précieux.";
      case "pride":
        return mode === "kids"
          ? "Tu as raison d'être fier ! C'est super d'être content de soi. Continue comme ça, tu fais de belles choses !"
          : "Votre fierté est justifiée ! C'est important de reconnaître vos réussites et d'être bienveillant envers vous-même.";
      case "anger":
        return mode === "kids"
          ? "La colère est une émotion normale. C'est bien de la reconnaître. Prends quelques grandes respirations, et si tu as besoin d'en parler avec quelqu'un, n'hésite pas."
          : "La colère est une émotion légitime. Reconnaître ce que vous ressentez est important. Prenez le temps de respirer et d'identifier ce qui a provoqué cette émotion.";
      case "stress":
        return mode === "kids"
          ? "Le stress, c'est quand on se sent un peu sous pression. C'est normal d'en ressentir. Essaie de respirer calmement, et rappelle-toi que tu fais de ton mieux."
          : "Le stress fait partie de la vie. Prenez un moment pour respirer profondément. Vous faites de votre mieux, et c'est suffisant.";
      case "tiredness":
        return mode === "kids"
          ? "Tu as l'air fatigué. C'est important de bien se reposer. Prends soin de toi, et n'oublie pas que ton corps a besoin de repos pour être en forme !"
          : "La fatigue est le signe que votre corps et votre esprit ont besoin de repos. Prenez soin de vous et accordez-vous du temps pour récupérer.";
      default:
        return mode === "kids"
          ? "Merci d'avoir partagé ce que tu ressens. C'est courageux de parler de ses émotions. N'oublie pas que tu peux toujours en parler à quelqu'un de confiance."
          : "Merci d'avoir exprimé ce que vous ressentez. Vos émotions sont valides et importantes.";
    }
  };

  return (
    <div
      className={`min-h-screen ${
        distressLevel === "critical"
          ? "bg-gradient-to-b from-pink-50 via-orange-50 to-yellow-50"
          : "bg-gradient-to-b from-sky-50 via-teal-50 to-yellow-50"
      } flex flex-col`}
    >
      {/* Header */}
      <div className="px-6 pt-12 pb-4">
        <button
          onClick={() => navigate("/emotions", { state: { mode } })}
          className="text-gray-500 mb-6 hover:text-gray-700 transition-colors"
        >
          ← Retour
        </button>
      </div>

      {/* Main Content */}
      <div className="flex-1 px-6 pb-12 overflow-y-auto">
        <div className="max-w-md mx-auto space-y-6">

          {/* Emotion icon */}
          {emotionId && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex justify-center mb-6"
            >
              <div className={`bg-gradient-to-br ${emotionColor} rounded-full p-6 shadow-xl inline-block`}>
                <div className="text-white drop-shadow-lg">
                  {getEmotionIcon(emotionId)}
                </div>
              </div>
            </motion.div>
          )}

          {/* ML score indicator — masqué en critique (anxiogène) et en mode enfant */}
          {finalScore !== null && distressLevel !== "critical" && mode !== "kids" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="flex justify-center"
            >
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/60 text-xs text-gray-500 backdrop-blur-sm">
                <div
                  className={`w-2 h-2 rounded-full ${
                    distressLevel === "elevated" ? "bg-orange-400" : "bg-green-400"
                  }`}
                />
                Analyse IA · score {Math.round(finalScore * 100)}%
              </div>
            </motion.div>
          )}

          {/* Critical — zone de sécurité */}
          {distressLevel === "critical" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="bg-gradient-to-br from-orange-100 to-pink-100 rounded-3xl p-6 shadow-xl border-2 border-pink-200"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-md">
                  <Shield className="w-6 h-6 text-pink-500" />
                </div>
                <h2 className="text-xl" style={{ color: "#2A5F7D" }}>
                  {mode === "kids" ? "Tu n'es pas seul" : "Vous n'êtes pas seul(e)"}
                </h2>
              </div>
              <p className="text-gray-700 leading-relaxed">{getSupportMessage()}</p>
            </motion.div>
          )}

          {/* Elevated / Light — message de soutien */}
          {distressLevel !== "critical" && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white/80 backdrop-blur-sm rounded-3xl p-6 shadow-lg"
            >
              <div className="flex items-start gap-3">
                <Heart className="w-6 h-6 text-pink-400 flex-shrink-0 mt-1" fill="currentColor" />
                <div>
                  <h2 className="mb-3" style={{ color: "#2A5F7D" }}>
                    {mode === "kids" ? "Message pour toi" : "Message de soutien"}
                  </h2>
                  <p className="text-gray-700 leading-relaxed">{getSupportMessage()}</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Ressources d'urgence — critical & elevated */}
          {(distressLevel === "critical" || distressLevel === "elevated") && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="space-y-3"
            >
              <h3 className="text-center mb-4" style={{ color: "#2A5F7D" }}>
                {mode === "kids"
                  ? "Des personnes sont là pour t'aider"
                  : "Ressources d'aide disponibles"}
              </h3>

              {distressLevel === "critical" && (
                <motion.a
                  href="tel:3114"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full bg-gradient-to-r from-pink-400 to-rose-400 rounded-2xl p-5 shadow-lg hover:shadow-xl transition-all flex items-center gap-4 border-2 border-pink-300"
                >
                  <div className="bg-white rounded-full p-3">
                    <Phone className="w-7 h-7 text-pink-600" />
                  </div>
                  <div className="text-left flex-1 text-white">
                    <div className="font-semibold text-lg">Ligne d'urgence 24/7</div>
                    <div className="text-sm opacity-90">3114 — Gratuit et confidentiel</div>
                  </div>
                </motion.a>
              )}

              <div className="w-full bg-gradient-to-r from-blue-200 to-cyan-200 rounded-2xl p-4 shadow-md flex items-center gap-4">
                <div className="bg-white rounded-full p-3">
                  <MessageCircle className="w-6 h-6 text-blue-600" />
                </div>
                <div className="text-left flex-1">
                  <div className="text-gray-800 font-medium">
                    {mode === "kids" ? "Parler à quelqu'un de confiance" : "Parler à un proche"}
                  </div>
                  <div className="text-sm text-gray-600">
                    {mode === "kids"
                      ? "Parent, ami, professeur..."
                      : "Famille, ami, ou professionnel"}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Encouragement — light */}
          {distressLevel === "light" && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-gradient-to-br from-yellow-100 to-amber-100 rounded-3xl p-6 shadow-lg text-center"
            >
              <Sparkles className="w-8 h-8 text-yellow-500 mx-auto mb-3" />
              <p className="text-gray-700">
                {mode === "kids"
                  ? "Continue à prendre soin de toi ! 🌟"
                  : "Continuez à cultiver ces émotions positives 🌱"}
              </p>
            </motion.div>
          )}

          {/* CTA → Solutions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            <button
              onClick={() => navigate("/solutions", { state: { diagnosticProfile } })}
              className="w-full bg-gradient-to-r from-teal-400 via-cyan-400 to-blue-400 text-white rounded-3xl py-5 shadow-lg hover:shadow-xl transition-all flex items-center justify-center gap-3 font-medium"
            >
              <span>
                {mode === "kids" ? "Voir ce qui peut t'aider" : "Voir mes pistes d'action"}
              </span>
              <ArrowRight className="w-5 h-5" />
            </button>
          </motion.div>

        </div>
      </div>

      <div className="h-8"></div>
    </div>
  );
}
