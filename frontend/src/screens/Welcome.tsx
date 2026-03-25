import { useNavigate } from "react-router";
import { motion } from "motion/react";
import { Sparkles, Heart } from "lucide-react";

export default function Welcome() {
  const navigate = useNavigate();

  const handleModeSelect = (mode: "kids" | "adult") => {
    navigate("/emotions", { state: { mode } });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-sky-100 via-teal-50 to-yellow-50 flex flex-col items-center justify-center px-6 relative overflow-hidden">
      {/* Decorative elements */}
      <div className="absolute top-20 left-10 opacity-30">
        <motion.div
          animate={{
            y: [0, -15, 0],
            rotate: [0, 5, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <Sparkles className="w-12 h-12 text-yellow-400" />
        </motion.div>
      </div>
      <div className="absolute top-40 right-10 opacity-30">
        <motion.div
          animate={{
            y: [0, 15, 0],
            rotate: [0, -5, 0],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <Heart className="w-10 h-10 text-pink-300" fill="currentColor" />
        </motion.div>
      </div>

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center z-10 max-w-sm"
      >
        {/* Icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", delay: 0.2, stiffness: 200 }}
          className="mb-8 inline-block"
        >
          <div className="w-24 h-24 bg-gradient-to-br from-teal-300 to-blue-300 rounded-full flex items-center justify-center shadow-xl">
            <Heart className="w-12 h-12 text-white" fill="white" />
          </div>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="text-4xl mb-4"
          style={{ color: "#2A5F7D" }}
        >
          Comment vous sentez-vous aujourd'hui ?
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="text-lg text-gray-600 mb-12"
        >
          On fait un point ensemble, sans pression.
        </motion.p>

        {/* Mode Selection Buttons */}
        <div className="space-y-4">
          <motion.button
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.8 }}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => handleModeSelect("kids")}
            className="w-full py-5 bg-gradient-to-r from-yellow-300 via-yellow-400 to-orange-300 rounded-3xl shadow-lg hover:shadow-xl transition-all text-white font-medium text-lg"
          >
            <div className="flex items-center justify-center gap-3">
              <Sparkles className="w-6 h-6" />
              <span>Mode Enfant</span>
            </div>
          </motion.button>

          <motion.button
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.9 }}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => handleModeSelect("adult")}
            className="w-full py-5 bg-gradient-to-r from-teal-400 via-cyan-400 to-blue-400 rounded-3xl shadow-lg hover:shadow-xl transition-all text-white font-medium text-lg"
          >
            <div className="flex items-center justify-center gap-3">
              <Heart className="w-6 h-6" />
              <span>Mode Adulte</span>
            </div>
          </motion.button>
        </div>

        {/* Helper text */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1 }}
          className="text-sm text-gray-500 mt-8"
        >
          Un espace sûr pour exprimer vos émotions 💙
        </motion.p>
      </motion.div>

      {/* Disclaimer */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.3 }}
        className="text-xs text-gray-400 text-center max-w-sm px-6 mt-6"
      >
        Cette application ne constitue pas un dispositif médical et ne remplace en aucun cas un avis médical, un diagnostic ou un traitement par un professionnel de santé.
      </motion.p>

      {/* Bottom Safe Area */}
      <div className="h-8"></div>
    </div>
  );
}
