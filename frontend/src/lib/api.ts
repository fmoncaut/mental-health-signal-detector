/**
 * Base URL de l'API backend.
 * - Dev : "" (Vite proxy vers localhost:8000)
 * - Prod Vercel : VITE_API_URL=https://mon-app.onrender.com
 */
export const API_BASE = import.meta.env.VITE_API_URL ?? "";

/**
 * Modèle NLP pour /predict.
 * - "baseline" → TF-IDF + LR (Render slim, défaut prod)
 * - "distilbert" → DistilBERT fine-tuned (local avec modèle)
 * - "mental_bert_v3" → Mental-BERT v3
 * - "mental_roberta" → Mental-RoBERTa
 */
const SUPPORTED_MODEL_TYPES = new Set([
	"baseline",
	"distilbert",
	"mental_bert_v3",
	"mental_roberta",
]);

const envModelType = (import.meta.env.VITE_MODEL_TYPE as string | undefined)?.trim();
export const MODEL_TYPE = envModelType && SUPPORTED_MODEL_TYPES.has(envModelType)
	? envModelType
	: "baseline";
