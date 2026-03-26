/**
 * sessionHistory — Historique local des sessions émotionnelles
 *
 * Stockage : sessionStorage (on-device uniquement, effacé à la fermeture de l'onglet)
 * Rétention : session courante uniquement, 10 sessions max
 * Déduplique automatiquement si la même session est soumise dans les 5 minutes
 *
 * Utilisé pour afficher la tendance (amélioration / stable / dégradation)
 * sur l'écran Solutions, sans aucune transmission de données personnelles.
 */

import type { ClinicalProfile } from "../types/diagnostic";
import type { TriageLevel } from "../types/solutions";

export interface SessionRecord {
  date: string;             // ISO string
  level: TriageLevel;
  emotionId: string;
  finalScore: number | null;
  clinicalProfile: ClinicalProfile;
}

export type Trend = "improving" | "stable" | "worsening";

const STORAGE_KEY  = "mh_session_history";
const MAX_SESSIONS = 10;
const DEDUP_MS     = 5  * 60 * 1000;            // 5 minutes

export function getSessions(): SessionRecord[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as SessionRecord[];
  } catch (err) {
    if (err instanceof SyntaxError) {
      // Données corrompues — nettoyage silencieux
      try { sessionStorage.removeItem(STORAGE_KEY); } catch { /* quota */ }
    }
    return [];
  }
}

/**
 * Efface tout l'historique local (RGPD Art. 17 — droit à l'effacement).
 * Peut être appelé via un bouton "Effacer mon historique" dans les paramètres.
 */
export function clearSessions(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // sessionStorage indisponible — silencieux
  }
}

export function saveSession(record: SessionRecord): void {
  try {
    const sessions = getSessions();

    // Déduplique : ne pas sauvegarder si une session existe déjà dans les 5 dernières minutes
    const dedupCutoff = Date.now() - DEDUP_MS;
    if (sessions.some((s) => new Date(s.date).getTime() > dedupCutoff)) return;

    sessions.push(record);
    // Garder les MAX_SESSIONS les plus récentes
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(-MAX_SESSIONS)));
  } catch {
    // sessionStorage indisponible (mode privé, quota dépassé) → silencieux
  }
}

/**
 * Calcule la tendance sur les 2 dernières sessions.
 * Retourne null si moins de 2 sessions disponibles.
 *
 * Seuil : delta de ≥ 1 niveau pour considérer une évolution significative.
 */
export function getTrend(sessions: SessionRecord[]): Trend | null {
  if (sessions.length < 2) return null;
  const delta = sessions[sessions.length - 1].level - sessions[sessions.length - 2].level;
  if (delta <= -1) return "improving";
  if (delta >= 1)  return "worsening";
  return "stable";
}
