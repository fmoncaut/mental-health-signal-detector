-- ============================================================
-- Migration Supabase — Table de collecte anonyme opt-in
-- Phase 8 Mental Health Signal Detector
--
-- À exécuter dans l'éditeur SQL de Supabase (app.supabase.com)
-- ============================================================

-- 1. Table principale
CREATE TABLE IF NOT EXISTS anonymous_feedback (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    text            text        NOT NULL CHECK (char_length(text) BETWEEN 1 AND 5000),
    emotion         text        NOT NULL CHECK (emotion IN ('joy','sadness','anger','fear','stress','calm','tiredness','pride')),
    distress_level  smallint    NOT NULL CHECK (distress_level BETWEEN 0 AND 4),
    score_ml        real        CHECK (score_ml BETWEEN 0.0 AND 1.0),
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- 2. Index pour faciliter les requêtes d'entraînement
CREATE INDEX IF NOT EXISTS idx_feedback_distress ON anonymous_feedback (distress_level);
CREATE INDEX IF NOT EXISTS idx_feedback_created  ON anonymous_feedback (created_at DESC);

-- 3. Row Level Security — écriture uniquement depuis le service_role
--    (aucun client ne peut lire les données directement)
ALTER TABLE anonymous_feedback ENABLE ROW LEVEL SECURITY;

-- Politique : seule la clé service_role peut insérer
CREATE POLICY "insert_service_only"
    ON anonymous_feedback
    FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Lecture : uniquement service_role (pour exports d'entraînement)
CREATE POLICY "select_service_only"
    ON anonymous_feedback
    FOR SELECT
    TO service_role
    USING (true);

-- 4. Rétention automatique — supprimer les entrées de plus de 2 ans
--    (optionnel : nécessite pg_cron activé dans Supabase)
-- SELECT cron.schedule('cleanup-old-feedback', '0 3 * * 0',
--   $$DELETE FROM anonymous_feedback WHERE created_at < now() - interval '2 years'$$);

-- ============================================================
-- Variables d'environnement à configurer dans Render/Vercel :
--   SUPABASE_URL         = https://<project-ref>.supabase.co
--   SUPABASE_SERVICE_KEY = <service_role key> (Settings > API)
-- ============================================================
