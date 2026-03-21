from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=("settings_",))

    # Reddit API
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "mh-detector/1.0"

    # Database
    database_url: str = "sqlite:///./mhdetector.db"

    # Model
    model_name: str = "distilbert-base-uncased"
    model_path: str = "./models/fine_tuned_v2"   # DistilBERT v2 — prod nécessite Render Starter ($7/mois)
    model_path_v3: str = "./models/fine_tuned_v3/mental_bert_v3"
    model_path_roberta: str = "./models/mental_roberta_base.pkl"  # Mental-RoBERTa — seuil 0.30
    # Optionnel : empreinte SHA-256 du fichier model_path_roberta pour prévenir la dérive/tampering.
    model_sha256_roberta: str = ""

    # Claude API — optionnel (POST /analyze)
    anthropic_api_key: str = ""

    # Supabase — collecte anonyme opt-in (POST /feedback)
    supabase_url: str = ""
    supabase_service_key: str = ""

    # App
    env: str = "development"
    log_level: str = "INFO"
    # CORS — origines autorisées en production (séparées par des virgules)
    # Ex: https://monapp.vercel.app,https://www.mondomaine.com
    # Mettre "*" pour tout autoriser (non recommandé en production)
    allowed_origins: str = "*"
    # Proxy headers (X-Forwarded-For) : activer seulement derrière un proxy de confiance.
    trust_proxy_headers: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
