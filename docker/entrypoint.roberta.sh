#!/bin/sh
# Entrypoint Mental-RoBERTa — télécharge le pkl depuis HF Hub si absent
# Variables requises :
#   HF_REPO_ID  : ex. fabrice/mh-mental-roberta   (repo HF contenant le pkl)
#   HF_TOKEN    : token HuggingFace (si repo privé)
set -e

MODEL_PKL="/app/models/mental_roberta_base.pkl"

if [ ! -f "$MODEL_PKL" ]; then
  echo "Modèle absent — téléchargement depuis HuggingFace Hub..."
  if [ -z "$HF_REPO_ID" ]; then
    echo "ERREUR : variable HF_REPO_ID non définie (ex: fabrice/mh-mental-roberta)"
    exit 1
  fi
  mkdir -p /app/models
  python - <<PYEOF
import os
from huggingface_hub import hf_hub_download

repo  = os.environ["HF_REPO_ID"]
token = os.environ.get("HF_TOKEN")

print("  Téléchargement mental_roberta_base.pkl (~476 Mo)...")
hf_hub_download(
    repo_id=repo,
    filename="mental_roberta_base.pkl",
    local_dir="/app/models",
    token=token,
)
print("  Modèle prêt.")
PYEOF
  echo "Téléchargement terminé ✅"
else
  echo "Modèle déjà présent ✅"
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
