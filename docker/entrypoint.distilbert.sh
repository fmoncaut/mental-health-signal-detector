#!/bin/sh
# Entrypoint DistilBERT v2 — télécharge le modèle depuis HF Hub si absent
set -e

MODEL_DIR="/app/models/fine_tuned_v2"
MARKER="$MODEL_DIR/model.safetensors"

if [ ! -f "$MARKER" ]; then
  echo "Modèle absent — téléchargement depuis HuggingFace Hub..."
  if [ -z "$HF_REPO_ID" ]; then
    echo "ERREUR : variable HF_REPO_ID non définie (ex: fabrice/mh-distilbert-v2)"
    exit 1
  fi
  mkdir -p "$MODEL_DIR"
  python - <<PYEOF
import os
from huggingface_hub import hf_hub_download

repo = os.environ["HF_REPO_ID"]
token = os.environ.get("HF_TOKEN")
dest = "/app/models/fine_tuned_v2"

for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
    print(f"  Téléchargement {f}...")
    hf_hub_download(repo_id=repo, filename=f, local_dir=dest, token=token)
    print(f"  OK")
print("Modèle prêt.")
PYEOF
  echo "Téléchargement terminé ✅"
else
  echo "Modèle déjà présent ✅"
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
