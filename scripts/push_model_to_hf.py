"""
Upload DistilBERT v2 vers HuggingFace Hub.

Usage :
    pip install huggingface_hub
    python scripts/push_model_to_hf.py --repo TON_USERNAME/mh-distilbert-v2

Le repo sera privé par défaut (données cliniques).
Récupérer le token sur : https://huggingface.co/settings/tokens (rôle Write)
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "fine_tuned_v2"

FILES_TO_UPLOAD = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Ex: fabrice-moncaut/mh-distilbert-v2")
    parser.add_argument("--token", default=None, help="HF token (sinon lit HF_TOKEN env var)")
    args = parser.parse_args()

    login(token=args.token)
    api = HfApi()

    api.create_repo(repo_id=args.repo, repo_type="model", private=True, exist_ok=True)
    print(f"Repo : https://huggingface.co/{args.repo}")

    for filename in FILES_TO_UPLOAD:
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"  SKIP {filename} (absent)")
            continue
        print(f"  Upload {filename} ({path.stat().st_size // 1024 // 1024} Mo)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  OK ✅")

    print(f"\nModèle disponible sur : https://huggingface.co/{args.repo}")
    print("Ajoute HF_REPO_ID et HF_TOKEN dans les env vars Render.")


if __name__ == "__main__":
    main()
