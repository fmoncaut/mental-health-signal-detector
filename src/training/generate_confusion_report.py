"""Generate a detailed comparative confusion-matrix image for local models.

Usage example:

python -m src.training.generate_confusion_report \
  --kaggle-path data/raw/reddit_depression_dataset.csv \
  --erisk25-path data/raw/erisk25 \
  --no-dair \
  --max-test-samples 2000
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.training.evaluate import plot_confusion_matrices_comparative
from src.training.predict import load_model
from src.training.preprocess import clean_text
from src.training.preprocess import build_dataset


@dataclass
class ModelEvalResult:
    name: str
    available: bool
    y_true: list[int] | None = None
    y_pred: list[int] | None = None
    reason: str | None = None


def _build_quick_erisk_eval_df(erisk25_path: str, max_rows: int = 4000, max_files: int = 50) -> pd.DataFrame:
    """Build a lightweight evaluation dataframe from a subset of eRisk files."""
    json_dir = Path(erisk25_path) / "final-eriskt2-dataset-with-ground-truth" / "all_combined"
    files = sorted(glob.glob(str(json_dir / "subject_*.json")))
    if not files:
        raise FileNotFoundError(f"No subject_*.json found in {json_dir}")

    rows: list[dict] = []
    for filepath in files[:max_files]:
        with open(filepath, encoding="utf-8") as f:
            posts = json.load(f)

        for post in posts:
            sub = post.get("submission", {})
            target = sub.get("target")
            if target is None:
                continue
            title = sub.get("title") or ""
            body = sub.get("body") or ""
            text = clean_text((title + " " + body).strip())
            if len(text) > 10:
                rows.append({"text": text, "label": int(bool(target))})
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    if not rows:
        raise ValueError("Quick eRisk sampling produced no rows")

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(f"Quick eRisk sample: {len(df)} rows | distribution:\n{df['label'].value_counts()}")
    return df


def _predict_transformer_labels(model_bundle: dict, texts: list[str], model_type: str, batch_size: int = 32) -> np.ndarray:
    import torch

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    model.eval()

    preds: list[int] = []
    threshold = 0.30 if model_type == "mental_roberta" else 0.65

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        # DistilBERT does not use token_type_ids.
        if model_type == "distilbert":
            inputs.pop("token_type_ids", None)

        with torch.no_grad():
            logits = model(**inputs).logits
            proba_pos = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        preds.extend((proba_pos > threshold).astype(int).tolist())

    return np.asarray(preds, dtype=int)


def _evaluate_one_model(name: str, model_type: str, texts: list[str], y_true: np.ndarray, batch_size: int) -> ModelEvalResult:
    try:
        loaded = load_model(model_type)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return ModelEvalResult(name=name, available=False, reason=f"load failed: {exc}")

    try:
        if model_type == "baseline":
            y_pred = loaded.predict(texts)
        else:
            y_pred = _predict_transformer_labels(loaded, texts, model_type=model_type, batch_size=batch_size)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return ModelEvalResult(name=name, available=False, reason=f"predict failed: {exc}")

    return ModelEvalResult(
        name=name,
        available=True,
        y_true=y_true.astype(int).tolist(),
        y_pred=np.asarray(y_pred, dtype=int).tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparative confusion matrices for local models")
    parser.add_argument("--kaggle-path", type=str, default="data/raw/reddit_depression_dataset.csv")
    parser.add_argument("--erisk25-path", type=str, default="data/raw/erisk25")
    parser.add_argument("--smhd-path", type=str, default=None)
    parser.add_argument("--go-emotions", action="store_true", help="Include GoEmotions in dataset build")
    parser.add_argument("--no-dair", action="store_true", help="Exclude DAIR dataset")
    parser.add_argument("--clinical-only", action="store_true", help="Clinical mode: no DAIR and no GoEmotions")
    parser.add_argument("--kaggle-samples", type=int, default=100000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="confusion_matrices_comparative.png")
    parser.add_argument("--quick-erisk-files", type=int, default=50)
    parser.add_argument("--quick-erisk-rows", type=int, default=4000)
    parser.add_argument(
        "--models",
        type=str,
        default="baseline,distilbert,mental_bert_v3,mental_roberta",
        help="Comma-separated model ids: baseline,distilbert,mental_bert_v3,mental_roberta",
    )
    args = parser.parse_args()

    use_dair = (not args.no_dair) and (not args.clinical_only)
    use_go = args.go_emotions and (not args.clinical_only)

    logger.info("Building evaluation split...")
    if args.kaggle_path == "" and args.erisk25_path and args.quick_erisk_files > 0:
        from sklearn.model_selection import train_test_split

        quick_df = _build_quick_erisk_eval_df(
            erisk25_path=args.erisk25_path,
            max_rows=args.quick_erisk_rows,
            max_files=args.quick_erisk_files,
        )
        _, test_df = train_test_split(
            quick_df,
            test_size=0.2,
            random_state=42,
            stratify=quick_df["label"],
        )
        test_df = test_df.reset_index(drop=True)
    else:
        _, test_df = build_dataset(
            kaggle_path=args.kaggle_path,
            use_dair=use_dair,
            use_go_emotions=use_go,
            erisk25_path=args.erisk25_path,
            smhd_path=args.smhd_path,
            kaggle_max_samples=args.kaggle_samples,
        )

    if args.max_test_samples and len(test_df) > args.max_test_samples:
        from sklearn.model_selection import train_test_split

        test_df, _ = train_test_split(
            test_df,
            train_size=args.max_test_samples,
            random_state=42,
            stratify=test_df["label"],
        )
        test_df = test_df.reset_index(drop=True)

    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["label"].to_numpy(dtype=int)

    all_model_specs = {
        "baseline": ("Baseline TF-IDF + LR", "baseline"),
        "distilbert": ("DistilBERT v2", "distilbert"),
        "mental_bert_v3": ("Mental-BERT v3", "mental_bert_v3"),
        "mental_roberta": ("Mental-RoBERTa", "mental_roberta"),
    }
    selected_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    model_specs = [all_model_specs[m] for m in selected_ids if m in all_model_specs]
    if not model_specs:
        raise ValueError("No valid model ids provided in --models")

    results: list[dict] = []
    for display_name, model_type in model_specs:
        logger.info(f"Evaluating {display_name} ({model_type})...")
        res = _evaluate_one_model(display_name, model_type, texts, y_true, batch_size=args.batch_size)
        results.append(
            {
                "name": res.name,
                "available": res.available,
                "y_true": res.y_true,
                "y_pred": res.y_pred,
                "reason": res.reason,
            }
        )

    output_path = plot_confusion_matrices_comparative(
        results,
        save=True,
        output_filename=args.output,
    )

    if output_path is None:
        raise SystemExit("Comparative confusion matrix image was not generated")

    logger.info(f"Done -> {output_path}")


if __name__ == "__main__":
    main()
