#!/usr/bin/env python3
"""
CrisisLens — Evaluation script for pipeline benchmarks.
Runs the pipeline on a test set and outputs precision/recall/F1.

Usage:
    python evaluate.py
    python evaluate.py --data data/samples/sample_messages.json
    python evaluate.py --dataset humaid --split test

Outputs a classification report and saves results to evaluation_results.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_data(path: Path):
    """Load evaluation data from JSON or CSV. Supports HumAID format (tweet_text, class_label)."""
    import pandas as pd

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [d["text"] for d in data], [d.get("type", d.get("label", "")) for d in data]
        raise ValueError("JSON must be a list of {text, type} objects")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        text_col = "tweet_text" if "tweet_text" in df.columns else "text"
        label_col = "class_label" if "class_label" in df.columns else ("label" if "label" in df.columns else "type")
        if text_col not in df.columns:
            raise ValueError("CSV must have 'text' or 'tweet_text' column")
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(str).tolist() if label_col in df.columns else [""] * len(texts)
        return texts, labels
    raise ValueError(f"Unsupported format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/samples/sample_messages.json",
        help="Path to test data (JSON or CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for metrics",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for quick eval)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (evaluates fine-tuned; omit for pipeline)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        samples_path = Path(__file__).parent / "data" / "samples" / "sample_messages.json"
        if samples_path.exists():
            logger.info("Using sample_messages.json for demo evaluation...")
            with open(samples_path, encoding="utf-8") as f:
                data = json.load(f)
            texts = [d["text"] for d in data]
            labels = [d.get("type", "NOT_CRISIS") for d in data]
        else:
            logger.error("No evaluation data found. Run data/download_datasets.py first.")
            raise SystemExit(1)
    else:
        texts, labels = load_sample_data(data_path)

    if args.limit:
        texts, labels = texts[: args.limit], labels[: args.limit]

    def to_gold_relevant(lbl):
        return str(lbl).lower() != "not_humanitarian" and "not" not in str(lbl).lower()

    if args.model_path:
        # Evaluate fine-tuned model
        logger.info(f"Evaluating fine-tuned model on {len(texts)} samples...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.eval()
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            preds = (logits.argmax(dim=1) == 1).tolist()
            for j, (text, gold) in enumerate(zip(batch, labels[i : i + batch_size])):
                pred_relevant = preds[j]
                gold_relevant = to_gold_relevant(gold)
                results.append({
                    "text": text[:80] + "...",
                    "gold": gold,
                    "pred_relevant": pred_relevant,
                    "gold_relevant": gold_relevant,
                    "match": pred_relevant == gold_relevant,
                })
    else:
        # Evaluate full pipeline (BART)
        logger.info(f"Evaluating pipeline on {len(texts)} samples...")
        from src.pipeline.orchestrator import CrisisLensPipeline

        pipeline = CrisisLensPipeline()
        pipeline.load_models()
        results = []
        for text, gold in zip(texts, labels):
            r = pipeline.analyze(text, skip_dedup=True)
            pred_relevant = r.is_relevant
            gold_relevant = to_gold_relevant(gold)
            results.append({
                "text": text[:80] + "...",
                "gold": gold,
                "pred_relevant": pred_relevant,
                "gold_relevant": gold_relevant,
                "match": pred_relevant == gold_relevant,
            })

    # Compute metrics
    tp = sum(1 for r in results if r["pred_relevant"] and r["gold_relevant"])
    fp = sum(1 for r in results if r["pred_relevant"] and not r["gold_relevant"])
    fn = sum(1 for r in results if not r["pred_relevant"] and r["gold_relevant"])
    tn = sum(1 for r in results if not r["pred_relevant"] and not r["gold_relevant"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(results) if results else 0

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "n_samples": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    logger.info("=" * 50)
    logger.info("Evaluation Results (Relevance Classification)")
    logger.info("=" * 50)
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1:        {metrics['f1']:.4f}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info("=" * 50)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
