#!/usr/bin/env python3
"""
CrisisLens — Generate evaluation figures for the fine-tuned relevance classifier.
Produces: Confusion Matrix, ROC curve (AUC), Precision-Recall curve (AP).

Usage:
    python evaluate_plots.py --data data/raw/humaid/test.csv --model_path models/finetuned --output_dir figures
    python evaluate_plots.py --data data/raw/humaid/test.csv --model_path models/finetuned --limit 2000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sample_data(path: Path):
    """Load texts and labels from CSV (HumAID format)."""
    import pandas as pd
    df = pd.read_csv(path)
    text_col = "tweet_text" if "tweet_text" in df.columns else "text"
    label_col = "class_label" if "class_label" in df.columns else "label"
    if text_col not in df.columns:
        raise ValueError("CSV must have 'text' or 'tweet_text' column")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist() if label_col in df.columns else [""] * len(texts)
    return texts, labels


def to_binary(lbl):
    """humanitarian -> 1, not_humanitarian -> 0."""
    return 0 if str(lbl).lower() == "not_humanitarian" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/humaid/test.csv", help="Test CSV path")
    parser.add_argument("--model_path", type=str, default="models/finetuned", help="Fine-tuned model path")
    parser.add_argument("--output_dir", type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (e.g. 2000 for speed)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return 1

    texts, labels = load_sample_data(data_path)
    y_true = np.array([to_binary(l) for l in labels])
    if args.limit:
        texts, y_true = texts[: args.limit], y_true[: args.limit]

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    # Get probability of positive class (humanitarian)
    all_probs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        all_probs.extend(probs.tolist())
    y_score = np.array(all_probs)
    y_pred = (y_score >= 0.5).astype(int)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Confusion Matrix ───
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not crisis", "Crisis"])
    ax.set_yticklabels(["Not crisis", "Crisis"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.colorbar(im, ax=ax, label="Count")
    plt.title("Confusion Matrix (Relevance Classification)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'confusion_matrix.png'}")

    # ─── 2. ROC Curve + AUC ───
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Relevance Classification)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'roc_curve.png'} (AUC = {roc_auc:.4f})")

    # ─── 3. Precision-Recall Curve + Average Precision ───
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Relevance Classification)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'precision_recall_curve.png'} (AP = {ap:.4f})")

    # Summary metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("\nMetrics: Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}, AUC = {:.4f}, AP = {:.4f}".format(p, r, f1, roc_auc, ap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
