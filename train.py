#!/usr/bin/env python3
"""
CrisisLens — Fine-tuning script for crisis relevance classification.
Fine-tunes XLM-RoBERTa on HumAID for binary crisis/not-crisis classification.

Usage:
    python train.py --data_dir data/raw/humaid --output_dir models/finetuned
    python train.py --data_dir data/raw/humaid --epochs 3 --batch_size 16

Requires: pip install datasets transformers accelerate
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/humaid")
    parser.add_argument("--output_dir", type=str, default="models/finetuned")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        import numpy as np
        from sklearn.metrics import f1_score, precision_recall_fscore_support
    except ImportError as e:
        logger.error(
            "Install required packages: pip install datasets transformers accelerate"
        )
        raise SystemExit(1) from e

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(
            f"Data dir {data_dir} not found. Run: python data/download_datasets.py"
        )
        logger.info("HumAID download: answer 'y' when prompted for HumAID dataset")
        raise SystemExit(1)

    # Load from local CSV (HumAID format) or JSON
    train_files = list(data_dir.glob("*train*.csv")) + list(data_dir.glob("*train*.json"))
    if not train_files:
        logger.error(f"No train files in {data_dir}")
        raise SystemExit(1)

    val_files = list(data_dir.glob("*val*")) + list(data_dir.glob("*development*")) + list(data_dir.glob("*test*"))
    data_files = {"train": str(train_files[0])}
    if val_files:
        data_files["validation"] = str(val_files[0])

    dataset = load_dataset(
        "csv" if str(train_files[0]).endswith(".csv") else "json",
        data_files=data_files,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # HumAID: tweet_text, class_label (not_humanitarian → 0, else 1)
    text_col = "tweet_text" if "tweet_text" in dataset["train"].column_names else "text"
    label_col = "class_label" if "class_label" in dataset["train"].column_names else "label"

    def to_binary(labels):
        return [0 if str(l).lower() == "not_humanitarian" else 1 for l in labels]

    def tokenize(examples):
        out = tokenizer(
            examples[text_col],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        if label_col in dataset["train"].column_names:
            out["labels"] = to_binary(examples[label_col])
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"f1": float(f1), "precision": float(p), "recall": float(r)}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
    )

    eval_ds = tokenized.get("validation") or tokenized.get("development") or tokenized["train"]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info(f"Model saved to {args.output_dir}")
    if tokenized.get("validation"):
        metrics = trainer.evaluate()
        logger.info(f"Validation F1: {metrics.get('eval_f1', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
