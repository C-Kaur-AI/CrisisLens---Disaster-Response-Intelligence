# CrisisLens — Evaluation Metrics

This document summarizes the benchmark metrics and how to reproduce them (and generate figures) for the repo.

---

## Summary table (for README / reports)

| Metric | Zero-Shot (BART-MNLI) | Fine-Tuned (XLM-RoBERTa on HumAID) |
|--------|------------------------|-------------------------------------|
| **Relevance F1** | 0.76–0.82 | **0.9649** |
| **Precision** | — | 0.9521 |
| **Recall** | — | 0.978 |
| **Type Macro-F1** | 0.68–0.74 | — |
| **Urgency κ** | 0.65–0.71 | — |
| **Geocoding Recall** | ~0.72 | — |

- **Relevance**: Binary crisis vs non-crisis (HumAID: humanitarian / not_humanitarian).
- **Fine-tuned**: XLM-RoBERTa on HumAID validation set (7,793 samples). Test-set numbers from `evaluate.py` (see below).
- **Type / Urgency / Geocoding**: Pipeline (BART + heuristics); reported ranges from internal eval.

---

## How to reproduce metrics

### 1. Relevance (fine-tuned model)

**Requirements:** HumAID test data and fine-tuned model at `models/finetuned` (or run training first).

```bash
# From project root
python evaluate.py --data data/raw/humaid/test.csv --model_path models/finetuned --output test_results.json
```

- **Output:** `test_results.json` with `metrics` (precision, recall, f1, accuracy) and per-sample `results`.
- Use these values to update the table above if you re-train or use a different split.

### 2. Relevance (full pipeline with BART)

```bash
python evaluate.py --data data/raw/humaid/test.csv --output pipeline_results.json
# Omit --model_path to use the default pipeline (BART-based relevance).
```

### 3. Evaluation figures (confusion matrix, ROC, PR curve)

Uses the fine-tuned model and HumAID test (or subset).

```bash
python evaluate_plots.py --data data/raw/humaid/test.csv --model_path models/finetuned --output_dir docs/figures
```

**Generated files in `docs/figures/`:**

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Confusion matrix (Not crisis / Crisis). |
| `roc_curve.png` | ROC curve with AUC. |
| `precision_recall_curve.png` | Precision–Recall curve with Average Precision. |

Add these to the repo and reference them in README or METRICS.md, e.g.:

```markdown
### Relevance classification (fine-tuned)

![Confusion Matrix](figures/confusion_matrix.png)
![ROC Curve](figures/roc_curve.png)
![Precision-Recall Curve](figures/precision_recall_curve.png)
```

---

## Where to put numbers in the repo

- **README:** Keep the “Benchmark & Evaluation Metrics” table; optionally add one line: *“See [docs/METRICS.md](docs/METRICS.md) for reproduction steps.”*
- **Dashboard sidebar:** Already shows F1, Type Macro-F1, Urgency κ, geocoding recall; keep in sync with this doc.
- **Reports / applications:** Copy the summary table and, if needed, attach `test_results.json` or the generated figures.
