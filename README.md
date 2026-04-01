# Financial News Sentiment (BERT)

This project fine-tunes a pretrained BERT model on financial news headlines from `dataset/all-data.csv` and supports reproducible training with clean logs, early stopping, and report-friendly artifacts.

## Dataset

- File: `dataset/all-data.csv`
- Format: no header, 2 columns
  - Column 1: sentiment label (`positive`, `negative`, `neutral`)
  - Column 2: news headline text
- Full size: 4,846 samples

## Environment and Dependencies
`environment.yml`

## One-Time Pretrained Download (Local Reuse)

Download once and reuse local files in all later runs:

```powershell
python download_pretrained.py --model-id prajjwal1/bert-tiny --output-dir pretrained/prajjwal1-bert-tiny
```

## Current Best Trained Run (Project Status)

Latest organized run:

- Output directory: `outputs/bert_financial_sentiment_es50_p5`
- Max epochs: 50
- Early stopping: patience=5, min_delta=0.001
- Triggered early stop at epoch 27

Key metrics from `training_artifacts.json`:

- Best validation macro F1: `0.759630094438087`
- Test accuracy: `0.7587628865979381`
- Test macro F1: `0.732219786238157`

Per-class (test set) from `classification_report.json`:

- Negative: precision `0.6912`, recall `0.7705`, F1 `0.7287`
- Neutral: precision `0.8315`, recall `0.8056`, F1 `0.8183`
- Positive: precision `0.6449`, recall `0.6544`, F1 `0.6496`

## Reproduce the Latest Training Configuration

```powershell
python train.py ^
  --epochs 50 ^
  --batch-size 16 ^
  --model-name pretrained/prajjwal1-bert-tiny ^
  --output-dir outputs/bert_financial_sentiment_es50_p5 ^
  --early-stop-patience 5 ^
  --early-stop-min-delta 0.001 ^
  --clean-output
```

Notes:

- `--clean-output` removes existing output directory before training, preventing mixed or appended logs.
- The model is loaded from local pretrained assets first.

## Inference

```powershell
python predict.py --model-dir outputs/bert_financial_sentiment_es50_p5/best --artifacts outputs/bert_financial_sentiment_es50_p5/training_artifacts.json --text "The company posted stronger earnings and raised its full-year guidance"
```

Output fields:

- Predicted label
- Confidence (max softmax probability)
- Scores (per-class probabilities)

## Output Structure

Each run directory contains:

- `best/` : best validation checkpoint (model + tokenizer)
- `training_artifacts.json` : config, label maps, split info, history, early stopping status, final test metrics
- `reports/train_split.csv` : train split samples used in the run
- `reports/val_split.csv` : validation split samples used in the run
- `reports/test_split.csv` : test split samples used in the run
- `reports/epoch_metrics.csv` : per-epoch metrics (single-run clean log)
- `reports/test_predictions.csv` : test text with true and predicted labels
- `reports/classification_report.json` : precision/recall/F1 report
- `reports/confusion_matrix.csv` : confusion matrix table
- `reports/training_curves.png` : training and validation curves
- `reports/confusion_matrix.png` : confusion matrix figure

## Core Components

- `config.py` : default training configuration and hyperparameters
- `data_utils.py` : CSV loading, cleaning, label mapping, stratified split, dataloaders
- `model.py` : sequence classification model builder (local files + safetensors support)
- `download_pretrained.py` : one-time pretrained model/tokenizer download and local persistence
- `train.py` : training loop, optimizer/scheduler, early stopping, best checkpoint, report generation
- `predict.py` : single-sentence inference with confidence and class probability table
