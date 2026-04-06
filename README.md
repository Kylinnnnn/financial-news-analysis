# Financial News Sentiment (BERT)

This project fine-tunes a pretrained BERT model on financial news headlines from `dataset/all-data.csv` and supports reproducible training with clean logs, early stopping, and report-friendly artifacts.

It now also includes an isolated `rss_module/` that fetches finance-relevant RSS headlines, cleans them, and sends cleaned titles into the existing prediction pipeline without changing the original training/inference structure.

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
or
```powershell
python train.py --clean-output
```
also works

Notes:

- `--clean-output` removes existing output directory before training, preventing mixed or appended logs.
- The model is loaded from local pretrained assets first.

## Inference
Use `outputs/bert_financial_sentiment_es50_p5/best` as default model directory and use `outputs/bert_financial_sentiment_es50_p5/training_artifacts.json` as default artifacts path

**One-time usage**:
```powershell
python predict.py --text "The company posted stronger earnings and raised its full-year guidance"
```

**Use as a module and transfer text list**:
```python
# your python file
from predict import predict_sentiment

strs = ["This is str1","This is str2","This is str3"]

# example return 
# [
#   {
#       "text":"This is str1",
#       "predicted_label": "label result",
#       "confidence": "confidence score",
#       "scores": "scores for each label"
#   }，
#   {
#       "text":"This is str2",
#       "predicted_label": "label result",
#       "confidence": "confidence score",
#       "scores": "scores for each label"
#   }，
#   {
#       "text":"This is str3",
#       "predicted_label": "label result",
#       "confidence": "confidence score",
#       "scores": "scores for each label"
#   }，
# ]
results = predict_sentiment(texts=strs)
```
Output fields:

- Predicted label
- Confidence (max softmax probability)
- Scores (per-class probabilities)

## RSS Module

The repository now includes a new folder:

- `rss_module/`

This folder adds RSS fetching and cleaning only. The original model files remain where they were, and `rss_module/run_rss_pipeline.py` imports the existing root-level `predict.py`.

What was added:

- `rss_module/feeds.json`
  Finance-focused Google News RSS source list with small keyword gates
- `rss_module/fetch_clean_news.py`
  RSS XML parsing, headline cleaning, age filtering, de-duplication, and light relevance filtering
- `rss_module/run_rss_pipeline.py`
  End-to-end RSS fetch + prediction runner using the existing sentiment model

What was not changed:

- `train.py`
- `predict.py`
- the model architecture
- the existing training outputs

### RSS Usage

Prepare a Python environment that satisfies the project dependencies first.
Using the packages in `environment.yml` is recommended.

Minimum runtime requirements for the RSS pipeline:

- Python
- `torch`
- `transformers`
- `pandas`

If you hit an OpenMP conflict on Windows, set the following environment variable before running:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

Run the full RSS pipeline from the repository root:

```powershell
python -m rss_module.run_rss_pipeline
```

Optional arguments:

```powershell
python -m rss_module.run_rss_pipeline --max-items-per-feed 6 --timeout-sec 10
```

### RSS Input

The RSS pipeline does not require manual text input.

It reads source definitions from:

- `rss_module/feeds.json`

The model input is the cleaned RSS `title` only.
`summary` is stored for inspection but is not sent into the model.

### RSS Output

RSS pipeline outputs are written to:

- `outputs/rss_module/rss_items.json`
- `outputs/rss_module/predictions.json`
- `outputs/rss_module/predictions.csv`

`rss_items.json` contains cleaned source items:

- `id`
- `source`
- `source_id`
- `title`
- `summary`
- `url`
- `published_at`

`predictions.json` and `predictions.csv` add:

- `predicted_label`
- `confidence`
- `scores`

The CSV is written as `utf-8-sig` so Excel on Windows can display punctuation correctly.

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
