# Financial News Sentiment Project Core Guide

## 1. Project Goal

This project builds a 3-class sentiment classifier for financial news headlines.

- Labels: negative, neutral, positive
- Input: short financial news headline text
- Output: predicted label + confidence + class probabilities

The pipeline is end-to-end:

1. Data loading and cleaning
2. Train/val/test split
3. BERT fine-tuning
4. Evaluation and report export
5. Single-text inference

## 2. Dataset and Data Pipeline

Data source:

- `dataset/all-data.csv`

File format:

- No header
- Column 1: sentiment label
- Column 2: headline text

Processing flow (implemented in `data_utils.py`):

1. Read CSV with explicit column names: `label`, `text`
2. Drop rows with missing label/text
3. Normalize labels (strip + lower)
4. Strip text and remove empty text rows
5. Build label map:
   - negative -> 0
   - neutral -> 1
   - positive -> 2
6. Stratified split:
   - train: 80%
   - val: 10%
   - test: 10%
7. Tokenize text with fixed max length and attention mask
8. Build PyTorch `Dataset` and `DataLoader`

## 3. Model Architecture

Core model:

- Pretrained BERT from local cache: `pretrained/prajjwal1-bert-tiny`
- Task head: sequence classification head (`AutoModelForSequenceClassification`)
- Number of labels: 3

Model builder is in `model.py`:

- Supports local checkpoint loading
- Uses safetensors-compatible loading path

Inference path (`predict.py`):

1. Load tokenizer + model from checkpoint directory
2. Encode input text
3. Forward pass to obtain logits
4. Apply softmax to convert logits into probabilities
5. Return:
   - predicted label
   - confidence (max probability)
   - per-class scores

## 4. Training Core Design

Training implementation: `train.py`

### 4.1 Loss Function

- `CrossEntropyLoss` for 3-class classification
- Standard objective:

  `L = -sum(y_c * log(p_c))`

### 4.2 Optimizer and Scheduler

- Optimizer: `AdamW`
- Learning rate scheduler: linear warmup + linear decay (`get_linear_schedule_with_warmup`)
- Gradient clipping: `max_norm = 1.0`

### 4.3 Default/Current Hyperparameters

From current organized run:

- model_name: `pretrained/prajjwal1-bert-tiny`
- max_length: 128
- batch_size: 16
- learning_rate: 2e-5
- weight_decay: 0.01
- warmup_ratio: 0.1
- random_seed: 42
- max epochs: 50

### 4.4 Early Stopping

Early stopping is based on validation macro-F1.

- `patience = 5`
- `min_delta = 0.001`
- Stop when validation macro-F1 has no sufficient improvement for 5 consecutive epochs.

### 4.5 Clean Logging / No Contamination

To prevent mixed logs and appended artifacts across runs, training uses a clean-output strategy:

- `--clean-output` removes existing output directory before starting a new run.
- Each experiment should use an independent output directory.

## 5. Evaluation Metrics

The project reports multiple metrics:

1. Loss (train/val/test)
2. Accuracy
3. Macro-F1 (primary early-stopping metric)
4. Per-class precision/recall/F1
5. Confusion matrix

Why macro-F1 matters:

- Class distribution is not perfectly balanced.
- Macro-F1 gives equal weight to each class.
- It is better than accuracy for tracking minority-class quality.

## 6. Current Best Organized Run Summary

Run directory:

- `outputs/bert_financial_sentiment_es50_p5`

Training setup:

- Max epochs: 50
- Early stopping: patience=5, min_delta=0.001
- Early stop triggered at epoch 27

Key results:

- Best validation macro-F1: 0.759630094438087
- Test accuracy: 0.7587628865979381
- Test macro-F1: 0.732219786238157

Per-class test metrics:

- negative: precision 0.6912, recall 0.7705, f1 0.7287
- neutral: precision 0.8315, recall 0.8056, f1 0.8183
- positive: precision 0.6449, recall 0.6544, f1 0.6496

## 7. Report Artifacts (for papers/slides)

Inside each output directory:

- `training_artifacts.json`: config, split info, history, early stopping, final metrics
- `reports/epoch_metrics.csv`: epoch-level training log
- `reports/training_curves.png`: metric curves
- `reports/confusion_matrix.csv`: confusion matrix table
- `reports/confusion_matrix.png`: confusion matrix figure
- `reports/classification_report.json`: per-class precision/recall/F1
- `reports/test_predictions.csv`: row-level predictions on test set
- `reports/train_split.csv`, `reports/val_split.csv`, `reports/test_split.csv`: split snapshots

## 8. Core File Responsibilities

- `download_pretrained.py`: download pretrained model once and save locally
- `config.py`: central default training config
- `data_utils.py`: data loading/cleaning/splitting/dataloaders
- `model.py`: model construction wrapper
- `train.py`: training + early stopping + artifact generation
- `predict.py`: single-text prediction API

## 9. Reproducible Commands

Train (current recommended setup):

```powershell
python train.py --epochs 50 --batch-size 16 --model-name pretrained/prajjwal1-bert-tiny --output-dir outputs/bert_financial_sentiment_es50_p5 --early-stop-patience 5 --early-stop-min-delta 0.001 --clean-output
```

Predict:

```powershell
python predict.py --model-dir outputs/bert_financial_sentiment_es50_p5/best --artifacts outputs/bert_financial_sentiment_es50_p5/training_artifacts.json --text "The company posted stronger earnings and raised its full-year guidance"
```
