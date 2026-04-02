from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import TrainingConfig
from data_utils import build_dataloaders, load_dataset
from download_pretrained import download_pretrained
from model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BERT sentiment classifier on financial headlines."
    )
    parser.add_argument("--data-path", type=str, default="dataset/all-data.csv")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/bert_financial_sentiment"
    )
    parser.add_argument(
        "--model-name", type=str, default="pretrained/prajjwal1-bert-tiny"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-model-id", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove existing output directory before training to avoid mixed logs/artifacts.",
    )
    return parser.parse_args()


def evaluate(model, data_loader, device, criterion, return_predictions: bool = False):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    result = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro")),
    }
    if return_predictions:
        result["predictions"] = all_preds
        result["labels"] = all_labels
    return result


def _serialize_config(cfg: TrainingConfig) -> dict:
    cfg_dict = asdict(cfg)
    cfg_dict["data_path"] = str(cfg_dict["data_path"])
    cfg_dict["output_dir"] = str(cfg_dict["output_dir"])
    return cfg_dict


def _save_reports(
    output_dir: Path,
    history: list[dict],
    split_frames: dict[str, pd.DataFrame],
    id2label: dict[int, str],
    y_true: list[int],
    y_pred: list[int],
) -> None:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in split_frames.items():
        split_df.to_csv(
            report_dir / f"{split_name}_split.csv", index=False, encoding="utf-8"
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(report_dir / "epoch_metrics.csv", index=False, encoding="utf-8")

    if history:
        epochs = [item["epoch"] for item in history]
        train_loss = [item["train_loss"] for item in history]
        val_loss = [item["val_loss"] for item in history]
        val_f1 = [item["val_macro_f1"] for item in history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="train_loss", marker="o")
        plt.plot(epochs, val_loss, label="val_loss", marker="o")
        plt.plot(epochs, val_f1, label="val_macro_f1", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / "training_curves.png", dpi=150)
        plt.close()

    label_order = list(range(len(id2label)))
    label_names = [id2label[i] for i in label_order]

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(report_dir / "confusion_matrix.csv", encoding="utf-8")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(label_names))
    plt.xticks(ticks, label_names, rotation=45)
    plt.yticks(ticks, label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    test_df = split_frames["test"].copy()
    test_df["true_label"] = [id2label[i] for i in y_true]
    test_df["pred_label"] = [id2label[i] for i in y_pred]
    test_df.to_csv(report_dir / "test_predictions.csv", index=False, encoding="utf-8")

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    with open(report_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(cls_report, f, ensure_ascii=False, indent=2)


def _resolve_model_source(
    model_name: str, hub_model_id: str, num_labels: int
) -> tuple[str, bool]:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path), True

    if model_name.startswith("pretrained/"):
        print(
            f"Local model not found at {model_path}. Downloading from hub: {hub_model_id}"
        )
        download_pretrained(
            model_id=hub_model_id, output_dir=model_path, num_labels=num_labels
        )
        return str(model_path), True

    return model_name, False


def train() -> None:
    args = parse_args()

    cfg = TrainingConfig(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        random_seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    set_seed(cfg.random_seed)
    if args.clean_output and cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_dataset(str(cfg.data_path))
    label_space = sorted(df["label"].unique().tolist())
    num_labels = len(label_space)

    model_source, local_files_only = _resolve_model_source(
        cfg.model_name, args.hub_model_id, num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, local_files_only=local_files_only, use_fast=False
    )
    (
        train_loader,
        val_loader,
        test_loader,
        label2id,
        id2label,
        split_info,
        split_frames,
    ) = build_dataloaders(
        df,
        cfg,
        tokenizer,
    )

    model = build_model(
        model_source,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        local_files_only=local_files_only,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    total_steps = max(1, len(train_loader) * cfg.num_epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("Split info:", split_info)

    history = []
    best_val_f1 = -1.0
    best_ckpt_dir = cfg.output_dir / "best"
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(cfg.num_epochs):
        model.train()
        train_losses = []

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}", leave=False
        )
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_metrics = evaluate(model, val_loader, device, criterion)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_metrics)
        print(epoch_metrics)

        if val_metrics["macro_f1"] > (best_val_f1 + args.early_stop_min_delta):
            best_val_f1 = val_metrics["macro_f1"]
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_ckpt_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_ckpt_dir)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                stopped_early = True
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best val_macro_f1: {best_val_f1:.6f}"
                )
                break

    best_model = build_model(
        str(best_ckpt_dir),
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )
    best_model.to(device)
    test_eval = evaluate(
        best_model, test_loader, device, criterion, return_predictions=True
    )
    test_metrics = {
        "loss": test_eval["loss"],
        "accuracy": test_eval["accuracy"],
        "macro_f1": test_eval["macro_f1"],
    }

    _save_reports(
        output_dir=cfg.output_dir,
        history=history,
        split_frames=split_frames,
        id2label=id2label,
        y_true=test_eval["labels"],
        y_pred=test_eval["predictions"],
    )

    artifacts = {
        "config": _serialize_config(cfg),
        "model_source": model_source,
        "local_files_only": local_files_only,
        "early_stopping": {
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "stopped_early": stopped_early,
            "trained_epochs": len(history),
        },
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "split_info": split_info,
        "history": history,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
    }

    with open(cfg.output_dir / "training_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)

    print("Training complete.")
    print("Best validation macro F1:", best_val_f1)
    print("Test metrics:", test_metrics)
    print("Saved to:", cfg.output_dir)


if __name__ == "__main__":
    train()
