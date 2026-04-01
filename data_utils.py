from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import TrainingConfig


class FinancialNewsDataset(Dataset):
    def __init__(
        self, texts: list[str], labels: list[int], tokenizer, max_length: int
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_dataset(csv_path: str) -> pd.DataFrame:
    # all-data.csv has no header: first column is sentiment, second column is title text.
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def build_label_map(df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def split_dataframe(
    df: pd.DataFrame, cfg: TrainingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((cfg.train_size + cfg.val_size + cfg.test_size) - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - cfg.train_size),
        random_state=cfg.random_seed,
        stratify=df["label"],
    )

    val_ratio_in_temp = cfg.val_size / (cfg.val_size + cfg.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=cfg.random_seed,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _limit_samples(
    df: pd.DataFrame, max_samples: int | None, random_seed: int
) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)


def build_dataloaders(df: pd.DataFrame, cfg: TrainingConfig, tokenizer):
    label2id, id2label = build_label_map(df)
    train_df, val_df, test_df = split_dataframe(df, cfg)

    train_df = _limit_samples(train_df, cfg.max_train_samples, cfg.random_seed)
    val_df = _limit_samples(val_df, cfg.max_eval_samples, cfg.random_seed)
    test_df = _limit_samples(test_df, cfg.max_eval_samples, cfg.random_seed)

    train_labels = [label2id[label] for label in train_df["label"].tolist()]
    val_labels = [label2id[label] for label in val_df["label"].tolist()]
    test_labels = [label2id[label] for label in test_df["label"].tolist()]

    train_dataset = FinancialNewsDataset(
        train_df["text"].tolist(), train_labels, tokenizer, cfg.max_length
    )
    val_dataset = FinancialNewsDataset(
        val_df["text"].tolist(), val_labels, tokenizer, cfg.max_length
    )
    test_dataset = FinancialNewsDataset(
        test_df["text"].tolist(), test_labels, tokenizer, cfg.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    cfg_dict = asdict(cfg)
    cfg_dict["data_path"] = str(cfg_dict["data_path"])
    cfg_dict["output_dir"] = str(cfg_dict["output_dir"])

    split_info = {
        "full": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "config": cfg_dict,
        "label_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
    }

    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    return (
        train_loader,
        val_loader,
        test_loader,
        label2id,
        id2label,
        split_info,
        split_frames,
    )
