from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    data_path: Path = Path("dataset/all-data.csv")
    output_dir: Path = Path("outputs/bert_financial_sentiment")
    model_name: str = "pretrained/prajjwal1-bert-tiny"
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_seed: int = 42
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
