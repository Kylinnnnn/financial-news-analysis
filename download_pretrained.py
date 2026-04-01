from __future__ import annotations

import argparse
from pathlib import Path

from transformers import BertForSequenceClassification, BertTokenizer


def download_pretrained(model_id: str, output_dir: Path, num_labels: int = 3) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        use_safetensors=True,
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained BERT and save to a local folder."
    )
    parser.add_argument("--model-id", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument(
        "--output-dir", type=str, default="pretrained/prajjwal1-bert-tiny"
    )
    parser.add_argument("--num-labels", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = download_pretrained(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        num_labels=args.num_labels,
    )
    print(f"Saved pretrained assets to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
