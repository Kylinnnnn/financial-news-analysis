from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = "outputs/bert_financial_sentiment_es50_p5/best"
DEFAULT_ARTIFACTS_PATH = (
    "outputs/bert_financial_sentiment_es50_p5/training_artifacts.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sentiment for a financial headline."
    )
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--artifacts",
        type=str,
        default=DEFAULT_ARTIFACTS_PATH,
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def predict_sentiment(
    texts: list[str],
    model_dir: Path = DEFAULT_MODEL_DIR,
    artifacts_path: Path = DEFAULT_ARTIFACTS_PATH,
    max_length: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(artifacts_path, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    id2label = {int(k): v for k, v in artifacts["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    results = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

        pred_idx = int(torch.argmax(probs).item())
        pred_label = id2label[pred_idx]

        confidence = float(probs[pred_idx].item())
        score_table = {id2label[i]: float(probs[i].item()) for i in range(len(probs))}
        results.append(
            {
                "text": text,
                "predicted_label": pred_label,
                "confidence": confidence,
                "scores": score_table,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    results = predict_sentiment(
        texts=[args.text],
        model_dir=Path(args.model_dir),
        artifacts_path=Path(args.artifacts),
        max_length=args.max_length,
    )

    print("Input:", args.text)
    print("Predicted label:", results[0]["predicted_label"])
    print("Confidence:", round(results[0]["confidence"], 4))
    print("Scores:", results[0]["scores"])


if __name__ == "__main__":
    main()
