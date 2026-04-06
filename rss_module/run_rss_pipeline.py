from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from predict import predict_sentiment
from rss_module import fetch_news

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "rss_module"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSS fetch + sentiment inference.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items-per-feed", type=int, default=10)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source",
        "source_id",
        "published_at",
        "title",
        "summary",
        "url",
        "predicted_label",
        "confidence",
        "negative_score",
        "neutral_score",
        "positive_score",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            scores = row.get("scores") or {}
            writer.writerow(
                {
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "source_id": row.get("source_id"),
                    "published_at": row.get("published_at"),
                    "title": row.get("title"),
                    "summary": row.get("summary"),
                    "url": row.get("url"),
                    "predicted_label": row.get("predicted_label"),
                    "confidence": row.get("confidence"),
                    "negative_score": scores.get("negative"),
                    "neutral_score": scores.get("neutral"),
                    "positive_score": scores.get("positive"),
                }
            )


def attach_predictions(news_items: list[dict[str, object]]) -> list[dict[str, object]]:
    predictions = predict_sentiment([str(item.get("title") or "") for item in news_items])
    rows: list[dict[str, object]] = []
    for item, prediction in zip(news_items, predictions):
        row = dict(item)
        row["model_input"] = prediction["text"]
        row["predicted_label"] = prediction["predicted_label"]
        row["confidence"] = prediction["confidence"]
        row["scores"] = prediction["scores"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    news_items = fetch_news(
        timeout_sec=args.timeout_sec,
        max_items_per_feed=args.max_items_per_feed,
    )
    prediction_rows = attach_predictions(news_items)

    rss_path = args.output_dir / "rss_items.json"
    predictions_json_path = args.output_dir / "predictions.json"
    predictions_csv_path = args.output_dir / "predictions.csv"

    write_json(rss_path, news_items)
    write_json(predictions_json_path, prediction_rows)
    write_csv(predictions_csv_path, prediction_rows)

    print(f"saved {len(news_items)} rss items to {rss_path}")
    print(f"saved {len(prediction_rows)} prediction rows to {predictions_json_path}")
    print(f"saved csv export to {predictions_csv_path}")

    for row in prediction_rows[:10]:
        print(
            f"[{row['predicted_label']:<8}] "
            f"{float(row['confidence']):.4f} | "
            f"{row['title']}"
        )


if __name__ == "__main__":
    main()
