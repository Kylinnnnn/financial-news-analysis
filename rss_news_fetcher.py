from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import pandas as pd

from predict import predict_sentiment


DEFAULT_MODEL_DIR = "outputs/bert_financial_sentiment_es50_p5/best"
DEFAULT_ARTIFACTS_PATH = "outputs/bert_financial_sentiment_es50_p5/training_artifacts.json"
DEFAULT_LOCAL_DATA = "sample_news.csv"

RSS_ANALYSIS_PREFIX = "rss_analysis"
LOCAL_ANALYSIS_PREFIX = "local_analysis"

RSS_FEEDS = [
    {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news/rssindex",
    },
]

RESULT_DIR = Path("runtime_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = RESULT_DIR / "rss_seen_ids.txt"


def log_info(message: str) -> None:
    print(f"[Info] {message}")


def log_warn(message: str) -> None:
    print(f"[Warning] {message}")


def log_error(message: str) -> None:
    print(f"[Error] {message}")


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_seen_ids() -> set[str]:
    if CACHE_FILE.exists():
        return set(CACHE_FILE.read_text(encoding="utf-8").splitlines())
    return set()


def _save_seen_ids(seen_ids: set[str]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text("\n".join(sorted(seen_ids)), encoding="utf-8")


def _make_news_id(title: str, link: str) -> str:
    raw = f"{title.strip()}||{link.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_entry_datetime(entry) -> datetime | None:
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_time is not None:
        try:
            return datetime(*parsed_time[:6], tzinfo=timezone.utc)
        except Exception:
            pass

    raw_candidates = [entry.get("published", ""), entry.get("updated", "")]

    for raw_dt in raw_candidates:
        raw_dt = str(raw_dt).strip()
        if not raw_dt:
            continue

        try:
            dt = parsedate_to_datetime(raw_dt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

        try:
            dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    return None


def _is_within_days(dt: datetime | None, days: int = 7) -> bool:
    if dt is None:
        return False
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)


def _safe_parse_feed(feed_name: str, feed_url: str):
    try:
        parsed = feedparser.parse(feed_url)

        if getattr(parsed, "bozo", 0):
            bozo_exception = getattr(parsed, "bozo_exception", None)
            log_warn(f"RSS feed may have parsing/network issues: {feed_name} - {bozo_exception}")

        return parsed
    except Exception as e:
        log_error(f"Failed to fetch RSS feed: {feed_name} - {e}")
        return None


def analyze_texts(texts: list[str]) -> list[dict]:
    return predict_sentiment(
        texts=texts,
        model_dir=Path(DEFAULT_MODEL_DIR),
        artifacts_path=Path(DEFAULT_ARTIFACTS_PATH),
        max_length=128,
    )


def attach_prediction_results(df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    output_df = df.copy()

    safe_len = min(len(output_df), len(results))
    output_df = output_df.iloc[:safe_len].copy()
    results = results[:safe_len]

    output_df["predicted_label"] = [x.get("predicted_label") for x in results]
    output_df["confidence"] = [x.get("confidence") for x in results]
    output_df["score_negative"] = [x.get("scores", {}).get("negative") for x in results]
    output_df["score_neutral"] = [x.get("scores", {}).get("neutral") for x in results]
    output_df["score_positive"] = [x.get("scores", {}).get("positive") for x in results]

    return output_df


def build_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "predicted_label" not in df.columns:
        return pd.DataFrame(
            [
                {"sentiment": "positive", "count": 0, "ratio": 0.0, "percentage": "0.00%"},
                {"sentiment": "neutral", "count": 0, "ratio": 0.0, "percentage": "0.00%"},
                {"sentiment": "negative", "count": 0, "ratio": 0.0, "percentage": "0.00%"},
                {"sentiment": "TOTAL", "count": 0, "ratio": 0.0, "percentage": "0.00%"},
            ]
        )

    total = len(df)
    counts = df["predicted_label"].value_counts(dropna=False)

    label_order = ["positive", "neutral", "negative"]
    rows: list[dict] = []

    for label in label_order:
        count = int(counts.get(label, 0))
        ratio = count / total if total > 0 else 0.0
        rows.append(
            {
                "sentiment": label,
                "count": count,
                "ratio": ratio,
                "percentage": f"{ratio:.2%}",
            }
        )

    other_labels = [x for x in counts.index.tolist() if x not in label_order]
    for label in other_labels:
        count = int(counts.get(label, 0))
        ratio = count / total if total > 0 else 0.0
        rows.append(
            {
                "sentiment": str(label),
                "count": count,
                "ratio": ratio,
                "percentage": f"{ratio:.2%}",
            }
        )

    rows.append(
        {
            "sentiment": "TOTAL",
            "count": total,
            "ratio": 1.0 if total > 0 else 0.0,
            "percentage": "100.00%" if total > 0 else "0.00%",
        }
    )

    return pd.DataFrame(rows)


def print_sentiment_summary(summary_df: pd.DataFrame) -> None:
    total_row = summary_df[summary_df["sentiment"] == "TOTAL"]
    if not total_row.empty:
        log_info(f"Total analyzed news: {int(total_row.iloc[0]['count'])}")

    for _, row in summary_df.iterrows():
        sentiment = str(row["sentiment"])
        if sentiment == "TOTAL":
            continue
        log_info(f"{sentiment:<8}: {int(row['count'])} items, ratio {row['percentage']}")


def drop_export_only_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    if "text" in output_df.columns:
        output_df = output_df.drop(columns=["text"])
    return output_df


def save_csv(df: pd.DataFrame, file_path: Path) -> None:
    df.to_csv(file_path, index=False, encoding="utf-8-sig")


def save_timestamped_results(detail_df: pd.DataFrame, prefix: str) -> tuple[Path, Path]:
    timestamp = get_timestamp()
    detail_path = RESULT_DIR / f"{prefix}_{timestamp}.csv"
    summary_path = RESULT_DIR / f"{prefix}_summary_{timestamp}.csv"

    export_detail_df = drop_export_only_columns(detail_df)
    summary_df = build_sentiment_summary(detail_df)

    save_csv(export_detail_df, detail_path)
    save_csv(summary_df, summary_path)

    return detail_path, summary_path


def append_to_history(detail_df: pd.DataFrame, history_prefix: str) -> tuple[Path, Path, pd.DataFrame]:
    history_detail_path = RESULT_DIR / f"{history_prefix}_all.csv"
    history_summary_path = RESULT_DIR / f"{history_prefix}_summary_all.csv"

    export_detail_df = drop_export_only_columns(detail_df)

    if history_detail_path.exists():
        old_df = pd.read_csv(
            history_detail_path,
            encoding="utf-8",
            encoding_errors="replace",
        )
        combined_df = pd.concat([old_df, export_detail_df], ignore_index=True)
    else:
        combined_df = export_detail_df.copy()

    combined_summary_df = build_sentiment_summary(combined_df)

    save_csv(combined_df, history_detail_path)
    save_csv(combined_summary_df, history_summary_path)

    return history_detail_path, history_summary_path, combined_summary_df


def _prepare_texts(df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, list[str]]:
    output_df = df.copy()
    output_df[text_column] = output_df[text_column].astype(str).str.strip()
    output_df = output_df[output_df[text_column].str.len() > 0].reset_index(drop=True)
    texts = output_df[text_column].tolist()
    return output_df, texts


def _deduplicate_rows(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    existing_subset = [col for col in subset if col in df.columns]
    if not existing_subset:
        return df.copy()
    return df.drop_duplicates(subset=existing_subset, keep="first").reset_index(drop=True)


def _print_preview(df: pd.DataFrame, preview_columns: list[str], title: str) -> None:
    log_info(title)
    existing_columns = [col for col in preview_columns if col in df.columns]
    if not existing_columns:
        log_warn("No preview columns available.")
        return
    print(df[existing_columns].head(10))


def _run_analysis_pipeline(
    df: pd.DataFrame,
    texts: list[str],
    prefix: str,
    finish_message: str,
    preview_columns: list[str],
) -> None:
    if df.empty or not texts:
        log_info("No valid text available for sentiment analysis.")
        return

    results = analyze_texts(texts)
    if not results:
        log_warn("The model did not return valid prediction results.")
        return

    analyzed_df = attach_prediction_results(df, results)

    if analyzed_df.empty:
        log_warn("Prediction results are empty. Cannot generate analysis output.")
        return

    current_detail_path, current_summary_path = save_timestamped_results(analyzed_df, prefix)
    current_summary_df = build_sentiment_summary(analyzed_df)

    history_detail_path, history_summary_path, history_summary_df = append_to_history(
        analyzed_df, prefix
    )

    log_info(finish_message)
    log_info(f"Current analysis detail file saved to: {current_detail_path}")
    log_info(f"Current analysis summary file saved to: {current_summary_path}")
    log_info(f"History detail file updated at: {history_detail_path}")
    log_info(f"History summary file updated at: {history_summary_path}")

    print("\n[Info] Current analysis summary:")
    print_sentiment_summary(current_summary_df)

    print("\n[Info] Historical analysis summary:")
    print_sentiment_summary(history_summary_df)

    print()
    _print_preview(analyzed_df, preview_columns, "Current analysis detail preview:")


def fetch_rss_news() -> pd.DataFrame:
    seen_ids = _load_seen_ids()
    new_seen_ids = set(seen_ids)
    rows: list[dict] = []

    for feed in RSS_FEEDS:
        feed_name = feed["name"]
        feed_url = feed["url"]

        parsed = _safe_parse_feed(feed_name, feed_url)
        if parsed is None:
            continue

        entries = getattr(parsed, "entries", [])
        if not entries:
            log_warn(f"No entries were returned from feed: {feed_name}")
            continue

        for entry in entries:
            title = _clean_text(entry.get("title", ""))
            link = str(entry.get("link", "")).strip()

            if not title:
                continue

            entry_dt = _parse_entry_datetime(entry)
            if not _is_within_days(entry_dt, days=7):
                continue

            news_id = _make_news_id(title, link)
            if news_id in seen_ids:
                continue

            new_seen_ids.add(news_id)

            rows.append(
                {
                    "source": feed_name,
                    "title": title,
                    "text": title,
                    "link": link,
                    "published": entry_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    _save_seen_ids(new_seen_ids)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _deduplicate_rows(df, subset=["source", "title", "link"])
    return df


def run_rss_analysis() -> None:
    log_info("Fetching RSS financial news...")
    news_df = fetch_rss_news()

    if news_df.empty:
        log_info("No new news data was fetched.")
        return

    text_source_column = "text" if "text" in news_df.columns else "title"
    news_df, texts = _prepare_texts(news_df, text_source_column)

    _run_analysis_pipeline(
        df=news_df,
        texts=texts,
        prefix=RSS_ANALYSIS_PREFIX,
        finish_message="RSS news sentiment analysis completed.",
        preview_columns=[
            "source",
            "title",
            "predicted_label",
            "confidence",
            "score_negative",
            "score_neutral",
            "score_positive",
        ],
    )


def run_local_file_analysis() -> None:
    file_path = input(
        f"Please enter the local CSV file path (press Enter to use default: {DEFAULT_LOCAL_DATA}): "
    ).strip()

    if not file_path:
        file_path = DEFAULT_LOCAL_DATA
    else:
        file_path = file_path.strip('"').strip("'")

    text_column = input("Please enter the text column name (default: title): ").strip() or "title"

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        log_error(f"File does not exist: {file_path_obj}")
        return

    df = pd.read_csv(file_path_obj, encoding="utf-8", encoding_errors="replace")

    if text_column not in df.columns:
        log_error(f"Column '{text_column}' does not exist. Available columns: {list(df.columns)}")
        return

    df, texts = _prepare_texts(df, text_column)
    df["text"] = texts
    df = _deduplicate_rows(df, subset=[text_column])

    texts = df["text"].tolist()

    _run_analysis_pipeline(
        df=df,
        texts=texts,
        prefix=LOCAL_ANALYSIS_PREFIX,
        finish_message="Local file sentiment analysis completed.",
        preview_columns=[
            text_column,
            "predicted_label",
            "confidence",
            "score_negative",
            "score_neutral",
            "score_positive",
        ],
    )


if __name__ == "__main__":
    demo_df = fetch_rss_news()
    print(demo_df.head())
    print(f"Fetched a total of {len(demo_df)} new news articles from the last 7 days.")