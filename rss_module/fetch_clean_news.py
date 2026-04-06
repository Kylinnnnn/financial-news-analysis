from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_FEEDS_PATH = Path(__file__).with_name("feeds.json")

PUBLISHER_HINTS = {
    "24/7 wall st",
    "24/7 wall st.",
    "associated press",
    "ap news",
    "barron's",
    "bloomberg",
    "business insider",
    "cnbc",
    "coindesk",
    "forbes",
    "fortune",
    "investing.com",
    "marketwatch",
    "nvidia blog",
    "reuters",
    "seeking alpha",
    "south china morning post",
    "the wall street journal",
    "tom's hardware",
    "tweaktown",
    "wsj",
    "yahoo finance",
}


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def parse_datetime(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return utc_now_dt().isoformat()
    try:
        return parsedate_to_datetime(text).astimezone(timezone.utc).isoformat()
    except Exception:
        return utc_now_dt().isoformat()


def age_hours(timestamp: str | None) -> float:
    if not timestamp:
        return 10**9
    try:
        published_at = datetime.fromisoformat(timestamp).astimezone(timezone.utc)
    except Exception:
        return 10**9
    return max((utc_now_dt() - published_at).total_seconds() / 3600.0, 0.0)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def strip_html(text: str) -> str:
    return normalize_whitespace(re.sub(r"<[^>]+>", " ", text or ""))


def publisher_hint_match(text: str) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in PUBLISHER_HINTS)


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    hits = [keyword for keyword in keywords if keyword and keyword.lower() in lower]
    return list(dict.fromkeys(hits))


def is_probable_publisher_suffix(suffix: str) -> bool:
    suffix = normalize_whitespace(suffix)
    if not suffix:
        return False

    lower = suffix.lower()
    if publisher_hint_match(lower):
        return True
    if lower.endswith((".com", ".net", ".org", ".gov")):
        return True
    if len(suffix) > 45:
        return False
    if re.search(r"\d", suffix):
        return False
    if re.search(r"[!?;:]", suffix):
        return False

    words = [word for word in suffix.split() if word]
    if not 1 <= len(words) <= 5:
        return False

    titleish_words = sum(1 for word in words if word[:1].isupper())
    return titleish_words / len(words) >= 0.6


def clean_title(title: str) -> str:
    text = normalize_whitespace(html.unescape(title))
    for separator in [" - ", " | ", " — "]:
        if separator in text:
            head, tail = text.rsplit(separator, 1)
            if is_probable_publisher_suffix(tail):
                text = head
                break
    return text.strip(" '\"")


def clean_summary(description: str) -> str:
    text = normalize_whitespace(html.unescape(strip_html(description)))
    return clean_title(text).strip(" '\"")


def trim_redundant_summary(summary: str, cleaned_title: str) -> str:
    if not summary:
        return ""
    if summary == cleaned_title:
        return ""
    if summary.startswith(cleaned_title):
        tail = summary[len(cleaned_title) :].strip(" -|:")
        if not tail:
            return ""
        if is_probable_publisher_suffix(tail) or len(tail) <= 40:
            return ""
    return summary


def normalize_title_key(title: str) -> str:
    text = clean_title(title).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_whitespace(text)


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def iter_text(tag: ET.Element | None, *names: str) -> str:
    if tag is None:
        return ""
    for name in names:
        found = tag.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return ""


def parse_feed_items(xml_text: str, max_items: int) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    items: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        title = iter_text(item, "title")
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": iter_text(item, "link"),
                "description": iter_text(item, "description"),
                "published_at": parse_datetime(
                    iter_text(item, "pubDate", "published", "updated")
                ),
            }
        )
        if len(items) >= max_items:
            break
    return items


def fetch_feed(
    feed: dict[str, Any], timeout_sec: float, max_items_per_feed: int
) -> list[dict[str, str]]:
    request = Request(
        str(feed.get("url") or "").strip(),
        headers={
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "User-Agent": "financial-news-analysis/rss-module",
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            xml_text = response.read().decode("utf-8", errors="ignore")
        return parse_feed_items(xml_text, max_items=max_items_per_feed)
    except (HTTPError, URLError, ET.ParseError):
        return []


def fetch_news(
    feeds_path: Path = DEFAULT_FEEDS_PATH,
    timeout_sec: float = 10.0,
    max_items_per_feed: int = 10,
) -> list[dict[str, str]]:
    feeds_raw = load_json(feeds_path, {})
    feeds = feeds_raw.get("feeds") if isinstance(feeds_raw, dict) else []

    seen_keys: set[str] = set()
    normalized_items: list[dict[str, str]] = []

    for feed in feeds:
        if not isinstance(feed, dict) or not feed.get("url"):
            continue

        max_age_hours = float(feed.get("max_age_hours") or 72.0)
        required_keywords = [str(value).strip() for value in feed.get("required_keywords") or []]
        min_keyword_hits = int(feed.get("min_keyword_hits") or 0)

        for item in fetch_feed(
            feed, timeout_sec=timeout_sec, max_items_per_feed=max_items_per_feed
        ):
            cleaned_title = clean_title(item.get("title") or "")
            if not cleaned_title:
                continue

            cleaned_summary = trim_redundant_summary(
                clean_summary(item.get("description") or ""),
                cleaned_title=cleaned_title,
            )
            if required_keywords and min_keyword_hits > 0:
                combined_text = f"{cleaned_title} {cleaned_summary}".strip()
                if len(keyword_hits(combined_text, required_keywords)) < min_keyword_hits:
                    continue

            title_key = normalize_title_key(cleaned_title)
            if not title_key or title_key in seen_keys:
                continue

            published_at = item.get("published_at") or utc_now_dt().isoformat()
            if age_hours(published_at) > max_age_hours:
                continue

            seen_keys.add(title_key)
            normalized_items.append(
                {
                    "id": f"news-{feed['source_id']}-{short_hash(title_key)}",
                    "source": str(feed.get("name") or feed.get("source_id") or "rss"),
                    "source_id": str(feed.get("source_id") or "rss"),
                    "title": cleaned_title,
                    "summary": cleaned_summary,
                    "url": str(item.get("link") or ""),
                    "published_at": published_at,
                }
            )

    normalized_items.sort(key=lambda row: row["published_at"], reverse=True)
    return normalized_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and clean finance RSS items.")
    parser.add_argument("--feeds-file", type=Path, default=DEFAULT_FEEDS_PATH)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--max-items-per-feed", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = fetch_news(
        feeds_path=args.feeds_file,
        timeout_sec=args.timeout_sec,
        max_items_per_feed=args.max_items_per_feed,
    )
    body = json.dumps(items, ensure_ascii=False, indent=2)
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(body, encoding="utf-8")
        print(f"saved {len(items)} cleaned items to {args.output_file}")
        return
    print(body)


if __name__ == "__main__":
    main()
