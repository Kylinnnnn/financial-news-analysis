from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from flask import Flask, request, render_template_string, send_file, url_for

from rss_news_fetcher import (
    analyze_texts,
    append_to_history,
    attach_prediction_results,
    build_sentiment_summary,
    fetch_rss_news,
    save_timestamped_results,
)

app = Flask(__name__)


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Financial News Sentiment Analysis</title>
    <style>
        :root {
            --bg: #0b0f14;
            --card: #131a22;
            --card-2: #18212b;
            --text: #e6edf3;
            --muted: #9fb0c3;
            --border: #283341;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --success-bg: #0f2a1f;
            --success-text: #7ee787;
            --error-bg: #311111;
            --error-text: #ff9b9b;
            --table-head: #1b2530;
            --table-row: #111821;
            --table-row-alt: #0f141b;
            --shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background:
                radial-gradient(circle at top left, #16202a 0%, transparent 28%),
                radial-gradient(circle at top right, #1a2440 0%, transparent 24%),
                linear-gradient(180deg, #0b0f14 0%, #0a0d12 100%);
            color: var(--text);
            min-height: 100vh;
        }

        .container {
            width: 100%;
            max-width: 1180px;
            margin: 0 auto;
            padding: 32px 16px 48px;
        }

        .hero {
            margin-bottom: 24px;
            padding: 28px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(59,130,246,0.14), rgba(99,102,241,0.08));
            box-shadow: var(--shadow);
        }

        .hero h1 {
            margin: 0 0 10px;
            font-size: 32px;
            line-height: 1.2;
            color: #ffffff;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            font-size: 15px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }

        .card {
            background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px;
            box-shadow: var(--shadow);
        }

        .card h2 {
            margin-top: 0;
            margin-bottom: 14px;
            color: #ffffff;
            font-size: 22px;
        }

        .card p {
            color: var(--muted);
            margin-top: 0;
            margin-bottom: 18px;
            line-height: 1.6;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #dce6f2;
            font-size: 14px;
            font-weight: 600;
        }

        input[type="text"],
        input[type="file"] {
            width: 100%;
            max-width: 460px;
            padding: 12px 14px;
            margin-bottom: 16px;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: #0d131a;
            color: var(--text);
            outline: none;
        }

        input[type="text"]::placeholder {
            color: #6f8196;
        }

        input[type="text"]:focus,
        input[type="file"]:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        button,
        .download-btn {
            display: inline-block;
            border: none;
            border-radius: 10px;
            padding: 12px 18px;
            background: linear-gradient(180deg, var(--primary) 0%, var(--primary-hover) 100%);
            color: #ffffff;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
            transition: transform 0.15s ease, opacity 0.15s ease;
            margin-right: 10px;
            margin-bottom: 10px;
        }

        button:hover,
        .download-btn:hover {
            transform: translateY(-1px);
            opacity: 0.96;
        }

        .message {
            margin-bottom: 20px;
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid transparent;
            font-size: 14px;
            line-height: 1.5;
        }

        .message.info {
            background: var(--success-bg);
            color: var(--success-text);
            border-color: rgba(126, 231, 135, 0.2);
        }

        .message.error {
            background: var(--error-bg);
            color: var(--error-text);
            border-color: rgba(255, 155, 155, 0.18);
        }

        .table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            min-width: 760px;
            background: var(--table-row);
        }

        th, td {
            border-bottom: 1px solid var(--border);
            padding: 12px 14px;
            text-align: left;
            font-size: 14px;
            vertical-align: top;
        }

        th {
            background: var(--table-head);
            color: #ffffff;
        }

        tr:nth-child(even) td {
            background: var(--table-row-alt);
        }

        .footer-note {
            margin-top: 18px;
            color: #7f90a4;
            font-size: 12px;
        }

        .save-info {
            margin-top: 14px;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.8;
            word-break: break-all;
        }

        @media (min-width: 900px) {
            .grid.two {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <section class="hero">
            <h1>Financial News Sentiment Analysis</h1>
            <p>
                Analyze live RSS financial headlines or upload a local CSV file,
                save the results automatically, and download the generated CSV files.
            </p>
        </section>

        {% if message %}
        <div class="message {{ message_type }}">
            {{ message }}
        </div>
        {% endif %}

        <section class="grid two">
            <div class="card">
                <h2>Analyze RSS News</h2>
                <p>Fetch the latest unseen RSS financial news articles and run sentiment analysis.</p>
                <form method="post" action="/analyze_rss">
                    <button type="submit">Run RSS Analysis</button>
                </form>
            </div>

            <div class="card">
                <h2>Analyze Local CSV</h2>
                <p>Upload a CSV file and specify the text column to analyze your own data.</p>
                <form method="post" action="/analyze_csv" enctype="multipart/form-data">
                    <label for="csv_file">Upload CSV File</label>
                    <input id="csv_file" type="file" name="csv_file" accept=".csv" required>

                    <label for="text_column">Text Column Name</label>
                    <input id="text_column" type="text" name="text_column" placeholder="title">

                    <button type="submit">Upload and Analyze</button>
                </form>
            </div>
        </section>

        {% if detail_download_url or summary_download_url %}
        <section class="card" style="margin-top: 20px;">
            <h2>Download Results</h2>
            {% if detail_download_url %}
            <a class="download-btn" href="{{ detail_download_url }}">Download Detail CSV</a>
            {% endif %}
            {% if summary_download_url %}
            <a class="download-btn" href="{{ summary_download_url }}">Download Summary CSV</a>
            {% endif %}

            {% if detail_save_path or summary_save_path %}
            <div class="save-info">
                {% if detail_save_path %}<div><strong>Detail file:</strong> {{ detail_save_path }}</div>{% endif %}
                {% if summary_save_path %}<div><strong>Summary file:</strong> {{ summary_save_path }}</div>{% endif %}
            </div>
            {% endif %}
        </section>
        {% endif %}

        {% if summary_html %}
        <section class="card" style="margin-top: 20px;">
            <h2>Sentiment Summary</h2>
            <div class="table-wrap">
                {{ summary_html|safe }}
            </div>
        </section>
        {% endif %}

        {% if detail_html %}
        <section class="card" style="margin-top: 20px;">
            <h2>Detail Preview</h2>
            <div class="table-wrap">
                {{ detail_html|safe }}
            </div>
            <div class="footer-note">
                Showing the first 20 rows only.
            </div>
        </section>
        {% endif %}
    </div>
</body>
</html>
"""


def dataframe_to_html(df: pd.DataFrame | None, max_rows: int | None = None) -> str:
    if df is None:
        return ""

    display_df = df.head(max_rows) if max_rows is not None else df
    return display_df.to_html(index=False, border=0)


def render_result(
    message: str = "",
    message_type: str = "info",
    summary_df: pd.DataFrame | None = None,
    detail_df: pd.DataFrame | None = None,
    detail_save_path: str = "",
    summary_save_path: str = "",
    detail_download_url: str = "",
    summary_download_url: str = "",
):
    return render_template_string(
        HTML_PAGE,
        message=message,
        message_type=message_type,
        summary_html=dataframe_to_html(summary_df),
        detail_html=dataframe_to_html(detail_df, max_rows=20),
        detail_save_path=detail_save_path,
        summary_save_path=summary_save_path,
        detail_download_url=detail_download_url,
        summary_download_url=summary_download_url,
    )


@app.route("/", methods=["GET"])
def home():
    return render_result()


@app.route("/download")
def download_file():
    file_path = request.args.get("path", "").strip()
    if not file_path:
        return "Missing file path.", 400

    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        return "File not found.", 404

    return send_file(path_obj, as_attachment=True)


@app.route("/analyze_rss", methods=["POST"])
def analyze_rss():
    try:
        news_df = fetch_rss_news()

        if news_df.empty:
            return render_result(
                message="No new RSS news articles were fetched.",
                message_type="info",
            )

        text_col = "text" if "text" in news_df.columns else "title"
        news_df[text_col] = news_df[text_col].astype(str).str.strip()
        news_df = news_df[news_df[text_col].str.len() > 0].reset_index(drop=True)

        if news_df.empty:
            return render_result(
                message="The fetched RSS news does not contain valid text for analysis.",
                message_type="info",
            )

        texts = news_df[text_col].tolist()
        results = analyze_texts(texts)
        news_df = attach_prediction_results(news_df, results)
        summary_df = build_sentiment_summary(news_df)

        detail_path, summary_path = save_timestamped_results(news_df, "rss_analysis")
        append_to_history(news_df, "rss_analysis")

        preview_columns = [
            col for col in [
                "source",
                "title",
                "predicted_label",
                "confidence",
                "score_negative",
                "score_neutral",
                "score_positive",
            ] if col in news_df.columns
        ]

        return render_result(
            message="RSS news analysis completed successfully. Results have been saved.",
            message_type="info",
            summary_df=summary_df,
            detail_df=news_df[preview_columns],
            detail_save_path=str(detail_path),
            summary_save_path=str(summary_path),
            detail_download_url=url_for("download_file", path=str(detail_path)),
            summary_download_url=url_for("download_file", path=str(summary_path)),
        )
    except Exception as e:
        return render_result(
            message=f"RSS news analysis failed: {e}",
            message_type="error",
        )


@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():
    try:
        file = request.files.get("csv_file")
        text_column = (request.form.get("text_column") or "title").strip()

        if not file:
            return render_result(
                message="Please upload a CSV file first.",
                message_type="error",
            )

        content = file.read().decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(content))

        if text_column not in df.columns:
            return render_result(
                message=f"Column '{text_column}' was not found. Available columns: {list(df.columns)}",
                message_type="error",
            )

        df[text_column] = df[text_column].astype(str).str.strip()
        df = df[df[text_column].str.len() > 0].reset_index(drop=True)

        if df.empty:
            return render_result(
                message="The CSV file does not contain valid text for analysis.",
                message_type="info",
            )

        df["text"] = df[text_column]
        texts = df["text"].tolist()

        results = analyze_texts(texts)
        df = attach_prediction_results(df, results)
        summary_df = build_sentiment_summary(df)

        detail_path, summary_path = save_timestamped_results(df, "local_analysis")
        append_to_history(df, "local_analysis")

        preview_columns = [
            col for col in [
                text_column,
                "predicted_label",
                "confidence",
                "score_negative",
                "score_neutral",
                "score_positive",
            ] if col in df.columns
        ]

        return render_result(
            message="Local CSV analysis completed successfully. Results have been saved.",
            message_type="info",
            summary_df=summary_df,
            detail_df=df[preview_columns],
            detail_save_path=str(detail_path),
            summary_save_path=str(summary_path),
            detail_download_url=url_for("download_file", path=str(detail_path)),
            summary_download_url=url_for("download_file", path=str(summary_path)),
        )
    except Exception as e:
        return render_result(
            message=f"CSV analysis failed: {e}",
            message_type="error",
        )


if __name__ == "__main__":
    app.run(debug=True)