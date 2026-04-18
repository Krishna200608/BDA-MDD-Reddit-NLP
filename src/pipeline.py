from __future__ import annotations

import logging
import re
import warnings
import hashlib
from pathlib import Path
from typing import Final

import nltk
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from .scraper import PullPushScraper
except ImportError:
    from scraper import PullPushScraper


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z\s]")
MULTISPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
MIN_WORD_COUNT: Final[int] = 5
DATASET_SUMMARY_FILENAME: Final[str] = "dataset_summary.csv"


warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


def clean_text(text: object, stop_words: set[str]) -> str:
    if not isinstance(text, str):
        return ""

    cleaned_text = text.lower()
    cleaned_text = URL_PATTERN.sub("", cleaned_text)
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = NON_ALPHA_PATTERN.sub("", cleaned_text)
    cleaned_text = MULTISPACE_PATTERN.sub(" ", cleaned_text).strip()

    tokens = cleaned_text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)


def calculate_sentiment(text: object, analyzer: SentimentIntensityAnalyzer) -> float:
    return float(analyzer.polarity_scores(str(text))["compound"])


def build_text_hash(title: object, selftext: object) -> str:
    combined_text = f"{str(title).strip()} || {str(selftext).strip()}"
    return hashlib.sha256(combined_text.encode("utf-8")).hexdigest()


def resolve_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["post_id", "subreddit", "timestamp", "title", "selftext", "score", "num_comments", "author"]
    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["selftext"] = df["selftext"].fillna("").astype(str)
    df["author"] = df["author"].fillna("").astype(str)
    return df


def deduplicate_posts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    df["title_selftext_key"] = df["title"] + " || " + df["selftext"]

    duplicate_post_id_rows = int(df["post_id"].duplicated().sum())

    df = df.drop_duplicates(subset=["post_id"], keep="first")
    rows_after_post_id_dedup = len(df)
    duplicate_title_selftext_rows = int(df.duplicated(subset=["title_selftext_key"]).sum())
    df = df.drop_duplicates(subset=["title_selftext_key"], keep="first").drop(columns=["title_selftext_key"])

    removal_summary = {
        "duplicate_post_id_rows_removed": duplicate_post_id_rows,
        "rows_after_post_id_dedup": rows_after_post_id_dedup,
        "duplicate_title_selftext_rows_removed": duplicate_title_selftext_rows,
    }
    return df.reset_index(drop=True), removal_summary


def prepare_text_features(df: pd.DataFrame, stop_words: set[str]) -> pd.DataFrame:
    cleaned_texts = [
        clean_text(text, stop_words) for text in tqdm(df["selftext"], desc="Cleaning text", total=len(df))
    ]
    df["selftext_cleaned"] = cleaned_texts

    logging.info("Calculating word counts...")
    df["word_count"] = pd.Series(cleaned_texts, index=df.index).str.split().str.len()
    return df


def add_sentiment_scores(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    sentiment_scores = [
        calculate_sentiment(text, analyzer)
        for text in tqdm(df["selftext"], desc="Scoring sentiment", total=len(df))
    ]
    df["sentiment_score"] = sentiment_scores
    return df


def write_dataset_summary(
    summary_path: Path,
    *,
    rows_before_qa: int,
    rows_after_qa: int,
    rows_after_length_filter: int,
    dropped_short_posts: int,
    removal_summary: dict[str, int],
    df_final: pd.DataFrame,
) -> None:
    timestamp_series = pd.to_datetime(df_final["timestamp"], errors="coerce")
    summary_rows: list[dict[str, object]] = [
        {"section": "rows", "metric": "rows_before_qa", "value": rows_before_qa},
        {"section": "rows", "metric": "rows_after_qa", "value": rows_after_qa},
        {"section": "rows", "metric": "rows_after_length_filter", "value": rows_after_length_filter},
        {"section": "rows", "metric": "dropped_short_posts", "value": dropped_short_posts},
        {
            "section": "duplicates",
            "metric": "duplicate_post_id_rows_removed",
            "value": removal_summary["duplicate_post_id_rows_removed"],
        },
        {
            "section": "duplicates",
            "metric": "duplicate_title_selftext_rows_removed",
            "value": removal_summary["duplicate_title_selftext_rows_removed"],
        },
        {"section": "missing", "metric": "missing_selftext", "value": int(df_final["selftext"].isna().sum())},
        {
            "section": "missing",
            "metric": "missing_selftext_cleaned",
            "value": int(df_final["selftext_cleaned"].isna().sum()),
        },
        {"section": "length", "metric": "word_count_min", "value": int(df_final["word_count"].min())},
        {"section": "length", "metric": "word_count_median", "value": float(df_final["word_count"].median())},
        {"section": "length", "metric": "word_count_mean", "value": round(float(df_final["word_count"].mean()), 2)},
        {
            "section": "dates",
            "metric": "timestamp_min",
            "value": timestamp_series.min().isoformat() if not timestamp_series.dropna().empty else "",
        },
        {
            "section": "dates",
            "metric": "timestamp_max",
            "value": timestamp_series.max().isoformat() if not timestamp_series.dropna().empty else "",
        },
    ]

    for label, count in df_final["label"].value_counts().sort_index().items():
        summary_rows.append({"section": "labels", "metric": f"label_count_{label}", "value": int(count)})

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def main() -> None:
    logging.info("Starting Data Extraction & Cleaning Pipeline...")
    scraper = PullPushScraper()

    logging.info("Fetching MDD Classes...")
    mdd_1 = scraper.fetch_posts("depression", limit=2500)
    df_mdd_1 = pd.DataFrame(mdd_1)
    df_mdd_1["label"] = "Moderate MDD"

    mdd_2 = scraper.fetch_posts("SuicideWatch", limit=2500)
    df_mdd_2 = pd.DataFrame(mdd_2)
    df_mdd_2["label"] = "Severe Ideation"

    df_mdd = pd.concat([df_mdd_1, df_mdd_2], ignore_index=True)
    logging.info("Total Extracted MDD Posts: %s", df_mdd.shape[0])

    logging.info("Fetching Control Class (Class 0)...")
    control = scraper.fetch_posts("CasualConversation", limit=5000)
    df_control = pd.DataFrame(control)
    df_control["label"] = "Control"
    logging.info("Total Extracted Control Posts: %s", df_control.shape[0])

    df = pd.concat([df_mdd, df_control], ignore_index=True)
    df = ensure_required_columns(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    rows_before_qa = len(df)
    df, removal_summary = deduplicate_posts(df)
    rows_after_qa = len(df)
    df["text_hash"] = [build_text_hash(title, selftext) for title, selftext in zip(df["title"], df["selftext"])]

    logging.info(
        "QA deduplication removed %s duplicate post_id rows and %s duplicate title+selftext rows.",
        removal_summary["duplicate_post_id_rows_removed"],
        removal_summary["duplicate_title_selftext_rows_removed"],
    )

    base_dir = resolve_base_dir()
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "reddit_raw.csv"

    df.to_csv(raw_path, index=False)
    logging.info("Saved Raw Dataset to %s", raw_path)

    logging.info("Applying Text Cleaning (Regex, Stopwords) to selftext...")
    stop_words = set(stopwords.words("english"))
    df = prepare_text_features(df, stop_words)

    df_clean = df[df["word_count"] >= MIN_WORD_COUNT].copy()
    dropped = df.shape[0] - df_clean.shape[0]
    logging.info("Dropped %s overly short/empty posts. Final count: %s", dropped, df_clean.shape[0])

    logging.info("Calculating Baseline Sentiment Scores (VADER)...")
    analyzer = SentimentIntensityAnalyzer()
    df_clean = add_sentiment_scores(df_clean, analyzer)

    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_path = processed_dir / "reddit_mdd_cleaned.csv"
    summary_path = processed_dir / DATASET_SUMMARY_FILENAME
    df_clean.to_csv(clean_path, index=False)
    write_dataset_summary(
        summary_path,
        rows_before_qa=rows_before_qa,
        rows_after_qa=rows_after_qa,
        rows_after_length_filter=len(df_clean),
        dropped_short_posts=dropped,
        removal_summary=removal_summary,
        df_final=df_clean,
    )

    logging.info("Saved Processed Dataset to %s", clean_path)
    logging.info("Saved Dataset Summary to %s", summary_path)
    logging.info("Pipeline Execution Completed Successfully.")


if __name__ == "__main__":
    main()
