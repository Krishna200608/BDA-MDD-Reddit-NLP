from __future__ import annotations

import logging
import re
import warnings
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


def resolve_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

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
    df_clean.to_csv(clean_path, index=False)

    logging.info("Saved Processed Dataset to %s", clean_path)
    logging.info("Pipeline Execution Completed Successfully.")


if __name__ == "__main__":
    main()
