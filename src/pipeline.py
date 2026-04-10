import os
import pandas as pd
import re
import nltk
import logging
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scraper import PullPushScraper
from tqdm import tqdm

# Connect tqdm with pandas for progress_apply
tqdm.pandas()

# Ensure NLTK resources are downloaded silently
import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text, stop_words):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove newlines
    text = text.replace('\n', ' ')
    # Remove non-alphabet (punctuation and numbers)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stop words removal
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def main():
    logging.info("Starting Data Extraction & Cleaning Pipeline...")
    scraper = PullPushScraper()

    # 1.1 Fetching MDD Classes (Moderate and Severe)
    logging.info("Fetching MDD Classes...")
    mdd_1 = scraper.fetch_posts('depression', limit=2500)
    df_mdd_1 = pd.DataFrame(mdd_1)
    df_mdd_1['label'] = 'Moderate MDD'

    mdd_2 = scraper.fetch_posts('SuicideWatch', limit=2500)
    df_mdd_2 = pd.DataFrame(mdd_2)
    df_mdd_2['label'] = 'Severe Ideation'

    df_mdd = pd.concat([df_mdd_1, df_mdd_2], ignore_index=True)
    logging.info(f"Total Extracted MDD Posts: {df_mdd.shape[0]}")

    # 1.2 Fetching Control Class (Class 0)
    logging.info("Fetching Control Class (Class 0)...")
    control = scraper.fetch_posts('CasualConversation', limit=5000)
    df_control = pd.DataFrame(control)
    df_control['label'] = 'Control'
    logging.info(f"Total Extracted Control Posts: {df_control.shape[0]}")

    # 1.3 Combine and Save Raw Dataset
    df = pd.concat([df_mdd, df_control], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    
    # Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, 'reddit_raw.csv')
    
    df.to_csv(raw_path, index=False)
    logging.info(f"Saved Raw Dataset to {raw_path}")

    # 2. Text Cleaning & Preprocessing
    logging.info("Applying Text Cleaning (Regex, Stopwords) to selftext...")
    stop_words = set(stopwords.words('english'))
    df['selftext_cleaned'] = df.progress_apply(lambda row: clean_text(row['selftext'], stop_words), axis=1)
    
    logging.info("Calculating word counts...")
    df['word_count'] = df['selftext_cleaned'].str.split().str.len()

    # Drop overly short posts
    df_clean = df[df['word_count'] >= 5].copy()
    dropped = df.shape[0] - df_clean.shape[0]
    logging.info(f"Dropped {dropped} overly short/empty posts. Final count: {df_clean.shape[0]}")

    # 3. Feature Engineering (Sentiment Score)
    logging.info("Calculating Baseline Sentiment Scores (VADER)...")
    analyzer = SentimentIntensityAnalyzer()
    
    # We feed original uncleaned text to VADER
    df_clean['sentiment_score'] = df_clean.progress_apply(
        lambda row: analyzer.polarity_scores(str(row['selftext']))['compound'], axis=1
    )

    # 4. Save Secondary Dataset
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    clean_path = os.path.join(processed_dir, 'reddit_mdd_cleaned.csv')
    df_clean.to_csv(clean_path, index=False)
    
    logging.info(f"Saved Processed Dataset to {clean_path}")
    logging.info("Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    main()
