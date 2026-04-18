# Workflow Documentation: End-to-End Project Pipeline

**Deliverables for Assignment 1 & 2**  
**Task ID:** HDA-4 | **Group:** 4  
**Title:** Natural Language Processing of Social Media for MDD Symptom Identification

This document outlines the current end-to-end workflow implemented in the repository. The original assignment brief started with a simpler Reddit extraction and binary-classification idea, but the live project has evolved into a three-class severity pipeline with updated modeling, EDA, and automation.

---

## Phase 1: Data Architecture & Preprocessing

### 1.1 Data Source Selection
The team initially evaluated Reddit's official API (`praw`) and then adopted **PullPush.io** for the implemented pipeline.
- **Reason for the pivot:** no authentication requirement, historical retrieval, and less operational friction for an academic course project.
- **Current production source in code:** PullPush, via `src/scraper.py`.

### 1.2 Scraping Methodology
We implemented a `PullPushScraper` class in `src/scraper.py` using the `requests` library.
1. **Target subreddits and labels**
   - `r/depression` → `Moderate MDD` (target 2,500 posts)
   - `r/SuicideWatch` → `Severe Ideation` (target 2,500 posts)
   - `r/CasualConversation` → `Control` (target 5,000 posts)
2. **Pagination strategy**
   - PullPush returns at most 100 submissions per request.
   - The scraper walks backward in time using the `before=EPOCH_TIMESTAMP` parameter.
3. **Filtering at source**
   - Posts with `selftext` equal to `[removed]`, `[deleted]`, or an empty string are skipped.
4. **Reliability controls**
   - Three request retries are attempted on failure.
   - A one-second delay is inserted between successful requests.

### 1.3 Data Cleaning Pipeline
The preprocessing pipeline lives in `src/pipeline.py`.
- **Normalization:** lowercase conversion.
- **Regex cleaning:** URL removal, newline stripping, non-alphabet filtering, and multi-space collapsing.
- **Stop words:** NLTK English stopword filtering.
- **Retention rule:** posts with fewer than 5 cleaned words are dropped.
- **Outputs:** `data/raw/reddit_raw.csv` and `data/processed/reddit_mdd_cleaned.csv`.

### 1.4 Feature Engineering
We compute a baseline sentiment feature using **VADER** on the original uncleaned `selftext`.
- This preserves punctuation and casing cues that would be weakened by preprocessing.
- The processed CSV also stores `word_count` and keeps metadata such as `author`, `score`, and `num_comments`.

---

## Phase 2: Classification Models & Exploratory Analysis

### 2.1 Problem Framing and EDA
The current notebook treats the task as **three-class severity classification**:
1. `Control`
2. `Moderate MDD`
3. `Severe Ideation`

To study symptom and emotional language patterns, the notebook runs six exploratory analyses:
1. **DSM-5 symptom keyword frequency**
2. **Word clouds**
3. **VADER sentiment distribution**
4. **Post length analysis**
5. **Top bigrams**
6. **SHAP explainability for the sparse baseline**

`ROC` plots were part of an older binary framing and were dropped in the final multi-class notebook.

### 2.2 Dual-Track Machine Learning Pipeline
To evaluate predictive configurations, we designed a complementary experimental track architecture:
- **Track A (Sparse Baseline — Classical AI):** A **TF-IDF Vectorizer** captured the top 5,000 predictive unigrams and bigrams. This was paired with a **Logistic Regression** classifier operating on balanced class weights.
- **Track B (Dense Text Modeling — Deep AI):** Text sequences passed through HuggingFace's **TwitterRoBERTa** (`cardiffnlp/twitter-roberta-base`). We extracted 768-dimensional dense representations and fed them into a **Random Forest Classifier** (100 estimators).

*Performance conclusion:* The classical TF-IDF model outperformed the transformer embedding track on the current three-class setup:
- **TF-IDF + Logistic Regression:** 78.7% accuracy
- **TwitterRoBERTa + Random Forest:** 74.2% accuracy

See `docs/methods_and_results.md` for the full result summary.

### 2.3 Execution Profile
- The notebook can use CUDA when available.
- For local CPU-only runs, it subsamples to 2,000 rows to keep execution practical.

---

## Phase 3: Project Automation

To fulfill the automation requirement, the repo supports two update paths:
- **Primary path:** `.github/workflows/quarterly_update.yml` runs on the 1st of January, April, July, and October at 00:00 UTC and commits refreshed data if anything changed.
- **Local fallback:** `src/quarterly_updater.py` schedules the pipeline every 90 days using the `schedule` library.

---

## 4. Archival Outputs
Generated assets serving the downstream project operations:
- `data/raw/reddit_raw.csv` — the raw scrape output, targeting 10,000 posts before post-cleaning filters.
- `data/processed/reddit_mdd_cleaned.csv` — the processed dataset with metadata, labels, cleaned text, `word_count`, and `sentiment_score`.
- `notebooks/02_text_classification_models.ipynb` — the current classification, EDA, and SHAP notebook.
- `notebooks/Assignment_1_PRAW_Extraction.ipynb` — a legacy notebook from the original PRAW-based assignment plan, retained for coursework history rather than the production PullPush pipeline.
