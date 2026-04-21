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
- **QA hardening:** duplicate `post_id` rows are removed first, followed by exact duplicate `title+selftext` rows.
- **Leakage-aware identifier:** a deterministic `text_hash` is created from `title+selftext`.
- **Normalization:** lowercase conversion.
- **Regex cleaning:** URL removal, newline stripping, non-alphabet filtering, and multi-space collapsing.
- **Stop words:** NLTK English stopword filtering.
- **Retention rule:** posts with fewer than 5 cleaned words are dropped.
- **Outputs:** `data/raw/reddit_raw.csv`, `data/processed/reddit_mdd_cleaned.csv` (yielding a final hardened dataset of 9,607 rows), and `data/processed/dataset_summary.csv`.

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
- **Track A (Sparse Baseline — Classical AI):** A **TF-IDF Vectorizer** captured the top 5,000 predictive unigrams and bigrams. This is now paired with two sparse baselines:
  - **Logistic Regression** with balanced class weights
  - **LinearSVC** with balanced class weights
- **Track B (Dense Text Modeling — Deep AI):** Text sequences passed through HuggingFace's **TwitterRoBERTa** (`cardiffnlp/twitter-roberta-base`, successfully resolving previous HuggingFace 401 Gated Repository restrictions). We extracted 768-dimensional dense representations using **batched PyTorch iteration with TQDM acceleration**, and fed them into a **Random Forest Classifier** (100 estimators).

### 2.3 Execution Profile
- The official dense-model path is **Google Colab with a T4 GPU**, which keeps the full processed dataset for TwitterRoBERTa feature extraction.
- The notebook auto-detects CUDA and selects a larger embedding batch size on T4 hardware.
- For local CPU-only runs, it subsamples to 2,000 rows to keep execution practical.

### 2.4 Evaluation Upgrades
The current notebook now strengthens the evidence beyond one train/test split:
- **Fixed 80/20 stratified holdout** for demo confusion matrices
- **5-fold, 3-repeat `RepeatedStratifiedKFold`** for mean ± std metric reporting
- **Permutation test** for the TF-IDF + Logistic Regression baseline
- **Learning curve** for the main sparse baseline
- **Error analysis export** for representative holdout mistakes
- **Model-card preprocessing** for TwitterRoBERTa, replacing usernames with `@user` and links with `http`

The notebook exports synchronized evaluation artifacts to `data/processed/`, with `results_summary.csv` as the intended metrics source of truth after each run.

### 2.5 Current Committed Evaluation Snapshot
The latest Colab T4 execution produced the following committed metrics:
- **Repeated CV**
  - `TF-IDF + Logistic Regression`: accuracy `0.7762 ± 0.0100`, macro F1 `0.7251 ± 0.0109`
  - `TF-IDF + LinearSVC`: accuracy `0.7616 ± 0.0077`, macro F1 `0.7059 ± 0.0096`
  - `TwitterRoBERTa + Random Forest`: accuracy `0.7600 ± 0.0071`, macro F1 `0.6955 ± 0.0085`
- **Holdout**
  - `TF-IDF + Logistic Regression`: accuracy `0.7841`, macro F1 `0.7355`
  - `TF-IDF + LinearSVC`: accuracy `0.7700`, macro F1 `0.7175`
  - `TwitterRoBERTa + Random Forest`: accuracy `0.7596`, macro F1 `0.6969`
- **Permutation test**
  - `TF-IDF + Logistic Regression`: macro F1 `0.7221`, `p = 0.032258`

---

## Phase 3: Project Automation

To fulfill the automation requirement, the repo supports two update paths:
- **Primary path:** `.github/workflows/quarterly_update.yml` runs on the 1st of January, April, July, and October at 00:00 UTC and commits refreshed data if anything changed.
- **Local fallback:** `src/quarterly_updater.py` schedules the pipeline every 90 days using the `schedule` library.

---

## Phase 4: Production Dashboard UI

### 4.1 Live Inference Application
An end-to-end `streamlit` web application (`app.py`) is deployed for live academic demonstration of the saved `joblib` artifacts.
- Automatically processes clinical text inputs identically to the training-time preprocessing pipeline.
- Implements a dynamic, professional clinical dashboard UI. 
- Employs **Plotly** to render interactive XAI Log-Odds visual vectors and static JSON-derived benchmark metrics.

### 4.2 State Management and Theming
- Native Streamlit widgets have been overwritten using Python-injected CSS styling to build custom switches (including a dynamic dark mode knob).
- Elements are responsive and utilize `st.session_state` callbacks to seamlessly switch data sources or visual layers without requiring hard refreshes.

---

## 4. Archival Outputs
Generated assets serving the downstream project operations:
- `data/raw/reddit_raw.csv` — the raw scrape output, targeting 10,000 posts before post-cleaning filters.
- `data/processed/reddit_mdd_cleaned.csv` — the processed dataset with metadata, labels, cleaned text, `word_count`, `sentiment_score`, and `text_hash`.
- `data/processed/dataset_summary.csv` — a compact QA summary covering duplicate removal, row counts, label counts, missingness, length stats, and date range.
- `data/processed/results_summary.csv` — the synchronized metrics export covering repeated CV, holdout metrics, and the permutation-test result.
- `data/processed/error_analysis_holdout.csv` — representative holdout misclassifications with explanation notes.
- `data/processed/top_tokens_by_class.csv` — top sparse-model tokens exported for interpretation.
- `notebooks/02_text_classification_models.ipynb` — the current classification, CV, SHAP, error analysis, and EDA notebook.
- `notebooks/Assignment_1_PRAW_Extraction.ipynb` — a legacy notebook from the original PRAW-based assignment plan, retained for coursework history rather than the production PullPush pipeline.
