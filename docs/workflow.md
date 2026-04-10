# Workflow Documentation: End-to-End Project Pipeline

**Deliverables for Assignment 1 & 2**  
**Task ID:** HDA-4 | **Group:** 4  
**Title:** Natural Language Processing of Social Media for MDD Symptom Identification

This document outlines the complete technical workflow of the project, expanding upon the original data extraction pipeline to include exploratory data analysis, machine learning classification tracks, and automated dataset refreshing.

---

## Phase 1: Data Architecture & Preprocessing

### 1.1 Data Source Selection
Initially, Reddit's official API (`praw`) was considered. However, due to restrictive modern API limits and manual developer approval delays, we pivoted to **PullPush.io**, a recognized archival dataset for academic research.
- **Advantage:** Unrestricted rate limits, historical retrieval without API keys, access to deleted/removed records.

### 1.2 Scraping Methodology
We implemented a custom Python object (`PullPushScraper` in `src/scraper.py`) utilizing the `requests` library.
1. **Target Subreddits:**
   - **MDD Class (Class 1):** `r/depression` (2,500 posts), `r/SuicideWatch` (2,500 posts)
   - **Control Class (Class 0):** `r/CasualConversation` (5,000 posts)
2. **Pagination Strategy:** Since PullPush allows max 100 posts per request, we iteratively send requests using `before=EPOCH_TIMESTAMP`, extracting progressively older posts over time.
3. **Filtering at Source:** We excluded submission bodies that were exactly `[removed]` or `[deleted]` to ensure high data quality upfront.

### 1.3 Data Cleaning Pipeline
Conducted strictly through Python object orchestration within `src/pipeline.py`.
- **Normalization:** Lowercase conversion of all characters.
- **Regex Cleaning:** Removed all HTTP links, special symbols, newlines, and digits, leaving strictly `[a-z\s]`.
- **Stop Words:** Implemented NLTK's English stop words removal to increase the signal-to-noise ratio for TF-IDF / embeddings.
- **Length Filtering:** Automatically removed empty strings or documents containing fewer than 5 cleaned words.

### 1.4 Baseline Feature Engineering
We initialized a foundational sentiment analysis layer over the uncleaned text using **VADER** from `vaderSentiment`. This generates a `sentiment_score` reflecting dimensional polarity before transitioning to semantic modeling.

---

## Phase 2: Classification Models & Exploratory Analysis

### 2.1 Exploratory Data Analysis (EDA)
To identify language patterns signifying MDD distress, we deployed six analytical procedures over the corpus (detailed in `notebooks/02_text_classification_models.ipynb`):
1. **Symptom Keywords Analysis:** Comparing DSM-5 complaint term frequencies via heatmaps.
2. **Word Clouds:** Visualizing dominating lexical deviations between MDD and Control classes.
3. **Sentiment Distribution:** Using KDE plots to characterize the negative polarity shift in depressive texts.
4. **Post Length Analysis:** Measuring structural verbosity connected to text rumination trends.
5. **Top Bigrams Extraction:** Identifying frequent multi-word linguistic phrases of distress.
6. **ROC Validation:** Plotting comparative curves to evaluate the true/false positive tradeoffs.

### 2.2 Dual-Track Machine Learning Pipeline
To evaluate predictive configurations, we designed a complementary experimental track architecture:
- **Track A (Sparse Baseline — Classical AI):** A **TF-IDF Vectorizer** captured the top 5,000 predictive unigrams and bigrams. This was paired with a **Logistic Regression** classifier operating on balanced class weights.
- **Track B (Dense Text Modeling — Deep AI):** Text sequences passed through HuggingFace's **Bio_ClinicalBERT** (fine-tuned on clinical mimic-iii notes). We fed the continuous 768-dimensional `[CLS]` token vectors into a **Random Forest Classifier** (100 estimators).

*Performance Conclusion:* The classical TF-IDF model successfully outperformed deep embeddings (91.7% vs 85.8%), highlighting that Reddit's informal clinical slang is better captured by domain-specific keyword presence rather than rigorous academic embeddings. See `docs/methods_and_results.md`.

---

## Phase 3: Project Automation

To fulfill the continuous integration (CI) requirement, an automated dataset refresh protocol runs recursively:
- **Primary CI/CD Stack:** A GitHub Actions workflow (`.github/workflows/quarterly_update.yml`) spins up every 90 days to autonomously acquire newer Reddit posts and re-commit the fresh `reddit_mdd_cleaned.csv` to the repository.
- **Localized Daemon (Fallback):** In case of CI constraints, `src/quarterly_updater.py` acts as a static background scheduler utilizing the `schedule` python daemon to repeat extraction events securely. 

---

## 4. Archival Outputs
Generated assets serving the downstream project operations:
- `data/raw/reddit_raw.csv` — The original scraped baseline (~10K records).
- `data/processed/reddit_mdd_cleaned.csv` — The preprocessed NLP training matrix containing embeddings, labels, and scoring columns.
- `notebooks/02_text_classification_models.ipynb` — The executed visualization and deep-learning artifact notebook.
