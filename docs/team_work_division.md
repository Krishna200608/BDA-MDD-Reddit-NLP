# Team Work Division — Evaluation Cheat Sheet

**Course:** Big Data Analytics (BDA) · 6th Semester  
**Task ID:** HDA-4 · **Group:** 4  
**Repository:** [Krishna200608/BDA-MDD-Reddit-NLP](https://github.com/Krishna200608/BDA-MDD-Reddit-NLP)

---

## Part A: Assignment 1 — Data Extraction

---

### Krishna Sikheriya (IIT2023139) — Data Collection & API Integration

**What I built:**
- Researched available Reddit data sources — evaluated official Reddit API (PRAW) vs. PullPush.io archive.
- Chose PullPush because the official API requires manual developer approval and has restrictive rate limits; PullPush provides unrestricted academic-grade access to historical data.
- Wrote `src/scraper.py` — the core `PullPushScraper` class that handles all HTTP communication with the PullPush API.

**Key technical details to explain:**
- The API returns max 100 posts per request, so we paginate using a `before=EPOCH_TIMESTAMP` parameter, requesting progressively older posts.
- Built-in **retry logic** (3 retries with exponential backoff) to handle transient network failures.
- Filtering at source: we skip posts where `selftext` is `[removed]` or `[deleted]` to avoid polluting the dataset.
- Added a 1-second `time.sleep()` between requests to be respectful to the API server.
- Used `tqdm` for real-time progress tracking during extraction.

**Files I can point to:**
- `src/scraper.py` (lines 10–74 — the `PullPushScraper` class)
- `.env.example` — environment variable template

**If the TA asks: "Why not use the official Reddit API?"**
> "We initially planned to use PRAW, but Reddit's developer application approval was delayed. PullPush.io is a widely-used academic proxy for Pushshift data that doesn't require authentication. We documented this pivot decision in `docs/workflow.md`."

---

### Priyam Jyoti Chakrabarty (IIT2023147) — Text Preprocessing & NLP Pipeline

**What I built:**
- Wrote the `clean_text()` function and the full preprocessing pipeline in `src/pipeline.py`.
- Designed the text cleaning ladder: lowercase → URL removal → newline stripping → non-alphabet removal → multi-space collapse → stopword filtering.
- Integrated NLTK's English stopword list to strip noise before vectorization.
- Added a minimum word-count filter (≥ 5 words after cleaning) to drop empty or near-empty posts that would add noise to the model.

**Key technical details to explain:**
- Regex patterns used:
  - `http\S+|www\.\S+` — removes all URLs (common in Reddit posts).
  - `[^a-z\s]` — removes everything except lowercase letters and spaces.
  - `\s+` — collapses multiple spaces into one.
- Stopword removal: used NLTK's built-in English corpus (179 words like "the", "is", "at", etc.) to increase signal-to-noise ratio for downstream TF-IDF.
- Used `pandas.progress_apply()` with `tqdm` for visual progress tracking during cleaning of 10,000 rows.

**Files I can point to:**
- `src/pipeline.py` (lines 22–39 — `clean_text()` function)
- `src/pipeline.py` (lines 78–89 — cleaning and word count logic)

**If the TA asks: "Why clean text before saving?"**
> "Raw Reddit text contains URLs, markdown formatting, and special characters that add noise. Cleaning before saving ensures anyone who downloads the processed CSV can directly use it for model training without repeating the preprocessing."

---

### Tavish Chawla (IIT2023150) — Feature Engineering, Validation & Documentation

**What I built:**
- Implemented the VADER sentiment analysis layer — computes the `sentiment_score` (compound polarity) for every post using the uncleaned `selftext` so we don't lose emotional punctuation cues.
- Orchestrated the full pipeline flow: data collection → raw CSV export → cleaning → feature engineering → processed CSV export.
- Added the QA hardening layer: duplicate removal, `text_hash`, and `dataset_summary.csv` generation before modeling.
- Validated the final dataset structure: checked schema, class distribution, and processed-output integrity after filtering. The current committed snapshot is three-class and securely deduplicated to a final 9,607 rows.
- Wrote all documentation: `docs/workflow.md`, the Assignment 1 notebook walkthrough (`notebooks/Assignment_1_PRAW_Extraction.ipynb`).
- Set up the project structure: `requirements.txt`, `.gitignore`, directory layout.

**Key technical details to explain:**
- VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based sentiment tool specifically tuned for social media text. It outputs a compound score from -1 (very negative) to +1 (very positive).
- We feed the **original uncleaned text** to VADER (not the cleaned version) because VADER relies on capitalization, punctuation, and emojis for sentiment intensity — cleaning would destroy those signals.
- The processed CSV currently has 13 columns: `post_id`, `subreddit`, `timestamp`, `title`, `selftext`, `score`, `num_comments`, `author`, `label`, `selftext_cleaned`, `word_count`, `sentiment_score`, `text_hash`.

**Files I can point to:**
- `src/pipeline.py` (lines 91–98 — VADER sentiment scoring)
- `src/pipeline.py` (lines 41–110 — full `main()` orchestration)
- `docs/workflow.md` (entire file)
- `notebooks/Assignment_1_PRAW_Extraction.ipynb`

**If the TA asks: "Why use VADER and not a transformer for sentiment?"**
> "VADER is lightweight, requires no training, and is specifically designed for social media text. It serves as a baseline feature, while the Assignment 2 classification notebook uses TwitterRoBERTa embeddings for the dense model track."

---
---

## Part B: Full Project (Assignment 1 + Assignment 2)

---

### Krishna Sikheriya (IIT2023139) — Data Engineering & Cloud Infrastructure

| Area | Contribution |
|:---|:---|
| **Assignment 1** | Wrote `src/scraper.py` — PullPush API client with pagination, retries, and rate limiting |
| **Assignment 2** | Set up GitHub/Colab integration — `GITHUB_TOKEN` secret, repo cloning, identity config |
| **Streamlit UI (Frontend)** | Wrote `app.py` and `src/dashboard_utils.py`, integrating the NeuroFetal-AI matching Dark Mode CSS architecture and layout logic. |
| **CI/CD** | Created `.github/workflows/quarterly_update.yml` — GitHub Actions for automated quarterly data refresh |
| **Infrastructure** | Repository setup, Git workflow, environment configuration |

**Key talking points:**
- "I handled the data acquisition layer and cloud deployment. The scraper uses paginated HTTP requests with retry logic. For Assignment 2, I integrated the notebook with GitHub so it can pull the latest code and push results directly from Colab."
- "The quarterly automation runs via GitHub Actions on a cron schedule (Jan, Apr, Jul, Oct) — no local machine needs to be running."

---

### Priyam Jyoti Chakrabarty (IIT2023147) — NLP & Machine Learning Models

| Area | Contribution |
|:---|:---|
| **Assignment 1** | Wrote `clean_text()` — regex-based preprocessing, stopword removal, word count filtering |
| **Assignment 2 — Model A** | TF-IDF vectorizer (5,000 features, unigrams + bigrams) + Logistic Regression with balanced class weights |
| **Assignment 2 — Model A2** | TF-IDF vectorizer + LinearSVC comparison baseline |
| **Assignment 2 — Model B** | TwitterRoBERTa embeddings (768-dim dense vectors) + Random Forest classifier (100 estimators) |
| **Assignment 2 — EDA** | DSM-5 symptom keyword analysis, word clouds, sentiment distribution, post length analysis, top bigrams, SHAP explainability |
| **Hardware** | Implemented dynamic GPU/CPU detection (full dataset on T4 GPU, 2k subset on CPU) and upgraded text-embedding generation to PyTorch batched iteration with TQDM progress tracking |
| **Streamlit UI (Data & XAI)** | Wrote `src/inference.py`, designed the interactive Plotly rendering logic for the dashboard, and mapped the saved model artifacts back to live benchmarks. |

**Key talking points:**
- "I built the classification comparison stack for the final three-class severity problem: TF-IDF + Logistic Regression, TF-IDF + LinearSVC, and TwitterRoBERTa + Random Forest."
- "We shifted to `cardiffnlp/twitter-roberta-base` because it perfectly matches informal online language and it bypassed the HuggingFace "401 Gated Repo" restrictions that were breaking our automated Colab pipeline."
- "The notebook now reports both a fixed holdout split and a stronger repeated cross-validation summary, so the results are easier to defend in front of the instructor."
- "The notebook auto-detects hardware: if a CUDA GPU is available, it evaluates the full processed dataset; otherwise it subsamples to 2,000 rows for local CPU practicality."
- "I also implemented the EDA section with six analyses plus SHAP, a learning curve, a permutation test, and an explicit error-analysis export."

---

### Tavish Chawla (IIT2023150) — Documentation, Evaluation & Automation

| Area | Contribution |
|:---|:---|
| **Assignment 1** | VADER sentiment scoring, data validation, `docs/workflow.md`, Assignment 1 notebook |
| **Assignment 2** | Wrote `docs/methods_and_results.md` — full evaluation report with accuracy, precision, recall, F1, EDA findings |
| **Automation** | Wrote `src/quarterly_updater.py` — local daemon using `schedule` library (90-day loop) |
| **Documentation** | README.md (architecture diagram, results table, EDA summary, usage guide), Context.md |

**Key talking points:**
- "I handled all evaluation and documentation. The methods document now explains the upgraded protocol: duplicate-safe data QA, fixed holdout evaluation, repeated cross-validation, permutation testing, learning curve analysis, and holdout error analysis."
- "The report and README now point to generated artifacts such as `dataset_summary.csv`, `results_summary.csv`, `error_analysis_holdout.csv`, and `top_tokens_by_class.csv` so the latest notebook outputs stay synchronized."
- "For automation, we have two options: GitHub Actions (cloud-based, zero maintenance) and a local Python daemon using the `schedule` library. Both re-run the full pipeline every quarter to keep the dataset fresh."
- "The README includes a Mermaid architecture diagram that GitHub renders natively, showing the full data flow from scraping to evaluation and EDA."

---
---

## Quick Reference: What Each File Was Built By

| File | Owner |
|:---|:---|
| `src/scraper.py` | Krishna |
| `src/pipeline.py` (clean_text, preprocessing) | Priyam |
| `src/pipeline.py` (main orchestration, VADER) | Tavish |
| `src/quarterly_updater.py` | Tavish |
| `src/inference.py` | Priyam |
| `src/dashboard_utils.py` | Krishna |
| `app.py` | Krishna |
| `notebooks/Assignment_1_PRAW_Extraction.ipynb` | Tavish |
| `notebooks/02_text_classification_models.ipynb` (TF-IDF + LR) | Priyam |
| `notebooks/02_text_classification_models.ipynb` (TwitterRoBERTa + RF) | Priyam |
| `notebooks/02_text_classification_models.ipynb` (EDA & Language Pattern Detection) | Priyam |
| `notebooks/02_text_classification_models.ipynb` (Colab/Git setup) | Krishna |
| `.github/workflows/quarterly_update.yml` | Krishna |
| `docs/workflow.md` | Tavish |
| `docs/methods_and_results.md` | Tavish |
| `README.md` | Tavish |
| `docs/Context.md` | Tavish |
| `requirements.txt` / `.gitignore` / `.env.example` | Krishna |

---

> **Tip:** Don't memorize this word-for-word. Read your section once, understand *why* each decision was made, and explain it naturally in your own words. TAs are looking for understanding, not recitation.
