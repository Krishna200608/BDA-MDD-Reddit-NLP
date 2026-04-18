# Project Context — BDA: Reddit MDD NLP Corpus

> **Living Document** — Updated throughout the project lifecycle.  
> Last Updated: 2026-04-18

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Course** | Big Data Analytics (BDA) — 6th Semester |
| **Institution** | Indian Institute of Information Technology, Allahabad (IIIT-A) |
| **Professor** | Prof. Sonali Agarwal |
| **Task ID** | HDA-4 |
| **Title** | Natural Language Processing of Social Media for MDD Symptom Identification (Reddit MDD Corpus) |
| **Repo** | `Krishna200608/BDA-MDD-Reddit-NLP` |

## 2. Team

| Name | Roll Number | Role |
|---|---|---|
| Krishna Sikheriya | IIT2023139 | — |
| Priyam Jyoti Chakrabarty | IIT2023147 | — |
| Tavish Chawla | IIT2023150 | — |

## 3. Objective

- Compile a dataset of self-reported **Major Depressive Disorder (MDD)** posts from Reddit for NLP-based analysis.
- **ML Goal**: Classify posts into 3 severity tiers (Control, Moderate MDD, Severe Ideation); detect symptom and emotional language patterns.

## 4. Target Subreddits

| Subreddit | Purpose | Label |
|---|---|---|
| `r/depression` | MDD-related posts | Moderate MDD |
| `r/SuicideWatch` | Severe risk posts | Severe Ideation |
| `r/CasualConversation` | Neutral/control posts | Control |

## 5. Assignments & Deliverables

### Assignment 1 — Data Extraction
| Deliverable | Format | Status |
|---|---|---|
| Secondary dataset (post IDs, dates, cleaned text, labels) | `.csv` | ✅ Complete |
| Python Script for Reddit scraping & cleaning | `.py` | ✅ Complete |
| Documentation of workflow | `.md` | ✅ Complete |

### Assignment 2 — Classification & Pipeline (Current) ⬅️
| Deliverable | Format | Status |
|---|---|---|
| Python Notebook for text classification models | `.ipynb` | ✅ Complete |
| EDA & Language Pattern Detection (symptom keywords, word clouds, sentiment, bigrams, SHAP) | `.ipynb` | ✅ Complete |
| Documentation of methods and results | `.md` | ✅ Complete |
| Automated pipeline for quarterly updates | `.py` | ✅ Complete |

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Environment | `.venv` locally + Google Colab (T4 GPU recommended for official notebook reproduction) |
| API | PullPush / Pushshift API (No authentication required) |
| Requests | `requests` (Python HTTP library) |
| NLP | NLTK, regex, VADER Sentiment, wordcloud |
| Embeddings | TwitterRoBERTa (`cardiffnlp/twitter-roberta-base`) |
| ML | scikit-learn (Assignment 2) |
| Data | pandas, numpy |
| Version Control | Git + GitHub |

## 7. Data Schema (Target CSV)

| Column | Type | Description |
|---|---|---|
| `post_id` | `str` | Unique Reddit post ID |
| `subreddit` | `str` | Source subreddit name |
| `timestamp` | `datetime` | UTC time of post creation |
| `title` | `str` | Post title |
| `selftext` | `str` | Raw Reddit post body |
| `score` | `int` | Post upvote score |
| `num_comments` | `int` | Number of comments |
| `author` | `str` | Reddit username captured in the scrape |
| `label` | `str` | `Control`, `Moderate MDD`, or `Severe Ideation` |
| `selftext_cleaned` | `str` | Lowercased, regex-cleaned, stopwords removed |
| `word_count` | `int` | Word count of cleaned text |
| `sentiment_score` | `float` | VADER compound sentiment score |
| `text_hash` | `str` | Deterministic hash of `title+selftext` for QA and leakage-aware analysis |

## 8. Project Directory Structure (Current)

```
BDA-MDD-Reddit-NLP/
├── data/
│   ├── raw/                    # Raw scraped data (gitignored if large)
│   └── processed/              # Cleaned, labeled CSVs + QA/evaluation artifacts
│       ├── reddit_mdd_cleaned.csv
│       ├── dataset_summary.csv
│       ├── results_summary.csv
│       ├── error_analysis_holdout.csv
│       └── top_tokens_by_class.csv
├── notebooks/
│   ├── Assignment_1_PRAW_Extraction.ipynb    # Legacy notebook from the original PRAW plan
│   └── 02_text_classification_models.ipynb     # TF-IDF, TwitterRoBERTa, SHAP
├── src/
│   ├── scraper.py              # PullPush API scraper using requests
│   ├── pipeline.py             # Main Extraction & Cleaning pipeline
│   └── quarterly_updater.py    # Local automation fallback using schedule
├── docs/
│   ├── assignments/
│   │   └── Our_Project_Task.md # Original grading criteria
│   ├── assets/
│   │   ├── Reddit_Proxy_API.pdf
│   │   └── reddit_api_project_brief.md
│   ├── methods_and_results.md  # Final Evaluation Document
│   ├── workflow.md             # Workflow documentation (deliverable)
│   └── team_work_division.md   # Group work allocation cheat sheet
├── .github/
│   └── workflows/
│       └── quarterly_update.yml # CI/CD: auto-refresh dataset every quarter
├── .env.example
├── .gitignore
├── Context.md                  # This file
├── README.md                   # Professional repo README
└── requirements.txt            # pip dependencies
```

## 9. API Credentials Status

| Item | Status |
|---|---|
| PullPush API keys | **Not Required** |
| Reddit API keys | Not required for the current committed pipeline; only a historical backup idea from the early project stage |

## 10. Key Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-02 | Project initiated | Assignment 1 released by Prof. Sonali Agarwal |
| 2026-04-02 | Reddit API access requested | Needed for PRAW-based data extraction |
| 2026-04-06 | Transformer hardware optimization | Dynamic GPU (Colab) scaling implemented to speed up embedding generation |
| 2026-04-06 | Colab Github Auth Secret Sync | Integrated dynamic repo sync to fix absolute filepath breakages in cloud rendering |
| 2026-04-06 | Quarterly automation → GitHub Actions | Replaced local `schedule` daemon with CI/CD cron workflow for zero-maintenance refresh |
| 2026-04-07 | EDA & Language Pattern Detection section | Added 6 analyses: DSM-5 symptom keyword frequency, word clouds, VADER sentiment distribution, post length distribution, top bigrams, SHAP explainability |
| 2026-04-10 | Tertiary Severity Mapping | Upgraded labeling from binary to Control vs. Moderate vs. Severe |
| 2026-04-10 | Gated Repo Fix (TwitterRoBERTa) | Switched from mental-roberta to cardiffnlp/twitter-roberta-base to prevent 401 Gated Error on Colab |
| 2026-04-10 | PyTorch TQDM Acceleration | Rewrote embedding generation from apply() to batched iteration |
| 2026-04-18 | Dataset QA Hardening | Added duplicate removal, `text_hash`, and `dataset_summary.csv` generation to reduce leakage risk |
| 2026-04-18 | Evaluation Upgrade | Added repeated CV, LinearSVC baseline, permutation test, learning curve, and holdout error analysis |
| 2026-04-18 | Final Colab T4 Evaluation Run | Generated `results_summary.csv`, `error_analysis_holdout.csv`, and `top_tokens_by_class.csv` from the completed notebook run |

## 11. Latest Snapshot

| Item | Value |
|---|---|
| **Rows before QA** | `9,800` |
| **Rows after QA / modeling snapshot** | `9,607` |
| **Duplicate `post_id` rows removed** | `145` |
| **Exact duplicate `title+selftext` rows removed** | `48` |
| **Date range** | `2025-03-05T06:40:41` → `2025-05-19T18:11:58` |
| **Labels** | `Control` 4,903 · `Moderate MDD` 2,408 · `Severe Ideation` 2,296 |
| **Best repeated-CV model** | `TF-IDF + Logistic Regression` — accuracy `0.7762 ± 0.0100`, macro F1 `0.7251 ± 0.0109` |
| **Best holdout model** | `TF-IDF + Logistic Regression` — accuracy `0.7841`, macro F1 `0.7355` |
| **Permutation test** | macro F1 `0.7221`, `p = 0.032258` |
| **Generated artifacts** | `dataset_summary.csv`, `results_summary.csv`, `error_analysis_holdout.csv`, `top_tokens_by_class.csv` |

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PullPush / proxy availability changes | Can interrupt future refreshes | Keep the scraper resilient with retries and preserve committed dataset snapshots |
| Proxy rate limiting or unstable responses | Slower data collection / partial refreshes | Retry handling, polite delays, and rerunning the pipeline when needed |
| Subreddit posts removed/deleted | Incomplete dataset | Collect surplus data, document exclusions |
| Three-class label skew | Can bias per-class performance, especially for Moderate vs Severe confusion | Stratified splitting, macro-F1 reporting, and per-class precision/recall |

## 13. References

- [PullPush API Brief](docs/assets/reddit_api_project_brief.md)
- [Reddit API Terms](https://www.reddit.com/wiki/api/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [TwitterRoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base)

---

> **Note**: This is the living project context document. For the latest committed evaluation metrics, use `data/processed/results_summary.csv` as the metrics source of truth.
