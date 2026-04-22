# Project Context — BDA: Reddit MDD NLP Corpus

> **Living Document** — Updated throughout the project lifecycle.  
> Last Updated: 2026-04-02

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
- **ML Goal**: Classify posts as MDD vs. Control; detect symptom and emotional language patterns.

## 4. Target Subreddits

| Subreddit | Purpose | Label |
|---|---|---|
| `r/depression` | MDD-related posts | MDD |
| `r/SuicideWatch` | MDD-related posts | MDD |
| `r/mentalhealth` | MDD-related posts (filtered) | MDD |
| `r/CasualConversation` | Neutral/control posts | Control |

## 5. Assignments & Deliverables

### Assignment 1 — Data Extraction (Current) ⬅️
| Deliverable | Format | Status |
|---|---|---|
| Secondary dataset (post IDs, dates, cleaned text, labels) | `.csv` | ✅ Complete |
| Python Script for Reddit scraping & cleaning | `.py` | ✅ Complete |
| Documentation of workflow | `.md` | ✅ Complete |

### Assignment 2 — Classification & Pipeline (Future)
| Deliverable | Format | Status |
|---|---|---|
| Python Notebook for text classification models | `.ipynb` | ⬜ Not Started |
| Documentation of methods and results | `.md` | ⬜ Not Started |
| Automated pipeline for quarterly updates | `.py` | ⬜ Not Started |

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Environment | `.venv` (local virtual environment) |
| API | PullPush / Pushshift API (No authentication required) |
| Requests | `requests` (Python HTTP library) |
| NLP | NLTK, spaCy, regex |
| Embeddings | ClinicalBERT (future — Assignment 2) |
| ML | scikit-learn (future — Assignment 2) |
| Data | pandas, numpy |
| Version Control | Git + GitHub |

## 7. Data Schema (Target CSV)

| Column | Type | Description |
|---|---|---|
| `post_id` | `str` | Unique Reddit post ID |
| `subreddit` | `str` | Source subreddit name |
| `timestamp` | `datetime` | UTC time of post creation |
| `title` | `str` | Post title |
| `selftext` | `str` | Post body (cleaned) |
| `score` | `int` | Post upvote score |
| `num_comments` | `int` | Number of comments |
| `label` | `str` | `MDD` or `Control` |
| `selftext_cleaned` | `str` | Lowercased, stopwords removed, lemmatized |
| `word_count` | `int` | Word count of cleaned text |
| `sentiment_score` | `float` | VADER / TextBlob sentiment polarity |

## 8. Project Directory Structure (Planned)

```
BDA-MDD-Reddit-NLP/
├── data/
│   ├── raw/                    # Raw scraped data (gitignored if large)
│   └── processed/              # Cleaned, labeled CSVs
├── notebooks/
│   └── Assignment_1_PRAW_Extraction.ipynb
├── src/
│   ├── __init__.py
│   ├── scraper.py              # PullPush API scraper using requests
│   ├── pipeline.py             # Main Extraction & Cleaning pipeline
│   └── utils.py                # Helper functions
├── docs/
│   ├── assignments/
│   │   └── Our_Project_Task.md # Original grading criteria
│   ├── assets/
│   │   ├── Reddit_Proxy_API.pdf
│   │   └── reddit_api_project_brief.md
│   └── workflow.md             # Workflow documentation (deliverable)
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
| Reddit API keys | Requested as backup (Optional) |

## 10. Key Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-02 | Project initiated | Assignment 1 released by Prof. Sonali Agarwal |
| 2026-04-02 | Reddit API access requested | Needed for PRAW-based data extraction |
| — | — | — |

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Reddit API approval delayed | Blocks data extraction | Use Pushshift archive / academic datasets as fallback |
| Rate limiting by Reddit API | Slow data collection | Implement exponential backoff, batch requests |
| Subreddit posts removed/deleted | Incomplete dataset | Collect surplus data, document exclusions |
| Class imbalance (MDD >> Control) | Biased model | Stratified sampling, balance dataset sizes |

## 12. References

- [PRAW Documentation](https://praw.readthedocs.io/)
- [Reddit API Terms](https://www.reddit.com/wiki/api/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

---

> **Note**: This document is the single source of truth for the project. Update it as decisions are made, deliverables are completed, or the architecture evolves.
