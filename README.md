<div align="center">

# Reddit MDD NLP Corpus

**Natural Language Processing of Social Media for Major Depressive Disorder Symptom Identification**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Bio__ClinicalBERT-Transformers-FFD21E)](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Big Data Analytics (BDA) · 6th Semester · IIIT Allahabad*  
*Domain: HDA-4 · Group: 4*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository presents a complete NLP pipeline for **binary classification of Reddit posts** into Major Depressive Disorder (**MDD**) indicators versus healthy control text. The project addresses a critical gap in scalable mental health screening by leveraging publicly available social media data.

We scrape self-reported posts from mental-health-focused subreddits (`r/depression`, `r/SuicideWatch`) and a neutral baseline (`r/CasualConversation`), apply classical and deep-learning NLP techniques, and evaluate which approach better captures depressive language markers in informal online text.

### Highlights

- **10,000-post** balanced corpus (5,000 MDD · 5,000 Control)
- **91.7% accuracy** with TF-IDF + Logistic Regression baseline
- **85.8% accuracy** with Bio_ClinicalBERT (768-dim) + Random Forest
- **Hardware-agnostic** notebook — auto-detects CUDA GPU or falls back to CPU
- **Automated quarterly refresh** via GitHub Actions CI/CD

---

## Key Results

| Model | Accuracy | Precision (MDD) | Recall (MDD) | F1 (Weighted) |
|:---|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | **91.7%** | 94% | 89% | 0.92 |
| Bio_ClinicalBERT + Random Forest | 85.8% | 87% | 83% | 0.86 |

> **Finding:** The classical sparse-vector approach outperformed the deep transformer model on this task. Forum-specific vocabulary (slang, colloquialisms) provided stronger discriminative signals than the clinical semantics captured by Bio_ClinicalBERT — a result consistent with domain-adaptation literature for social media NLP.

Full analysis → [`docs/methods_and_results.md`](docs/methods_and_results.md)

---

## Architecture

```mermaid
flowchart TD
    A["Reddit (PullPush Proxy)"] --> B["src/scraper.py"]
    
    subgraph DataSources ["Data Sources"]
        A1["r/depression"] --> A
        A2["r/SuicideWatch"] --> A
        A3["r/CasualConversation"] --> A
    end

    B --> C["src/pipeline.py\nRegex · NLTK · VADER"]
    C --> D["reddit_mdd_cleaned.csv\n~10,000 posts"]

    D --> E["Track A — Classical NLP\n(CPU)"]
    D --> F["Track B — Deep NLP\n(GPU / CPU)"]

    subgraph TrackA ["Baseline Track"]
        E --> E1["TF-IDF Vectorizer\n5,000 features · unigrams + bigrams"]
        E1 --> E2["Logistic Regression\nbalanced class weights"]
    end

    subgraph TrackB ["Advanced Track"]
        F --> F1["Bio_ClinicalBERT\n768-dim dense embeddings"]
        F1 --> F2["Random Forest\n100 estimators"]
    end

    E2 --> G["Evaluation\nAccuracy · F1 · Confusion Matrix"]
    F2 --> G

    style A fill:#4A90D9,stroke:#333,color:#fff
    style C fill:#6C5CE7,stroke:#333,color:#fff
    style D fill:#00B894,stroke:#333,color:#fff
    style G fill:#E17055,stroke:#333,color:#fff
    style E fill:#FDCB6E,stroke:#333,color:#333
    style F fill:#FDCB6E,stroke:#333,color:#333
```

---

## Project Structure

```
BDA-MDD-Reddit-NLP/
│
├── data/
│   ├── raw/                                  # Raw scraped CSVs
│   └── processed/                            # Cleaned & labeled dataset
│       └── reddit_mdd_cleaned.csv
│
├── notebooks/
│   ├── Assignment_1_PRAW_Extraction.ipynb    # Data extraction walkthrough
│   └── 02_text_classification_models.ipynb   # ML classification (TF-IDF & BERT)
│
├── src/
│   ├── __init__.py
│   ├── scraper.py                            # PullPush API client
│   ├── pipeline.py                           # End-to-end extraction + cleaning
│   ├── quarterly_updater.py                  # Automated 90-day refresh daemon
│   └── utils.py                              # Shared utilities
│
├── docs/
│   ├── assignments/
│   │   └── Our_Project_Task.md               # Original grading rubric
│   ├── assets/                               # Reference PDFs & briefs
│   ├── methods_and_results.md                # Evaluation report
│   └── workflow.md                           # Data pipeline documentation
│
├── .env.example                              # Environment variable template
├── .gitignore
├── Context.md                                # Living project context document
├── README.md                                 # ← You are here
└── requirements.txt                          # Python dependencies
```

---

## Getting Started

### Prerequisites

| Requirement | Version |
|:---|:---|
| Python | 3.12+ |
| pip | Latest |
| Git | 2.x |
| *(Optional)* NVIDIA GPU + CUDA | For accelerated BERT inference |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Krishna200608/BDA-MDD-Reddit-NLP.git
cd BDA-MDD-Reddit-NLP

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Assignment 1 — Data Extraction Pipeline

Run the complete scraping and preprocessing pipeline locally:

```bash
python src/pipeline.py
```

This will:
1. Scrape 10,000 posts via the PullPush proxy API
2. Clean text (regex, stopword removal, lowercasing)
3. Compute VADER sentiment scores
4. Export `data/raw/reddit_raw.csv` and `data/processed/reddit_mdd_cleaned.csv`

### Assignment 2 — Text Classification Models

#### Option A: Google Colab *(Recommended)*

1. Open [`notebooks/02_text_classification_models.ipynb`](notebooks/02_text_classification_models.ipynb) in Google Colab
2. Add a Colab Secret named `GITHUB_TOKEN` with your GitHub **Personal Access Token**
3. Set runtime to **T4 GPU** → *Runtime > Change runtime type > T4 GPU*
4. **⚠️ Important:** Update the first code cell with your own GitHub identity:
   ```python
   REPO_URL = f"https://{GITHUB_TOKEN}@github.com/<YOUR_USERNAME>/BDA-MDD-Reddit-NLP.git"
   !git config --global user.email "<YOUR_EMAIL>"
   !git config --global user.name  "<YOUR_USERNAME>"
   ```
5. Run all cells

#### Option B: Local Execution

Simply open the notebook locally. The hardware-detection logic will:
- Automatically fall back to CPU
- Subsample the dataset to **2,000 rows** for faster processing

### Quarterly Automation (CI/CD)

The dataset is **automatically refreshed every quarter** via a GitHub Actions workflow — no local machine needed.

| Property | Value |
|:---|:---|
| **Schedule** | 1st of Jan, Apr, Jul, Oct (00:00 UTC) |
| **Trigger** | Cron schedule + manual `workflow_dispatch` |
| **Workflow File** | [`.github/workflows/quarterly_update.yml`](.github/workflows/quarterly_update.yml) |

The workflow checks out the repo, runs `src/pipeline.py` on a GitHub-hosted runner, and commits the refreshed CSV back to `data/`.

> **Manual trigger:** Go to *Actions → Quarterly Dataset Update → Run workflow* to refresh on demand.

<details>
<summary>Local fallback (optional)</summary>

If you prefer running locally, a standalone daemon script is also available:

```bash
python src/quarterly_updater.py
```

This uses the [`schedule`](https://pypi.org/project/schedule/) library and runs persistently in the foreground. Terminate with `Ctrl+C`.

</details>

---

## Dataset

| Property | Value |
|:---|:---|
| **Total Posts** | ~10,000 (after cleaning) |
| **MDD Class** | `r/depression` + `r/SuicideWatch` (5,000 posts) |
| **Control Class** | `r/CasualConversation` (5,000 posts) |
| **Features** | Post ID, subreddit, title, selftext, cleaned text, word count, VADER sentiment |
| **Source** | [PullPush.io](https://pullpush.io) (Pushshift proxy) |
| **Split** | 80% train / 20% test (stratified) |

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| **Language** | Python 3.12+ |
| **Data** | pandas · NumPy |
| **NLP** | NLTK · regex · VADER Sentiment |
| **Embeddings** | [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) (HuggingFace Transformers) |
| **ML** | scikit-learn (Logistic Regression, Random Forest, TF-IDF) |
| **Deep Learning** | PyTorch (CUDA / CPU) |
| **Automation** | schedule |
| **Environment** | venv · pip |
| **Version Control** | Git + GitHub |

---

## Team

| Name | Roll Number |
|:---|:---|
| **Krishna Sikheriya** | IIT2023139 |
| **Priyam Jyoti Chakrabarty** | IIT2023147 |
| **Tavish Chawla** | IIT2023150 |

**Instructor:** Prof. Sonali Agarwal  
**Institution:** Indian Institute of Information Technology, Allahabad (IIIT-A)

---

## Acknowledgements

- [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — Clinical text embeddings pre-trained on MIMIC-III notes
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) — Lexicon-based sentiment scoring
- [PullPush.io](https://pullpush.io) — Pushshift API proxy for historical Reddit data
- [PRAW](https://praw.readthedocs.io/) — Python Reddit API Wrapper
- [scikit-learn](https://scikit-learn.org/) — Machine learning framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — Transformer model ecosystem

---

<div align="center">

*Big Data Analytics — IIIT Allahabad, 2026*

</div>
