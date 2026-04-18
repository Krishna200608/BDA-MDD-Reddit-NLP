<div align="center">

# Reddit MDD NLP Corpus

**Natural Language Processing of Social Media for Major Depressive Disorder Symptom Identification**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/TwitterRoBERTa-Transformers-FFD21E)](https://huggingface.co/cardiffnlp/twitter-roberta-base)
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

This repository presents a complete NLP pipeline for **multi-class severity classification of Reddit posts** into Moderate MDD, Severe Ideation, and healthy control text. The project addresses a critical gap in scalable mental health screening by leveraging publicly available social media data.

We scrape posts from `r/SuicideWatch`, `r/depression`, and a neutral baseline `r/CasualConversation`, apply classical and transformer-based NLP techniques, and use Explainable AI (SHAP) to interpret predictive language features in informal social-media text.

### Highlights

- **10,000-post raw extraction target** with a **9,800-row processed snapshot** after minimum-length filtering
- Tertiary severity classification (Control, Moderate MDD, Severe Ideation)
- **Classical NLP:** TF-IDF + Logistic Regression
- **Deep Representation:** `twitter-roberta-base` + Random Forest
- **Explainable AI (XAI):** SHAP integration for clinical transparency
- **Comprehensive EDA** — DSM-5 symptom keyword analysis, word clouds, sentiment distributions, post length profiling, and bigram analysis
- **Hardware-agnostic** notebook — auto-detects CUDA GPU or falls back to CPU
- **Automated quarterly refresh** via GitHub Actions CI/CD

---

## Key Results

*(Note: Quantitative metrics below reflect the current three-class severity setup. Sparse TF-IDF remained the stronger baseline, while the transformer track was updated from older clinical-note models to the social-media-trained `cardiffnlp/twitter-roberta-base`.)*

| Model | Accuracy | Macro F1 | Weighted F1 | Precision (Severe) |
|:---|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | **78.7%** | 0.74 | 0.79 | 65% |
| TwitterRoBERTa + Random Forest | 74.2% | 0.67 | 0.73 | 65% |

> **Explainability Milestone:** The complete analytical ensemble relies on **SHAP (SHapley Additive exPlanations)** mapped to the sparse Logistic Regression model. This isolates specific dialectal indicators driving high-risk categorizations (like *Severe Ideation*) to avoid black-box psychiatric evaluations.

### EDA and Language Pattern Detection

Six exploratory analyses were conducted to detect symptom and emotional language patterns:

| Analysis | Key Insight |
|:---|:---|
| **DSM-5 Symptom Keywords** | MDD posts contain significantly higher symptom keyword density; *depression*, *anxiety*, *die*, *pain* dominate |
| **Word Clouds** | MDD vocabulary is emotionally charged; Control vocabulary is casual and action-oriented |
| **Sentiment Distribution** | MDD posts show a clear negative shift in VADER compound scores |
| **Post Length** | MDD posts are longer on average, consistent with rumination patterns |
| **Top Bigrams** | Severe bigrams capture distress phrases; Control bigrams reflect everyday topics |
| **XAI (SHAP)** | Local force-plots demystify specific text predictions for maximum triage transparency |

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
    C --> D["reddit_mdd_cleaned.csv\ncurrent snapshot: 9,800 posts"]

    D --> E["Track A — Classical NLP\n(CPU)"]
    D --> F["Track B — Deep NLP\n(GPU / CPU)"]

    subgraph TrackA ["Baseline Track"]
        E --> E1["TF-IDF Vectorizer\n5,000 features · unigrams + bigrams"]
        E1 --> E2["Logistic Regression\nbalanced class weights"]
    end

    subgraph TrackB ["Advanced Track"]
        F --> F1["TwitterRoBERTa\n768-dim dense embeddings"]
        F1 --> F2["Random Forest\n100 estimators"]
    end

    E2 --> G["Evaluation\nAccuracy · Macro F1 · Weighted F1 · Confusion Matrix"]
    F2 --> G

    D --> H["EDA & Language Patterns"]

    subgraph EDA ["Exploratory Analysis"]
        H --> H1["Symptom Keywords\nDSM-5 Frequency Heatmap"]
        H --> H2["Word Clouds\nMDD vs Control"]
        H --> H3["Sentiment · Length · Bigrams"]
    end

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
│   ├── Assignment_1_PRAW_Extraction.ipynb    # Legacy notebook from the original PRAW plan
│   └── 02_text_classification_models.ipynb   # ML classification (TF-IDF & TwitterRoBERTa)
│
├── src/
│   ├── scraper.py                            # PullPush API client
│   ├── pipeline.py                           # End-to-end extraction + cleaning
│   └── quarterly_updater.py                  # Local 90-day refresh fallback
│
├── docs/
│   ├── assignments/
│   │   └── Our_Project_Task.md               # Original grading rubric
│   ├── assets/                               # Reference PDFs & briefs
│   ├── methods_and_results.md                # Evaluation report
│   ├── workflow.md                           # Data pipeline documentation
│   └── team_work_division.md                 # Group work allocation cheat sheet
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
1. Attempt to scrape 10,000 posts via the PullPush proxy API
2. Clean text (regex, stopword removal, lowercasing)
3. Drop posts with fewer than 5 cleaned words and compute VADER sentiment scores
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
| **Current Committed Snapshot** | 9,800 processed rows |
| **Label Distribution** | `Control` 4,992 · `Moderate MDD` 2,458 · `Severe Ideation` 2,350 |
| **Raw Extraction Target** | `r/depression` 2,500 · `r/SuicideWatch` 2,500 · `r/CasualConversation` 5,000 |
| **Features** | `post_id`, `subreddit`, `timestamp`, `title`, `selftext`, `score`, `num_comments`, `author`, `label`, `selftext_cleaned`, `word_count`, `sentiment_score` |
| **Source** | [PullPush.io](https://pullpush.io) (Pushshift proxy) |
| **Split** | 80% train / 20% test (stratified, in the notebook experiments) |

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| **Language** | Python 3.12+ |
| **Data** | pandas · NumPy |
| **NLP** | NLTK · regex · VADER Sentiment · wordcloud |
| **Embeddings** | [TwitterRoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base) (HuggingFace Transformers) |
| **ML** | scikit-learn (Logistic Regression, Random Forest, TF-IDF) |
| **Deep Learning** | PyTorch (CUDA / CPU) |
| **Automation** | GitHub Actions CI/CD + `schedule` fallback |
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

- [cardiffnlp/twitter-roberta-base](https://huggingface.co/cardiffnlp/twitter-roberta-base) — Social-media transformer used for dense embedding experiments
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) — Lexicon-based sentiment scoring
- [PullPush.io](https://pullpush.io) — Pushshift API proxy for historical Reddit data
- [scikit-learn](https://scikit-learn.org/) — Machine learning framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — Transformer model ecosystem

---

<div align="center">

*Big Data Analytics — IIIT Allahabad, 2026*

</div>
