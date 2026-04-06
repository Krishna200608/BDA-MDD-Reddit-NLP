<div align="center">

# 🧠 Reddit MDD NLP Corpus

**Natural Language Processing of Social Media for Major Depressive Disorder Symptom Identification**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Bio__ClinicalBERT-Transformers-FFD21E)](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Big Data Analytics (BDA) · 6th Semester · IIIT Allahabad*  
*Domain: HDA-4 · Group: 4*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Team](#-team)
- [Acknowledgements](#-acknowledgements)

---

## Overview

This repository presents a complete NLP pipeline for **binary classification of Reddit posts** into Major Depressive Disorder (**MDD**) indicators versus healthy control text. The project addresses a critical gap in scalable mental health screening by leveraging publicly available social media data.

We scrape self-reported posts from mental-health-focused subreddits (`r/depression`, `r/SuicideWatch`) and a neutral baseline (`r/CasualConversation`), apply classical and deep-learning NLP techniques, and evaluate which approach better captures depressive language markers in informal online text.

### Highlights

- 📊 **10,000-post** balanced corpus (5,000 MDD · 5,000 Control)
- 🏆 **91.7% accuracy** with TF-IDF + Logistic Regression baseline
- 🧬 **85.8% accuracy** with Bio_ClinicalBERT (768-dim) + Random Forest
- ⚡ **Hardware-agnostic** notebook — auto-detects CUDA GPU or falls back to CPU
- 🔄 **Automated quarterly refresh** via `schedule` daemon

---

## 📊 Key Results

| Model | Accuracy | Precision (MDD) | Recall (MDD) | F1 (Weighted) |
|:---|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | **91.7%** | 94% | 89% | 0.92 |
| Bio_ClinicalBERT + Random Forest | 85.8% | 87% | 83% | 0.86 |

> **Finding:** The classical sparse-vector approach outperformed the deep transformer model on this task. Forum-specific vocabulary (slang, colloquialisms) provided stronger discriminative signals than the clinical semantics captured by Bio_ClinicalBERT — a result consistent with domain-adaptation literature for social media NLP.

Full analysis → [`docs/methods_and_results.md`](docs/methods_and_results.md)

---

## 🏗 Architecture

```
                   ┌─────────────────────┐
                   │   Reddit (PullPush)  │
                   │  r/depression        │
                   │  r/SuicideWatch      │
                   │  r/CasualConversation│
                   └────────┬────────────┘
                            │
                   ┌────────▼────────────┐
                   │    src/scraper.py    │
                   │  (PullPush Proxy)    │
                   └────────┬────────────┘
                            │
                   ┌────────▼────────────┐
                   │   src/pipeline.py    │
                   │  Regex · NLTK · VADER│
                   └────────┬────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
     ┌────────▼─────────┐      ┌──────────▼──────────┐
     │  Track A (CPU)   │      │  Track B (GPU/CPU)   │
     │  TF-IDF (5k feat)│      │  Bio_ClinicalBERT   │
     │  Logistic Reg.   │      │  768-dim embeddings  │
     │                  │      │  Random Forest       │
     └────────┬─────────┘      └──────────┬───────────┘
              │                           │
              └──────────┬────────────────┘
                         │
                ┌────────▼────────┐
                │   Evaluation    │
                │ Accuracy · F1   │
                │ Confusion Matrix│
                └─────────────────┘
```

---

## 📁 Project Structure

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

## 🚀 Getting Started

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

## 💡 Usage

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

### Quarterly Automation

A long-running daemon that re-executes the full pipeline every 90 days:

```bash
python src/quarterly_updater.py
```

> The scheduler uses the [`schedule`](https://pypi.org/project/schedule/) library and runs persistently in the foreground. Terminate with `Ctrl+C`.

---

## 📦 Dataset

| Property | Value |
|:---|:---|
| **Total Posts** | ~10,000 (after cleaning) |
| **MDD Class** | `r/depression` + `r/SuicideWatch` (5,000 posts) |
| **Control Class** | `r/CasualConversation` (5,000 posts) |
| **Features** | Post ID, subreddit, title, selftext, cleaned text, word count, VADER sentiment |
| **Source** | [PullPush.io](https://pullpush.io) (Pushshift proxy) |
| **Split** | 80% train / 20% test (stratified) |

---

## 🛠 Tech Stack

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

## 👥 Team

| Name | Roll Number |
|:---|:---|
| **Krishna Sikheriya** | IIT2023139 |
| **Priyam Jyoti Chakrabarty** | IIT2023147 |
| **Tavish Chawla** | IIT2023150 |

**Instructor:** Prof. Sonali Agarwal  
**Institution:** Indian Institute of Information Technology, Allahabad (IIIT-A)

---

## 🙏 Acknowledgements

- [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — Clinical text embeddings pre-trained on MIMIC-III notes
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) — Lexicon-based sentiment scoring
- [PullPush.io](https://pullpush.io) — Pushshift API proxy for historical Reddit data
- [PRAW](https://praw.readthedocs.io/) — Python Reddit API Wrapper
- [scikit-learn](https://scikit-learn.org/) — Machine learning framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — Transformer model ecosystem

---

<div align="center">

*Built with ❤️ for Big Data Analytics — IIIT Allahabad, 2026*

</div>
