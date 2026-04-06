# MDD Reddit Analytics
**Big Data Analytics (BDA) Course Project — 6th Semester**  
**Domain**: HDA-4 | **Group**: 4

## Overview
This repository contains an academic research implementation for text classification. Specifically, we classify and detect Natural Language indicators pointing to Major Depressive Disorder (MDD) symptoms extracted from Reddit self-reports.

### Team Members
- Krishna Sikheriya (IIT2023139)
- Priyam Jyoti Chakrabarty (IIT2023147)
- Tavish Chawla (IIT2023150)

## Directory Structure
- `data/` : `raw` backups and `processed` training vectors (CSVs).
- `notebooks/` :
  - `Assignment_1_PRAW_Extraction.ipynb`: Official Extraction Submission.
  - `02_text_classification_models.ipynb`: Official Machine Learning Classification algorithms (TF-IDF vs. Bio_ClinicalBERT) with Colab T4 hardware mapping.
- `src/` : Modular, purely `.py` implementations (`scraper.py`, `pipeline.py`, `quarterly_updater.py`).
- `docs/` : Methodology workflows, evaluation metrics, PDF assets, and initial instructions.

## Installation & Setup
1. Standardize virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Running Assignment 1
To complete the *Data Extraction* assignment, run the core pipeline Python script:
1. Open terminal in the project root.
2. Ensure you have activated your environment and installed dependencies.
3. Run the complete pipeline via terminal:
```bash
python src/pipeline.py
```
This automatically scrapes PullPush.io, exports `data/raw/reddit_raw.csv`, processes linguistic parameters via NLP layers, and yields the final `data/processed/reddit_mdd_cleaned.csv` all autonomously.

## Running Assignment 2
The second assignment focuses on textual classification models:
1. **Google Colab (Recommended):** Upload the `02_text_classification_models.ipynb` from the `notebooks` directory into a Google Colab environment.
2. Add your GitHub Personal Access Token natively inside the Colab side panel as a new Secret named `GITHUB_TOKEN`.
3. Set your Colab runtime natively to an **NVIDIA T4 GPU**, and select *Run All*.
4. **Identity Configuration:** In the first code cell of the notebook, ensure you update the `REPO_URL`, `user.email`, and `user.name` with your own GitHub credentials to enable push/pull synchronization.
5. *Alternatively for Local testing:* This notebook possesses dynamic native routing. Running it locally will force execution via standard CPUs while shrinking the test dataset down to 2000 points to prevent laptop CPU throttling algorithms.

### Quarterly Automation Daemon
To activate the automation requirement natively in your filesystem:
```bash
python src/quarterly_updater.py
```
> Wait until active. The system will recursively spawn extraction loops automatically every 90 days persistently over existing processes using the `schedule` standard module.
