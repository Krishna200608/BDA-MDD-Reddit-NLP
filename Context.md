# Project Context — BDA: Reddit MDD NLP Corpus

> **Living Agent Handoff Document**  
> Last Updated: 2026-04-21

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Course** | Big Data Analytics (BDA) — 6th Semester |
| **Institution** | Indian Institute of Information Technology, Allahabad (IIIT-A) |
| **Professor** | Prof. Sonali Agarwal |
| **Task ID** | HDA-4 |
| **Repository** | `Krishna200608/BDA-MDD-Reddit-NLP` |
| **Project Title** | Natural Language Processing of Social Media for MDD Symptom Identification |
| **Primary Goal** | Build a coursework-quality Reddit NLP pipeline for 3-class severity classification and interpretable language analysis |

This repository is no longer just a data-collection assignment. It is now an end-to-end coursework project with:
- Reddit scraping through PullPush
- QA-safe dataset construction
- classical and transformer-based model comparison
- exported inference-ready model artifacts
- a local-first Streamlit dashboard for live demo inference

---

## 2. Team

| Name | Roll Number | Notes |
|---|---|---|
| Krishna Sikheriya | IIT2023139 | Main repo owner |
| Priyam Jyoti Chakrabarty | IIT2023147 | Team member |
| Tavish Chawla | IIT2023150 | Team member |

---

## 3. Problem Framing

### Current supervised task
Classify Reddit posts into:

| Label | Meaning | Source Subreddit |
|---|---|---|
| `Control` | Neutral / casual conversation baseline | `r/CasualConversation` |
| `Moderate MDD` | Depression-related distress language | `r/depression` |
| `Severe Ideation` | Stronger suicide / severe-risk language | `r/SuicideWatch` |

### Important framing constraint
These are **proxy labels derived from subreddit origin**, not medical diagnoses.

The project is intended for:
- coursework
- NLP experimentation
- interpretable model comparison
- live academic demo

It is **not** intended for:
- clinical screening
- healthcare deployment
- diagnosis

---

## 4. Current Repo State

### Current processed dataset snapshot

| Item | Value |
|---|---|
| Rows before QA | `9,800` |
| Rows after QA / modeling snapshot | `9,607` |
| Duplicate `post_id` rows removed | `145` |
| Exact duplicate `title+selftext` rows removed | `48` |
| Date range | `2025-03-05T06:40:41` → `2025-05-19T18:11:58` |
| Label distribution | `Control` 4,903 · `Moderate MDD` 2,408 · `Severe Ideation` 2,296 |

### Best committed evaluation snapshot

| Item | Value |
|---|---|
| Best repeated-CV model | `TF-IDF + Logistic Regression` |
| Repeated-CV accuracy | `0.7762 ± 0.0100` |
| Repeated-CV macro F1 | `0.7251 ± 0.0109` |
| Best holdout model | `TF-IDF + Logistic Regression` |
| Holdout accuracy | `0.7841` |
| Holdout macro F1 | `0.7355` |
| Permutation test | macro F1 `0.7221`, `p = 0.032258` |

### Source of truth files
- Metrics source of truth: `data/processed/results_summary.csv`
- Dataset QA source of truth: `data/processed/dataset_summary.csv`
- Saved deployment registry: `models/model_metadata.json`
- Label-order source of truth: `models/class_labels.json`

---

## 5. Main Deliverables Now Present

### Assignment 1
| Deliverable | Status |
|---|---|
| Scraping script | ✅ Complete |
| Cleaned CSV dataset | ✅ Complete |
| Workflow documentation | ✅ Complete |

### Assignment 2
| Deliverable | Status |
|---|---|
| Strengthened classification notebook | ✅ Complete |
| EDA and language-pattern analysis | ✅ Complete |
| Documentation of methods and results | ✅ Complete |
| Quarterly refresh automation | ✅ Complete |
| Saved `.joblib` deployment artifacts | ✅ Complete |
| Streamlit live inference dashboard | ✅ Complete |

---

## 6. Architecture Overview

### Data pipeline
1. `src/scraper.py`
   - PullPush scraper using `requests`
   - pagination / retry behavior
   - subreddit-based data collection

2. `src/pipeline.py`
   - combines subreddit pulls
   - enforces required columns
   - deduplicates by `post_id`
   - deduplicates exact `title+selftext`
   - creates deterministic `text_hash`
   - cleans `selftext` into `selftext_cleaned`
   - computes `word_count`
   - computes VADER `sentiment_score`
   - exports raw + processed CSVs
   - exports `dataset_summary.csv`

### Modeling pipeline
Notebook: `notebooks/02_text_classification_models.ipynb`

Tracks:
- Sparse Track A:
  - TF-IDF + Logistic Regression
  - TF-IDF + LinearSVC
- Dense Track B:
  - TwitterRoBERTa embeddings + Random Forest

Evaluation:
- fixed 80/20 stratified holdout
- 5-fold, 3-repeat repeated CV
- permutation test on main sparse baseline
- learning curve
- SHAP / token-level interpretation
- holdout error analysis

### Deployment / live inference pipeline
- `models/` stores exported deployment artifacts
- `src/inference.py` loads artifacts and runs prediction
- `app.py` serves the Streamlit dashboard
- `src/dashboard_utils.py` contains UI rendering/styling helpers

---

## 7. Files and What They Mean

### Core runtime files

| File | Role |
|---|---|
| `src/scraper.py` | PullPush Reddit collection |
| `src/pipeline.py` | QA, cleaning, CSV generation |
| `notebooks/02_text_classification_models.ipynb` | canonical evaluation + artifact export notebook |
| `src/inference.py` | shared inference logic for saved models |
| `src/dashboard_utils.py` | Streamlit layout/styling helpers |
| `app.py` | Streamlit entry point |

### Data / evaluation artifacts

| File | Role |
|---|---|
| `data/processed/reddit_mdd_cleaned.csv` | final processed dataset |
| `data/processed/dataset_summary.csv` | dataset QA summary |
| `data/processed/results_summary.csv` | committed evaluation metrics |
| `data/processed/error_analysis_holdout.csv` | saved misclassification analysis |
| `data/processed/top_tokens_by_class.csv` | token-level class interpretation |

### Deployment artifacts

| File | Role |
|---|---|
| `models/tfidf_logreg_pipeline.joblib` | default live inference model |
| `models/tfidf_linearsvc_pipeline.joblib` | secondary sparse model |
| `models/roberta_rf_classifier.joblib` | saved Random Forest head for RoBERTa runtime inference |
| `models/model_metadata.json` | dashboard metadata + benchmark registry |
| `models/class_labels.json` | label order and numeric mapping |

---

## 8. Current Saved-Model Contract

### Default model
Default live model is:
- `tfidf_logreg`
- display name: `TF-IDF + Logistic Regression`

This is encoded in:
- `models/model_metadata.json`

### Important saved-model behavior

#### Sparse models
Saved as complete sklearn pipelines:
- vectorizer + classifier together
- loaded directly from `.joblib`

Important implementation note:
- these sparse models were trained on `selftext_cleaned`
- `src/inference.py` now preprocesses live text with the same regex + stopword-cleaning logic used in `src/pipeline.py`
- this alignment is important; otherwise live inference drifts from training behavior

#### RoBERTa model
Saved artifact is only:
- `roberta_rf_classifier.joblib`

The transformer encoder is **not** stored in the repo as weights.
At runtime:
- load `cardiffnlp/twitter-roberta-base`
- apply model-card preprocessing:
  - usernames → `@user`
  - links → `http`
- compute embeddings
- pass embeddings into saved RF classifier

This was chosen to keep the repo lighter and the export format practical.

---

## 9. Dashboard State

### Current dashboard purpose
The Streamlit dashboard is for:
- live classroom demo
- showing single-post inference
- switching between saved models
- presenting prediction results cleanly

### Current UX scope
V1 supports:
- single text input
- curated example texts
- model selector
- predict button
- clear result button

V1 does **not** support:
- batch CSV upload
- hosted cloud deployment workflow
- clinical usage

### Current visual direction
Inspired by the user’s NeuroFetal-AI project:
- dark sidebar
- bright main canvas
- hero/status banner
- metric cards
- central decision card
- compact analysis sections

### Important recent UI decisions
- dashboard was simplified to reduce clutter
- detailed sections were moved into tabs
- limitations were collapsed into an expander
- text contrast was improved for light backgrounds
- disabled text areas were made readable

---

## 10. Commands Another Agent Should Know

### Local environment setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Refresh dataset locally
```bash
python src/pipeline.py
```

### Run dashboard locally
```bash
streamlit run app.py
```

### Notebook reproduction path
Preferred:
- Google Colab
- T4 GPU
- run `notebooks/02_text_classification_models.ipynb`
- export fresh model artifacts
- push from Colab
- pull locally

---

## 11. Dependency / Runtime Constraints

### Important compatibility pin
`requirements.txt` pins:
- `scikit-learn==1.6.1`

Reason:
- the saved `.joblib` artifacts were exported with `scikit-learn 1.6.1`
- loading them under `1.8.0` produced `InconsistentVersionWarning`

The pin exists specifically to keep local live inference stable.

### Transformer runtime note
The first RoBERTa inference may take noticeably longer because:
- Hugging Face model + tokenizer load at runtime
- embeddings are computed live

This is expected behavior, not a bug.

---

## 12. Documentation Map

| File | What to read it for |
|---|---|
| `README.md` | polished repo overview and usage |
| `docs/workflow.md` | pipeline and methodology narrative |
| `docs/methods_and_results.md` | evaluation interpretation |
| `docs/team_work_division.md` | viva / contribution notes |
| `Context.md` | this full handoff and current-state guide |

---

## 13. Key Decision Log

| Date | Decision | Why it matters |
|---|---|---|
| 2026-04-02 | Project initiated | Assignment start |
| 2026-04-06 | Quarterly refresh moved to GitHub Actions | removed need for always-on local scheduler |
| 2026-04-07 | Added EDA and language-pattern detection | strengthened Assignment 2 beyond pure classification |
| 2026-04-10 | Switched to 3-class severity mapping | project evolved beyond binary framing |
| 2026-04-10 | Switched to `cardiffnlp/twitter-roberta-base` | avoided gated-repo issues and better matched social-media language |
| 2026-04-18 | Added QA hardening | duplicate removal + `text_hash` + dataset summary |
| 2026-04-18 | Upgraded evaluation protocol | repeated CV, LinearSVC, permutation test, learning curve, error analysis |
| 2026-04-18 | Final Colab T4 metrics committed | produced final CSV artifacts |
| 2026-04-21 | Saved deployment model artifacts | enabled live dashboard inference |
| 2026-04-21 | Built Streamlit demo app | local-first live inference experience |
| 2026-04-21 | Kept executed notebook committed | coursework evidence retained in repo |
| 2026-04-21 | Fixed live sparse preprocessing alignment | dashboard sparse inference now matches training-time cleaning |
| 2026-04-21 | Simplified dashboard layout | reduced clutter and improved visibility |

---

## 14. Known Risks / Caveats

| Risk | Meaning | Current mitigation |
|---|---|---|
| PullPush instability | future refreshes may fail or slow down | retries + keep committed snapshots |
| Proxy labels are noisy | labels are not diagnoses | explicit academic-only framing everywhere |
| Moderate vs Severe overlap | hardest classification boundary | report macro F1 and per-class metrics, not just accuracy |
| RoBERTa runtime cost | first inference is slower | keep logistic regression as default live model |
| Model/version mismatch | `.joblib` may warn or fail under different sklearn versions | requirements pin to `1.6.1` |

---

## 15. Working Tree Hygiene

Before making further changes, another agent should always check:
- `git status`
- whether the executed notebook has changed again after a new Colab run
- whether saved model artifacts in `models/` still match the current notebook export contract

This matters because this repo mixes:
- committed evaluation artifacts
- committed deployment artifacts
- notebook-driven updates that may be regenerated in Colab

---

## 16. What Another AI Agent Should Do First

If another agent enters this repo, recommended first steps are:

1. Read `git status` to separate committed repo truth from local uncommitted work.
2. Read `models/model_metadata.json` to understand the live inference contract.
3. Read `src/inference.py` and `app.py` together to understand current dashboard behavior.
4. Read `data/processed/results_summary.csv` for the final metrics source of truth.
5. Only then modify docs / dashboard / notebook as needed.

---

## 17. Single-Sentence Summary

This project is a 3-class Reddit mental-health NLP coursework pipeline with QA-safe data preparation, evaluated sparse and dense models, exported `.joblib` inference artifacts, and a polished local-first Streamlit dashboard for live demo prediction.

---

> **Final note for agents:** Treat `results_summary.csv` as the metrics source of truth, `model_metadata.json` as the deployment source of truth, and `git status` as the source of truth for whether the latest dashboard/context refinements are already committed.
