# Classification Methods and Results

**Course**: Big Data Analytics (BDA) — 6th Semester  
**Domain**: HDA-4 | **Group**: 4  
**Title**: Natural Language Processing of Social Media for MDD Symptom Identification

---

## 1. Methodology

The classification phase of this project aimed to identify granular language patterns associated with Major Depressive Disorder (MDD) by classifying text into three severity tiers: Severe Ideation (from *r/SuicideWatch*), Moderate MDD (from *r/depression*), and a general population baseline Control (from *r/CasualConversation*).

We employed a dual-track analytical architecture, contrasting classical sparse term-frequency machine learning against modern dense-vector deep learning.

### 1.1 Baseline Track: TF-IDF + Logistic Regression
- **Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF). We extracted the top 5,000 most predictive unigram and bigram features, filtering out standard English stop words.
- **Model**: Logistic Regression utilizing a 'balanced' class-weights parameter to counteract any minor asymmetries in class distributions.
- **Hardware Profile**: Pure CPU Execution. 

### 1.1(b) Additional Sparse Baseline: TF-IDF + LinearSVC
- **Vectorization**: The same TF-IDF feature space is reused to keep the comparison fair.
- **Model**: Linear Support Vector Classifier with balanced class weights.
- **Purpose**: This provides a stronger non-probabilistic classical baseline without expanding the project scope beyond the course expectation.

### 1.2 Deep NLP Track: TwitterRoBERTa + Random Forest
- **Vectorization**: We passed raw text through `cardiffnlp/twitter-roberta-base`, a transformer model pre-trained on large-scale social-media text. This yields 768-dimensional dense vectors that are better suited to informal online language than clinical-note models.
- **Preprocessing**: Prior to tokenization, usernames are normalized to `@user` and links to `http`, following the model-card recommendation.
- **Model**: A Random Forest classifier (100 estimators; max-depth untethered) mapped to the continuous vector mappings of the `[CLS]` token hidden states.
- **Hardware Profile**: Hybrid CPU/CUDA via Google Colab.

### 1.3 Data Quality and Leakage Controls
- Duplicate rows are removed in two stages: duplicate `post_id`, then exact duplicate `title+selftext`.
- A deterministic `text_hash` column is created from `title+selftext`.
- A compact QA artifact (`data/processed/dataset_summary.csv`) records row counts, duplicate removals, missing values, label counts, length statistics, and date range.

### 1.4 Latest QA-Validated Snapshot
- Rows before QA: `9,800`
- Rows after QA and length filtering: `9,607`
- Duplicate rows removed: `145` duplicate `post_id` rows and `48` duplicate `title+selftext` rows
- Date range: `2025-03-05T06:40:41` to `2025-05-19T18:11:58`
- Label counts:
  - `Control`: `4,903`
  - `Moderate MDD`: `2,408`
  - `Severe Ideation`: `2,296`

---

## 2. Experimental Results

We now evaluate the pipelines with a **two-layer protocol**:
- **Fixed 80/20 stratified holdout** for confusion matrices, classwise precision/recall, and error analysis
- **5-fold, 3-repeat `RepeatedStratifiedKFold`** for mean ± std reporting of accuracy, macro F1, and weighted F1

### 2.1 Exported Metrics Source of Truth
Rather than maintaining hard-coded markdown metrics that can drift as the dataset refreshes, the upgraded notebook exports a synchronized artifact:
- `data/processed/results_summary.csv`

This file contains:
- repeated-CV mean ± std metrics for the main models;
- holdout accuracy, macro F1, weighted F1, and per-class precision/recall;
- the permutation-test p-value for the TF-IDF + Logistic Regression baseline.

### 2.2 Repeated Cross-Validation Results

| Model | Accuracy (mean ± std) | Macro F1 (mean ± std) | Weighted F1 (mean ± std) |
|:---|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | **0.7762 ± 0.0100** | **0.7251 ± 0.0109** | **0.7747 ± 0.0099** |
| TF-IDF + LinearSVC | 0.7616 ± 0.0077 | 0.7059 ± 0.0096 | 0.7584 ± 0.0082 |
| TwitterRoBERTa + Random Forest | 0.7600 ± 0.0071 | 0.6955 ± 0.0085 | 0.7503 ± 0.0076 |

The repeated-CV ranking shows that the sparse **TF-IDF + Logistic Regression** baseline remained the strongest overall configuration on the current three-class dataset. The extra `LinearSVC` baseline was competitive but did not surpass Logistic Regression, and the dense TwitterRoBERTa feature-extraction track remained viable while trailing the best sparse model on macro F1.

### 2.3 Fixed Holdout Results

| Model | Accuracy | Macro F1 | Weighted F1 | Precision (Control) | Recall (Control) | Precision (Moderate) | Recall (Moderate) | Precision (Severe) | Recall (Severe) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | **0.7841** | **0.7355** | **0.7829** | 0.9059 | 0.9225 | 0.6347 | 0.6452 | **0.6721** | **0.6340** |
| TF-IDF + LinearSVC | 0.7700 | 0.7175 | 0.7673 | 0.8891 | 0.9235 | 0.6089 | 0.5975 | 0.6651 | 0.6231 |
| TwitterRoBERTa + Random Forest | 0.7596 | 0.6969 | 0.7517 | 0.8614 | 0.9501 | 0.5957 | 0.5685 | 0.6684 | 0.5534 |

The holdout split tells the same story as repeated CV: Logistic Regression is the best overall model on accuracy and macro F1, while the RoBERTa + Random Forest track still offers a strong social-media-aware baseline under the official **Tesla T4 / full-dataset** evaluation mode.

### 2.4 Additional Evaluation Signals
- **Permutation test**: the TF-IDF + Logistic Regression baseline achieved macro F1 `0.7221` with `p = 0.032258`, indicating performance above chance at the 5% level.
- **Learning curve**: a dedicated learning-curve figure is now produced for the main sparse baseline and can be reused directly in the report/demo.
- **Error analysis**: representative holdout mistakes are exported to `data/processed/error_analysis_holdout.csv`.
- **Interpretability artifact**: top influential tokens per class are exported to `data/processed/top_tokens_by_class.csv`.

### 2.5 Error Analysis Summary

The holdout error-analysis export contains `415` misclassified examples. The dominant confusions were between adjacent severity tiers:
- `Severe Ideation → Moderate MDD`: `123`
- `Moderate MDD → Severe Ideation`: `122`
- `Control → Moderate MDD`: `56`
- `Moderate MDD → Control`: `49`
- `Severe Ideation → Control`: `45`
- `Control → Severe Ideation`: `20`

This pattern is consistent with the qualitative notes in the artifact: the hardest cases are not cleanly separated by topic alone, but by **intensity** and the presence or absence of explicit ideation vocabulary. The most frequent explanation strings in the exported file highlight overlap around words like `feel`, `help`, `depression`, and generic distress phrasing.

### 2.6 Explainable AI (SHAP Visualization)

To avoid "black-box" clinical predictions, SHAP (SHapley Additive exPlanations) integration was introduced.
The notebook computes SHAP summaries for the Logistic Regression pipeline and also exports the strongest class-specific tokens to `data/processed/top_tokens_by_class.csv`. The latest run surfaced the following high-signal vocabulary:
- **Control**: `curious`, `lol`, `new`, `whats`, `coffee`, `share`, `chat`
- **Moderate MDD**: `depression`, `depressed`, `feel`, `advice`, `hate`, `therapy`, `help`
- **Severe Ideation**: `suicide`, `suicidal`, `die`, `kill`, `pain`, `attempt`, `anymore`

These tokens align with the class definitions and also explain the main error pattern: moderate and severe posts frequently share depressive vocabulary, while explicit self-harm and ideation terms provide the clearest separation into the Severe Ideation tier.

---

## 3. Exploratory Data Analysis and Language Pattern Detection

This section directly addresses the project objective of detecting *symptom and emotional language patterns* in Reddit posts across the three severity tiers. Six complementary analyses were conducted on the full corpus.

### 3.1 DSM-5 Symptom Keyword Frequency Analysis

A curated list of 24 DSM-5-aligned symptom keywords (e.g., *hopeless*, *worthless*, *suicide*, *insomnia*, *fatigue*, *guilt*, *empty*, *numb*) was compiled and compared across the three labels.

**Key Findings:**
- The distress-related classes contain much higher symptom-keyword density than the Control class.
- The strongest severe-tier cues include *suicide*, *suicidal*, *die*, and *pain*, while the moderate tier is dominated more by *depression* and *depressed*.
- The keyword-frequency differential validates the discriminative power of the sparse baseline because these domain-specific terms provide strong classification signals.

### 3.2 Word Clouds (MDD vs. Control)

Side-by-side word clouds were generated for all three classes to provide an intuitive comparison of everyday language, depressive discourse, and ideation-focused language.

**Key Findings:**
- The Control cloud is more casual and conversational.
- The Moderate MDD cloud is dominated by depressive self-description and help-seeking terms.
- The Severe Ideation cloud concentrates the most urgent distress and self-harm-adjacent language.

### 3.3 VADER Sentiment Score Distribution

A KDE (Kernel Density Estimate) comparison of VADER compound sentiment scores across the three labels.

**Key Findings:**
- Moderate and Severe posts both shift left into the negative range, with the Severe Ideation class concentrated most strongly in negative sentiment.
- Control posts remain centered closer to neutral or mildly positive values.
- This confirms that sentiment scoring is a useful supporting feature for distinguishing distress-related text from general text.

### 3.4 Post Length Distribution

A word-count comparison was used to study expression length across the three severity tiers.

**Key Findings:**
- Distress-related posts tend to be longer on average than Control posts, consistent with rumination-style self-expression.
- Control text remains comparatively shorter and more compact.

### 3.5 Top Bigrams (MDD vs. Control)

The top 15 bigrams (two-word phrases) were extracted to capture multi-word expressions that reflect distinct language patterns across Control, Moderate MDD, and Severe Ideation.

**Key Findings:**
- Moderate and Severe bigrams capture emotional and cognitive distress phrases characteristic of depressive narratives.
- Control bigrams reflect everyday conversational topics without clinical or emotional weight.
- Bigram analysis reinforces the effectiveness of TF-IDF with `ngram_range=(1,2)` in capturing these discriminative multi-word patterns.

### 3.6 Summary

The six EDA analyses collectively demonstrate that the three labels exhibit measurably different linguistic signatures across multiple dimensions: higher symptom keyword density, more negative sentiment, greater post length, and distinctive multi-word patterns in the two distress-related classes, with the strongest ideation terms concentrated in the Severe label. These features align with and reinforce the classification results, confirming that the project successfully detects symptom and emotional language patterns associated with Major Depressive Disorder severity.

---

## 4. Automation Implementation

In alignment with Deliverable requirements, we have provisioned an execution script `src/quarterly_updater.py`. 
Utilizing the static `schedule` Python daemon, this script operates persistently to spin up the data-scraper pipelines and data cleansing functions completely autonomously every **90 days** recursively, ensuring the Machine Learning pipelines consistently ingest fresh modern syntactic data.

Additionally, a GitHub Actions CI/CD workflow (`.github/workflows/quarterly_update.yml`) provides cloud-based automation that refreshes the dataset on the 1st of January, April, July, and October without requiring local machine uptime.

---

## 5. Limitations and Responsible Use

- The labels are **course-project proxy labels** derived from subreddit source, not clinician-verified diagnoses.
- Self-reported social-media text contains noise, ambiguity, sarcasm, and cross-over language between adjacent severity tiers.
- The outputs are suitable for **academic NLP analysis and classroom evaluation**, not for real-world mental-health triage.
- The source data concerns sensitive personal experiences, so qualitative examples should be handled with privacy and ethical caution in demos and reports.

