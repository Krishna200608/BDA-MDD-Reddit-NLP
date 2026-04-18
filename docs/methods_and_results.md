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

### 1.2 Deep NLP Track: TwitterRoBERTa + Random Forest
- **Vectorization**: We passed raw text through `cardiffnlp/twitter-roberta-base`, a transformer model pre-trained on large-scale social-media text. This yields 768-dimensional dense vectors that are better suited to informal online language than clinical-note models.
- **Model**: A Random Forest classifier (100 estimators; max-depth untethered) mapped to the continuous vector mappings of the `[CLS]` token hidden states.
- **Hardware Profile**: Hybrid CPU/CUDA via Google Colab.

---

## 2. Experimental Results

We evaluated the pipelines on a fixed 80/20 train-test split under the current three-class severity formulation.

### 2.1 Classical Baseline (TF-IDF Logistic Regression)
*The sparse-matrix representation performs exceptionally well across 3 classes, likely due to distinct, rigid vocabulary deviations found between ideation forums (r/SuicideWatch) versus general depression boards.*

- **Overall Accuracy:** 78.7%
- **Macro Avg F1-Score:** 0.74
- **Weighted Avg F1-Score:** 0.79
- **Precision (Severe Ideation):** 65% | **Recall:** 67%

### 2.2 Deep Representation (TwitterRoBERTa Random Forest)
*Swapping from the hospital-trained Bio_ClinicalBERT to the social-media-trained TwitterRoBERTa improved alignment with informal online language and slang, making the dense representation track more appropriate for Reddit-style text.*

- **Overall Accuracy:** 74.2%
- **Macro Avg F1-Score:** 0.67
- **Weighted Avg F1-Score:** 0.73
- **Precision (Severe Ideation):** 65% | **Recall:** 52%

### 2.3 Explainable AI (SHAP Visualization)

To avoid "black-box" clinical predictions, SHAP (SHapley Additive exPlanations) integration was introduced.
Local Force Plots and Global Summary Plots were computed for the Logistic Regression pipeline. This allows practitioners to visually confirm exactly which vocabulary terms (e.g., standard symptom keywords vs ideation slang) pushed a generic user's text toward a 'Severe Ideation' classification, ensuring the model avoided overfitting.

---

## 3. Exploratory Data Analysis and Language Pattern Detection

This section directly addresses the project objective of detecting *symptom and emotional language patterns* in MDD posts. Six complementary analyses were conducted on the full corpus.

### 3.1 DSM-5 Symptom Keyword Frequency Analysis

A curated list of 24 DSM-5-aligned MDD symptom keywords (e.g., *hopeless*, *worthless*, *suicide*, *insomnia*, *fatigue*, *guilt*, *empty*, *numb*) was compiled and compared between distress-related posts and the Control class.

**Key Findings:**
- MDD posts contain significantly higher average symptom keyword counts per post compared to Control posts.
- A log-normalized heatmap reveals that keywords such as *depression*, *depressed*, *anxiety*, *die*, and *pain* have dramatically higher frequency in MDD posts while remaining comparatively rare in the Control class.
- The keyword frequency differential validates the discriminative power of the TF-IDF baseline — these domain-specific terms provide strong classification signals.

### 3.2 Word Clouds (MDD vs. Control)

Side-by-side word clouds were generated to provide an intuitive comparison of dominant distress-related vocabulary versus the Control baseline.

**Key Findings:**
- The MDD word cloud is dominated by emotionally charged terms reflecting internal psychological distress (e.g., *feel*, *life*, *want*, *know*, *time*, *people*).
- The Control word cloud shows more casual, action-oriented language characteristic of general conversation topics.

### 3.3 VADER Sentiment Score Distribution

A KDE (Kernel Density Estimate) comparison of VADER compound sentiment scores between distress-related posts and the Control baseline.

**Key Findings:**
- MDD posts show a clear leftward shift in sentiment distribution, peaking in the negative sentiment range.
- Control posts are more normally distributed around neutral-to-positive sentiment values.
- This confirms that sentiment scoring is a valuable feature for distinguishing depressive text from general text.

### 3.4 Post Length Distribution

A violin plot comparing word counts of cleaned text between distress-related posts and the Control baseline.

**Key Findings:**
- MDD posts tend to be longer on average, consistent with psychological research suggesting that depressive rumination leads to more verbose self-expression.
- The Control class shows a tighter, more compact word count distribution.

### 3.5 Top Bigrams (MDD vs. Control)

The top 15 bigrams (two-word phrases) were extracted to capture multi-word expressions that reflect distinct language patterns in distress-related text versus Control text.

**Key Findings:**
- MDD bigrams capture emotional and cognitive distress phrases characteristic of depressive narratives.
- Control bigrams reflect everyday conversational topics without clinical or emotional weight.
- Bigram analysis reinforces the effectiveness of TF-IDF with `ngram_range=(1,2)` in capturing these discriminative multi-word patterns.

### 3.6 Summary

The six EDA analyses collectively demonstrate that MDD-related Reddit posts exhibit measurably different linguistic signatures across multiple dimensions: higher symptom keyword density, more negative sentiment, greater post length, and distinctive multi-word patterns. These features align with and reinforce the classification results, confirming that the project successfully detects symptom and emotional language patterns associated with Major Depressive Disorder.

---

## 4. Automation Implementation

In alignment with Deliverable requirements, we have provisioned an execution script `src/quarterly_updater.py`. 
Utilizing the static `schedule` Python daemon, this script operates persistently to spin up the data-scraper pipelines and data cleansing functions completely autonomously every **90 days** recursively, ensuring the Machine Learning pipelines consistently ingest fresh modern syntactic data.

Additionally, a GitHub Actions CI/CD workflow (`.github/workflows/quarterly_update.yml`) provides cloud-based automation that refreshes the dataset on the 1st of January, April, July, and October without requiring local machine uptime.

