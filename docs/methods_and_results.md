# Classification Methods and Results

**Course**: Big Data Analytics (BDA) — 6th Semester  
**Domain**: HDA-4 | **Group**: 4  
**Title**: Natural Language Processing of Social Media for MDD Symptom Identification

---

## 1. Methodology

The text classification phase of this project aimed to identify language patterns associated with Major Depressive Disorder (MDD) by distinguishing self-reported MDD text (from *r/depression* & *r/SuicideWatch*) against a general population baseline (from *r/CasualConversation*). 

We employed a dual-track analytical architecture, contrasting classical sparse term-frequency machine learning against modern dense-vector deep learning.

### 1.1 Baseline Track: TF-IDF + Logistic Regression
- **Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF). We extracted the top 5,000 most predictive unigram and bigram features, filtering out standard English stop words.
- **Model**: Logistic Regression utilizing a 'balanced' class-weights parameter to counteract any minor asymmetries in class distributions.
- **Hardware Profile**: Pure CPU Execution. 

### 1.2 Deep NLP Track: Bio_ClinicalBERT + Random Forest
- **Vectorization**: We passed raw text through `emilyalsentzer/Bio_ClinicalBERT`, a transformer-based masked language model pre-trained distinctly on clinical notes (MIMIC-III database). This algorithm abstracts the sentences into 768-dimensional dense vectors to encapsulate complex semantic and syntactical depressions markers natively.
- **Model**: A Random Forest classifier (100 estimators; max-depth untethered) mapped to the continuous vector mappings of the `[CLS]` token hidden states.
- **Hardware Profile**: Hybrid CPU/CUDA. Processing 10,000 records dynamically scaled to utilize NVIDIA T4 Tensor Cores via Google Colab.

---

## 2. Experimental Results

We evaluated both pipelines against a rigid 80/20 train-test split configuration (1,960 testing samples) maintaining rigorous stratified sampling matrices map.

### 2.1 Classical Baseline (TF-IDF Logistic Regression)
*The sparse-matrix representation performed exceptionally well, likely due to distinct, rigid vocabulary deviations found between mental health aid forums versus casual forums.*

- **Overall Accuracy:** 91.7%
- **MDD Class Recall:** 89%
- **MDD Class Precision:** 94%
- **Weighted F1-Score:** 0.92

### 2.2 Deep Representation (ClinicalBERT Random Forest)
*The high-density continuous embeddings successfully captured clinical nuance but slightly underperformed the distinct keyword-matching of TF-IDF, suggesting forum-specific slang outweighed biological terminology in Reddit parameters.*

- **Overall Accuracy:** 85.8%
- **MDD Class Recall:** 83%
- **MDD Class Precision:** 87%
- **Weighted F1-Score:** 0.86

### 2.3 ROC Curve Comparison

ROC curves plot the True Positive Rate against the False Positive Rate at various classification thresholds, with AUC (Area Under the Curve) providing a single-number summary of classifier quality.

Both models were plotted on a single ROC chart for direct comparison. The TF-IDF + Logistic Regression model achieved a higher AUC, consistent with its superior accuracy on this dataset.

---

## 3. Exploratory Data Analysis and Language Pattern Detection

This section directly addresses the project objective of detecting *symptom and emotional language patterns* in MDD posts. Six complementary analyses were conducted on the full corpus.

### 3.1 DSM-5 Symptom Keyword Frequency Analysis

A curated list of 24 DSM-5-aligned MDD symptom keywords (e.g., *hopeless*, *worthless*, *suicide*, *insomnia*, *fatigue*, *guilt*, *empty*, *numb*) was compiled and mapped against both the MDD and Control classes.

**Key Findings:**
- MDD posts contain significantly higher average symptom keyword counts per post compared to Control posts.
- A log-normalized heatmap reveals that keywords such as *depression*, *depressed*, *anxiety*, *die*, and *pain* have dramatically higher frequency in MDD posts while remaining comparatively rare in the Control class.
- The keyword frequency differential validates the discriminative power of the TF-IDF baseline — these domain-specific terms provide strong classification signals.

### 3.2 Word Clouds (MDD vs. Control)

Side-by-side word clouds were generated for each class to provide an intuitive visual comparison of dominant vocabulary.

**Key Findings:**
- The MDD word cloud is dominated by emotionally charged terms reflecting internal psychological distress (e.g., *feel*, *life*, *want*, *know*, *time*, *people*).
- The Control word cloud shows more casual, action-oriented language characteristic of general conversation topics.

### 3.3 VADER Sentiment Score Distribution

A KDE (Kernel Density Estimate) comparison of VADER compound sentiment scores between classes.

**Key Findings:**
- MDD posts show a clear leftward shift in sentiment distribution, peaking in the negative sentiment range.
- Control posts are more normally distributed around neutral-to-positive sentiment values.
- This confirms that sentiment scoring is a valuable feature for distinguishing depressive text from general text.

### 3.4 Post Length Distribution

A violin plot comparing word counts of cleaned text between the two classes.

**Key Findings:**
- MDD posts tend to be longer on average, consistent with psychological research suggesting that depressive rumination leads to more verbose self-expression.
- The Control class shows a tighter, more compact word count distribution.

### 3.5 Top Bigrams (MDD vs. Control)

The top 15 bigrams (two-word phrases) were extracted for each class to capture multi-word expressions that reflect distinct language patterns.

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

