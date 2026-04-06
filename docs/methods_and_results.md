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

---

## 3. Automation Implementation
In alignment with Deliverable requirements, we have provisioned an execution script `src/quarterly_updater.py`. 
Utilizing the static `schedule` Python daemon, this script operates persistently to spin up the data-scraper pipelines and data cleansing functions completely autonomously every **90 days** recursively, ensuring the Machine Learning pipelines consistently ingest fresh modern syntactic data.
