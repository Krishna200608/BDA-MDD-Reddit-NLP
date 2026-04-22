### TASK ID HDA-4: Natural Language Processing of Social Media for MDD Symptom Identification (Reddit MDD Corpus)

**Objective**
* Compile a dataset of self-reported MDD posts for NLP-based analysis.

**ML/Data Analytics Objective**
* Classify posts as MDD vs control.
* Detect symptom and emotional language patterns.

**Data Extraction Techniques**
* Reddit API (PRAW) for r/depression, r/SuicideWatch.
* Keyword filtering and regex for self-reports.

**Example Features to Extract**
* Text embeddings (ClinicalBERT).
* Sentiment scores, symptom keyword counts.

**Deliverables 1**
* Secondary dataset in .csv with post IDs, dates, cleaned text, labels.
* Python Notebook for Reddit scraping and cleaning.
* Documentation of workflow.

**Deliverables 2**
* Python Notebook for text classification models.
* Documentation of methods and results.
* Automated pipeline for quarterly updates.