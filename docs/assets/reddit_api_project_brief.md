# Project Brief: Reddit MDD Corpus — Academic Research

## Institution
Indian Institute of Information Technology, Allahabad (IIIT-A)  
B.Tech CSE, 6th Semester — Big Data Analytics Course Project

## Team
| Name | Roll Number |
|---|---|
| Krishna Sikheriya | IIT2023139 |
| Priyam Jyoti Chakrabarty | IIT2023147 |
| Tavish Chawla | IIT2023150 |

## Project Title
**Natural Language Processing of Social Media for Major Depressive Disorder (MDD) Symptom Identification**

## Objective
Compile a dataset of self-reported MDD posts from Reddit for NLP-based text classification (MDD vs. control). The project aims to detect symptom and emotional language patterns using sentiment analysis and clinical text embeddings.

## API Usage
- **Read-only access** via PRAW (Python Reddit API Wrapper).
- Target subreddits: `r/depression`, `r/SuicideWatch`, `r/CasualConversation`, `r/mentalhealth`.
- Data collected: Post ID, timestamp, selftext, subreddit label.
- **No writing, posting, commenting, or voting** — purely data extraction.
- All API rate limits will be strictly respected.

## Data Handling & Ethics
- All data is **anonymized** — no usernames or PII will be stored or published.
- Dataset is used **exclusively for academic coursework**, not for commercial purposes.
- Results will be shared only within the university evaluation context.

## Tech Stack
Python, PRAW, NLTK, scikit-learn, ClinicalBERT, Jupyter Notebooks

## Contact
Krishna Sikheriya — krishnasikherya018@gmail.com
