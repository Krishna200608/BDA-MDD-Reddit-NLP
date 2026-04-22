# Model Artifacts

This directory stores deployment-ready inference artifacts exported from
`notebooks/02_text_classification_models.ipynb`.

Expected files after the Colab T4 rerun:

- `tfidf_logreg_pipeline.joblib`
- `tfidf_linearsvc_pipeline.joblib`
- `roberta_rf_classifier.joblib`
- `model_metadata.json`
- `class_labels.json`

These files are intended for the local Streamlit dashboard in `app.py`.
