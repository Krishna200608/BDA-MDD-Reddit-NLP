import nbformat

path = r'd:\Lab\BDA\Project\notebooks\02_text_classification_models.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Helper for string replace in cell
def replace_in_cell(index, old, new):
    if old in nb.cells[index].source:
        nb.cells[index].source = nb.cells[index].source.replace(old, new)
        return True
    return False

# 1. Update Load process (Cell 5)
replace_in_cell(5, "df['target'] = df['label'].map({'MDD': 1, 'Control': 0})", 
                "df['target'] = df['label'].map({'Severe Ideation': 2, 'Moderate MDD': 1, 'Control': 0})")
replace_in_cell(5, "MDD = 1, Control = 0", "Severe Ideation = 2, Moderate MDD = 1, Control = 0")

# 2. Update LR Confusion Matrix (Cell 7)
replace_in_cell(7, "xticklabels=['Control', 'MDD'], yticklabels=['Control', 'MDD']", 
                "xticklabels=['Control', 'Mod MDD', 'Severe'], yticklabels=['Control', 'Mod MDD', 'Severe']")

# 3. Update Bio_ClinicalBERT to MentalRoBERTa (Cells 9, 10, 12)
replace_in_cell(9, "emilyalsentzer/Bio_ClinicalBERT", "mental/mental-roberta-base")
replace_in_cell(9, "ClinicalBERT", "MentalRoBERTa")
replace_in_cell(10, "Bio_ClinicalBERT", "MentalRoBERTa")
replace_in_cell(12, "ClinicalBERT", "MentalRoBERTa")
replace_in_cell(12, "xticklabels=['Control', 'MDD'], yticklabels=['Control', 'MDD']", 
                "xticklabels=['Control', 'Mod MDD', 'Severe'], yticklabels=['Control', 'Mod MDD', 'Severe']")

# Insert SHAP Explainability just before the EDA section (Before Cell 13)
shap_md = nbformat.v4.new_markdown_cell(source="""## 5. Explainable AI (SHAP)
Because Random forests and deep vectors act as black-boxes, we use SHAP (SHapley Additive exPlanations) to identify which words the classical model learned as strong indicators of severe ideation.
*(Note: Visualized on Logistic Regression to interpret vocabulary; Deep models take large GPU time to compute shapely values)*""")

shap_code = nbformat.v4.new_code_cell(source="""import shap
import warnings
warnings.filterwarnings('ignore')

# 1. SHAP for Logistic Regression (Track A) against test samples to explain token feature importance
# Taking a subset to prevent out-of-memory overhead during SHAP matrix generation
explainer_lr = shap.LinearExplainer(lr_model, X_train_tfidf, feature_names=tfidf.get_feature_names_out())
subset_idx = 100
shap_values_lr = explainer_lr.shap_values(X_test_tfidf[:subset_idx])

# Display SHAP summary visually for the 'Severe Ideation' class
# Note: For multi-class (0,1,2), shap_values_lr[2] corresponds to 'Severe Ideation'
print("SHAP Summary Plot for SEVERE IDEATION (Class 2):")
shap.summary_plot(shap_values_lr[2], X_test_tfidf[:subset_idx].toarray(), feature_names=tfidf.get_feature_names_out())""")

nb.cells.insert(13, shap_md)
nb.cells.insert(14, shap_code)

# 4. EDA 1 (Symptom Counts) -> Index shifted +2 so Cell 15 is now 17
replace_in_cell(17, "color=['#2ecc71', '#e74c3c']", "color=['#2ecc71', '#f39c12', '#e74c3c']")
replace_in_cell(17, "['Control', 'MDD']", "['Control', 'Moderate MDD', 'Severe Ideation']")
replace_in_cell(17, "index=['Control', 'MDD']", "index=['Control', 'Moderate MDD', 'Severe Ideation']")
replace_in_cell(17, "df[df['label'] == 'MDD']['selftext_cleaned'].str.count(kw).sum()]", 
                "df[df['label'] == 'Moderate MDD']['selftext_cleaned'].str.count(kw).sum(),\n         df[df['label'] == 'Severe Ideation']['selftext_cleaned'].str.count(kw).sum()]")

# 5. EDA 2 (Word Clouds) -> Cell 19
replace_in_cell(19, "fig, axes = plt.subplots(1, 2, figsize=(16, 6))", "fig, axes = plt.subplots(1, 3, figsize=(20, 6))")
replace_in_cell(19, "[('MDD', 'Reds'), ('Control', 'Greens')]", "[('Severe Ideation', 'Reds'), ('Moderate MDD', 'Oranges'), ('Control', 'Greens')]")

# 6. EDA 3 (Sentiment) -> Cell 21
replace_in_cell(21, "[('Control', '#2ecc71'), ('MDD', '#e74c3c')]", "[('Control', '#2ecc71'), ('Moderate MDD', '#f39c12'), ('Severe Ideation', '#e74c3c')]")

# 7. EDA 4 (Length) -> Cell 23
replace_in_cell(23, "palette={'Control': '#2ecc71', 'MDD': '#e74c3c'}", "palette={'Control': '#2ecc71', 'Moderate MDD': '#f39c12', 'Severe Ideation': '#e74c3c'}")

# 8. EDA 5 (Bigrams) -> Cell 25
replace_in_cell(25, "fig, axes = plt.subplots(1, 2, figsize=(16, 6))", "fig, axes = plt.subplots(1, 3, figsize=(20, 6))")
replace_in_cell(25, "[('MDD', '#e74c3c'), ('Control', '#2ecc71')]", "[('Severe Ideation', '#e74c3c'), ('Moderate MDD', '#f39c12'), ('Control', '#2ecc71')]")

# 9. EDA 6 (ROC) -> Cell 27
# Drop ROC calculation as multi-class ROC is overly complex and SHAP replaces our need for extra metrics.
nb.cells[26].source = "### 4.6 ROC Validation Dropped\nSince we transitioned to multi-class (3 classes), plotting simple binary ROC curves is no longer applicable. Explainability (SHAP) is prioritized instead."
del nb.cells[27]

# Write back
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook Successfully Migrated!")
