from __future__ import annotations

import streamlit as st

from src.dashboard_utils import (
    benchmark_chart,
    end_panel,
    explanation_chart,
    inject_global_styles,
    probability_chart,
    render_decision_card,
    render_hero,
    render_metric_card,
    render_sidebar_brand,
    start_panel,
)
from src.inference import (
    get_available_models,
    get_dashboard_summary,
    get_default_model_key,
    get_sample_inputs,
    predict_text,
    summarize_input_text,
)


st.set_page_config(
    page_title="Reddit MDD NLP Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Dark Mode Toggle placement at top right
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

col1, col2 = st.columns([0.92, 0.08])
with col2:
    st.button(" ", key="theme_toggle_btn", on_click=toggle_theme)

inject_global_styles()
render_sidebar_brand()

dashboard_summary = get_dashboard_summary()
available_models = get_available_models()
default_model_key = get_default_model_key()
sample_inputs = get_sample_inputs()

if "input_text" not in st.session_state:
    st.session_state.input_text = next(iter(sample_inputs.values()))
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None
if "selected_model_key" not in st.session_state:
    st.session_state.selected_model_key = default_model_key

available_model_options = {
    model_info["display_name"]: model_key
    for model_key, model_info in available_models.items()
}

st.sidebar.markdown("### Inference Controls")
selected_model_display = st.sidebar.selectbox(
    "Choose model",
    list(available_model_options.keys()),
    index=max(
        0,
        list(available_model_options.values()).index(default_model_key)
        if default_model_key in available_model_options.values()
        else 0,
    ),
)
selected_model_key = available_model_options[selected_model_display]
if st.session_state.selected_model_key != selected_model_key:
    st.session_state.selected_model_key = selected_model_key
    st.session_state.prediction_result = None
    st.session_state.prediction_error = None
selected_model_info = available_models[selected_model_key]

def auto_load_sample():
    st.session_state.input_text = sample_inputs[st.session_state.sample_dropdown]
    st.session_state.sidebar_text_area = sample_inputs[st.session_state.sample_dropdown]
    st.session_state.prediction_result = None
    st.session_state.prediction_error = None

st.sidebar.selectbox(
    "Load sample text",
    list(sample_inputs.keys()),
    key="sample_dropdown",
    on_change=auto_load_sample
)

st.sidebar.markdown("### Text Input")
text_value = st.sidebar.text_area(
    "Reddit-style post text",
    value=st.session_state.input_text,
    height=260,
    key="sidebar_text_area",
)
st.session_state.input_text = text_value

predict_clicked = st.sidebar.button("Run Live Inference", type="primary", use_container_width=True)
reset_clicked = st.sidebar.button("Clear Result", use_container_width=True)

st.sidebar.markdown("### Deployment Status")
if selected_model_info.get("is_available"):
    st.sidebar.success("Saved artifact found")
else:
    st.sidebar.error("Saved artifact missing")
    st.sidebar.caption("Rerun the Colab notebook export cells and pull the repo after push.")

st.sidebar.markdown(
    """
    ### Demo Guidance
    - Default live model should be `TF-IDF + Logistic Regression`
    - Streamlit is for academic demonstration only
    - Do not interpret outputs as clinical diagnosis
    """
)

hero_badges = list(dashboard_summary["system_badges"])
hero_badges.append(selected_model_display)
if selected_model_info.get("is_available"):
    hero_badges.append("Artifact Loaded")
else:
    hero_badges.append("Awaiting Model Export")

render_hero(
    dashboard_summary["title"],
    dashboard_summary["subtitle"],
    hero_badges,
)

input_stats = summarize_input_text(st.session_state.input_text)

# Ensure proper metrics alignment like in NeuroFetal
st.divider()

top_cols = st.columns(4)
with top_cols[0]:
    render_metric_card("Model Configuration", selected_model_display, "Selected active engine")
with top_cols[1]:
    render_metric_card("Post Valid Words", str(input_stats["words"]), "Input semantic length")
with top_cols[2]:
    render_metric_card("Text Characters", str(input_stats["characters"]), "Raw byte stream size")
with top_cols[3]:
    render_metric_card(
        "Deployment System",
        "Ready" if selected_model_info.get("is_available") else "Missing",
        "Saved `.joblib` artifact status",
    )

prediction = None

if predict_clicked:
    if not st.session_state.input_text.strip():
        st.session_state.prediction_result = None
        st.session_state.prediction_error = "Please enter some text before running inference."
    else:
        try:
            st.session_state.prediction_result = predict_text(selected_model_key, st.session_state.input_text)
            st.session_state.prediction_error = None
        except Exception as exc:  # pragma: no cover - UI guard
            st.session_state.prediction_result = None
            st.session_state.prediction_error = str(exc)

if reset_clicked:
    st.session_state.prediction_result = None
    st.session_state.prediction_error = None

prediction = st.session_state.prediction_result
prediction_error = st.session_state.prediction_error

if prediction_error:
    st.error(prediction_error)

if prediction is None:
    st.divider()
    st.info("👈 Use the left sidebar to load an example or paste your own Reddit-style post, then click **Run Live Inference**.")
    start_panel("Model Export Status")
    status_rows = []
    for model_key, model_info in available_models.items():
        status_rows.append(
            {
                "Model Arch": model_info["display_name"],
                "Is Ready?": "Yes" if model_info.get("is_available") else "No",
                "OS Path": model_info.get("resolved_artifact_path", ""),
            }
        )
    st.dataframe(status_rows, use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="warning-box">
            Awaiting inference instructions!
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.divider()

    # The Decision block & Plotly chart side-by-side using vertical alignment
    primary_cols = st.columns([1.0, 2.0], vertical_alignment="center")
    
    with primary_cols[0]:
        descriptions = {
            "Control": "Language appears closer to neutral or everyday conversational patterns.",
            "Moderate MDD": "Language suggests distress and depressive symptom patterns without strongest ideation cues.",
            "Severe Ideation": "Language contains stronger high-risk markers and requires careful academic interpretation.",
        }
        render_decision_card(
            prediction.predicted_label,
            prediction.confidence,
            descriptions.get(prediction.predicted_label, "Predicted severity class."),
        )

    with primary_cols[1]:
        start_panel("Probability Analysis", "A compact geometric view showing how strongly the backend leans toward each tier.")
        st.plotly_chart(probability_chart(prediction.probabilities), use_container_width=True, theme=None)

    st.divider()

    # Organized spacious Tabs
    analysis_tabs = st.tabs(["Snapshot & Validation", "Explainable AI (XAI)", "Raw Input Vectors"])

    with analysis_tabs[0]:
        summary_cols = st.columns(2)
        with summary_cols[0]:
            start_panel("Evaluation Benchmark Snapshot")
            st.plotly_chart(benchmark_chart(prediction.benchmark_row), use_container_width=True, theme=None)
            if prediction.model_key == "roberta_rf":
                st.caption("First RoBERTa inference may natively lag due to huggingface cache loads.")
        
        with summary_cols[1]:
            start_panel("Inference Profile Meta")
            meta_data = [
                {"Parameter": "Predicted Tier", "Recorded Info": prediction.predicted_label},
                {"Parameter": "Mathematical Confidence", "Recorded Info": f"{prediction.confidence * 100:.2f}%"},
                {"Parameter": "Active Token Pool", "Recorded Info": str(input_stats["words"])},
                {"Parameter": "Artifact Key", "Recorded Info": selected_model_key},
            ]
            import pandas as pd
            st.dataframe(pd.DataFrame(meta_data), use_container_width=True, hide_index=True)

    with analysis_tabs[1]:
        start_panel("Model Attention / Log-Odds Trace", "Extracting exactly which word-level identifiers triggered the severity scale threshold.")
        st.plotly_chart(explanation_chart(prediction.explanation_rows), use_container_width=True, theme=None)
        if not prediction.explanation_rows:
            st.caption("⚠️ Dense Transformer features (like TwitterRoBERTa) cannot physically map back to specific token words linearly. Select TF-IDF for full XAI traces.")

    with analysis_tabs[2]:
        start_panel("System Text Pre-Processor")
        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.text_area("Submitted User Body", value=st.session_state.input_text, height=210, disabled=True)
        with preview_cols[1]:
            st.text_area("Final Cleaned Model-Layer Trace", value=prediction.cleaned_text, height=210, disabled=True)

    with st.expander("Validation and Limitations", expanded=False):
        st.markdown(
            """
            - The labels are **subreddit-derived academic proxy labels**, not clinical medical diagnoses.
            - `TF-IDF + Logistic Regression` is the **core** architecture supporting deep single-token mathematical XAI.
            - `TwitterRoBERTa + Random Forest` acts as our baseline semantic dense comparison.
            - The committed benchmark metrics come from `data/processed/results_summary.csv`.
            """
        )
