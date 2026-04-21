from __future__ import annotations

import streamlit as st

from src.dashboard_utils import (
    benchmark_table_row,
    end_panel,
    explanation_table,
    inject_global_styles,
    probability_chart,
    render_decision_card,
    render_hero,
    render_metric_card,
    render_sidebar_brand,
    render_summary_chips,
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

sample_name = st.sidebar.selectbox("Load sample text", list(sample_inputs.keys()))
if st.sidebar.button("Use selected sample", use_container_width=True):
    st.session_state.input_text = sample_inputs[sample_name]
    st.session_state.prediction_result = None
    st.session_state.prediction_error = None

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

top_cols = st.columns(4)
with top_cols[0]:
    render_metric_card("Model", selected_model_display, "Selected for this inference run")
with top_cols[1]:
    render_metric_card("Words", str(input_stats["words"]), "Input length")
with top_cols[2]:
    render_metric_card("Characters", str(input_stats["characters"]), "Raw text size")
with top_cols[3]:
    render_metric_card(
        "Runtime",
        "Ready" if selected_model_info.get("is_available") else "Waiting",
        "Saved artifact status",
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
    st.info("Use the left sidebar to load an example or paste your own Reddit-style post, then click `Run Live Inference`.")

    start_panel("Model Export Status", "The dashboard shell is ready. This panel shows whether the live inference artifacts are present locally.")
    status_rows = []
    for model_key, model_info in available_models.items():
        status_rows.append(
            {
                "Model": model_info["display_name"],
                "Artifact Ready": "Yes" if model_info.get("is_available") else "No",
                "Artifact Path": model_info.get("resolved_artifact_path", ""),
            }
        )
    st.dataframe(status_rows, use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="warning-box">
            Rerun the final Colab notebook, save the models into <code>models/</code>, push to GitHub, and then pull locally to enable live inference.
        </div>
        """,
        unsafe_allow_html=True,
    )
    end_panel()
else:
    overview_cols = st.columns(4)
    with overview_cols[0]:
        render_metric_card("Predicted Label", prediction.predicted_label, "Current live model output")
    with overview_cols[1]:
        render_metric_card("Confidence", f"{prediction.confidence * 100:.1f}%", "Top class probability")
    with overview_cols[2]:
        render_metric_card("Model", prediction.model_display_name, "Active inference backend")
    with overview_cols[3]:
        render_metric_card("Input Size", f"{input_stats['words']} words", f"{input_stats['characters']} characters")

    primary_cols = st.columns([1.0, 1.7])
    with primary_cols[0]:
        descriptions = {
            "Control": "Language appears closer to neutral or everyday conversational patterns.",
            "Moderate MDD": "Language suggests distress and depressive symptom patterns without the strongest ideation cues.",
            "Severe Ideation": "Language contains stronger high-risk markers and requires careful academic interpretation.",
        }
        render_decision_card(
            prediction.predicted_label,
            prediction.confidence,
            descriptions.get(prediction.predicted_label, "Predicted severity class from saved NLP model."),
        )

    with primary_cols[1]:
        start_panel("Live Probability Analysis", "A compact probability view designed for quick explanation during a live classroom demo.")
        render_summary_chips(
            [
                ("Default Model", "TF-IDF + Logistic Regression"),
                ("Runtime", "Saved artifact loaded"),
                ("Best Holdout", "0.7841 accuracy" if prediction.benchmark_row else "Benchmark unavailable"),
            ]
        )
        st.plotly_chart(probability_chart(prediction.probabilities), use_container_width=True)
        st.caption("The bar chart shows how strongly the selected model leans toward each severity class for the current input.")
        end_panel()

    analysis_tabs = st.tabs(["Model Snapshot", "Explainability", "Input Trace"])

    with analysis_tabs[0]:
        summary_cols = st.columns([1.05, 1.15])
        with summary_cols[0]:
            start_panel("Model Summary", "Benchmark snapshot for the currently selected model.")
            summary_table = benchmark_table_row(prediction.benchmark_row)
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
            if prediction.model_key == "roberta_rf":
                st.caption("First RoBERTa inference may take longer while the Hugging Face encoder loads from cache.")
            end_panel()

        with summary_cols[1]:
            start_panel("Inference Summary", "Compact run metadata inspired by the clean clinical-summary card pattern from the reference dashboard.")
            render_summary_chips(
                [
                    ("Predicted Label", prediction.predicted_label),
                    ("Confidence", f"{prediction.confidence * 100:.2f}%"),
                    ("Words", str(input_stats["words"])),
                    ("Characters", str(input_stats["characters"])),
                ]
            )
            st.markdown("**Runtime notes**")
            for note in prediction.runtime_notes:
                st.markdown(f"- {note}")
            end_panel()

    with analysis_tabs[1]:
        start_panel("Explainable AI (XAI) Analysis", "A lightweight runtime explanation view for the saved models.")
        st.dataframe(explanation_table(prediction.explanation_rows), use_container_width=True, hide_index=True)
        if not prediction.explanation_rows:
            st.caption("Runtime token explanation is currently available for the sparse models. The RoBERTa path surfaces preprocessing/runtime notes instead.")
        end_panel()

    with analysis_tabs[2]:
        start_panel("Preprocessing Preview", "Show raw input and model-facing text without crowding the main dashboard view.")
        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.text_area("Submitted text", value=st.session_state.input_text, height=190, disabled=True)
        with preview_cols[1]:
            st.text_area("Model-facing text", value=prediction.cleaned_text, height=190, disabled=True)
        end_panel()

    with st.expander("Validation and Limitations", expanded=False):
        st.markdown(
            """
            - The labels are subreddit-derived academic proxy labels, not medical diagnoses.
            - `TF-IDF + Logistic Regression` is the recommended default because it is the strongest and most reliable saved live model.
            - `TwitterRoBERTa + Random Forest` is available as an advanced model option when its saved classifier artifact is present.
            - The committed benchmark metrics come from `data/processed/results_summary.csv`.
            - This dashboard is for live coursework demonstration, not healthcare deployment.
            """
        )
