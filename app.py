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
selected_model_info = available_models[selected_model_key]

sample_name = st.sidebar.selectbox("Load sample text", list(sample_inputs.keys()))
if st.sidebar.button("Use selected sample", use_container_width=True):
    st.session_state.input_text = sample_inputs[sample_name]

st.sidebar.markdown("### Text Input")
text_value = st.sidebar.text_area(
    "Reddit-style post text",
    value=st.session_state.input_text,
    height=260,
    key="sidebar_text_area",
)
st.session_state.input_text = text_value

predict_clicked = st.sidebar.button("Run Live Inference", type="primary", use_container_width=True)

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
prediction_error = None

if predict_clicked:
    if not st.session_state.input_text.strip():
        prediction_error = "Please enter some text before running inference."
    else:
        try:
            prediction = predict_text(selected_model_key, st.session_state.input_text)
        except Exception as exc:  # pragma: no cover - UI guard
            prediction_error = str(exc)

if prediction_error:
    st.error(prediction_error)

if prediction is None:
    st.info("Use the left sidebar to load example text or paste your own Reddit-style post, then click `Run Live Inference`.")

    start_panel("Model Export Status", "This dashboard is ready, but live model files may still need to be generated from the Colab notebook.")
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
    primary_cols = st.columns([1.05, 1.95])
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
        start_panel("Live Probability Analysis", "The chart below shows the full class distribution returned by the selected inference model.")
        st.plotly_chart(probability_chart(prediction.probabilities), use_container_width=True)
        end_panel()

    summary_cols = st.columns([1.05, 1.95])
    with summary_cols[0]:
        start_panel("Model Summary", "Quick benchmark snapshot and run metadata for the selected model.")
        summary_table = benchmark_table_row(prediction.benchmark_row)
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
        st.markdown("**Runtime notes**")
        for note in prediction.runtime_notes:
            st.markdown(f"- {note}")
        end_panel()

    with summary_cols[1]:
        start_panel("Explainable AI (XAI) Analysis", "This section highlights the input cues that most influenced the model decision.")
        st.dataframe(explanation_table(prediction.explanation_rows), use_container_width=True, hide_index=True)
        if not prediction.explanation_rows:
            st.caption("Runtime token explanation is currently available for the saved sparse models. The RoBERTa path shows preprocessing/runtime notes instead.")
        end_panel()

    detail_cols = st.columns([1.1, 1.9])
    with detail_cols[0]:
        start_panel("Inference Summary", "A compact prediction record inspired by the clinical-style summary card in the reference dashboard.")
        summary_rows = [
            {"Metric": "Predicted Label", "Value": prediction.predicted_label},
            {"Metric": "Confidence", "Value": f"{prediction.confidence * 100:.2f}%"},
            {"Metric": "Model", "Value": prediction.model_display_name},
            {"Metric": "Words", "Value": str(input_stats["words"])},
            {"Metric": "Characters", "Value": str(input_stats["characters"])},
        ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        end_panel()

    with detail_cols[1]:
        start_panel("Preprocessing Preview", "The dashboard keeps the text visible so the demo audience can connect raw input with model behavior.")
        st.text_area("Submitted text", value=st.session_state.input_text, height=180, disabled=True)
        st.text_area("Model-facing text", value=prediction.cleaned_text, height=180, disabled=True)
        end_panel()

    start_panel("Validation and Limitations", "This dashboard is intended for live coursework demonstration, not for healthcare deployment.")
    st.markdown(
        """
        - The labels are subreddit-derived academic proxy labels, not medical diagnoses.
        - `TF-IDF + Logistic Regression` is the recommended default because it is the strongest and most reliable saved live model.
        - `TwitterRoBERTa + Random Forest` is available as an advanced model option when its saved classifier artifact is present.
        - The committed benchmark metrics come from `data/processed/results_summary.csv`.
        """
    )
    end_panel()
