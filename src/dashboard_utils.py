from __future__ import annotations

from html import escape
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import label_theme


GLOBAL_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.08), transparent 30%),
            linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
        color: #0f172a;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #21242e 0%, #171922 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] * {
        color: #f4f6fb;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption {
        color: #eef2ff !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stSidebar"] .stButton button {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #ff5a5f 0%, #ff3f50 100%);
        color: white !important;
        border: none;
        box-shadow: 0 12px 24px rgba(255, 79, 100, 0.25);
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.6rem;
        max-width: 1320px;
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.96);
        border-left: 6px solid #1473e6;
        border-radius: 26px;
        padding: 1.45rem 1.7rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        color: #123d7a;
        margin-bottom: 0.15rem;
    }
    .hero-subtitle {
        color: #475467;
        font-size: 0.98rem;
        margin-bottom: 0.85rem;
    }
    .badge-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.25rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #f5f9ff;
        color: #2459c4;
        border: 1px solid #d7e7ff;
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 22px;
        padding: 1.05rem 1rem;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        min-height: 132px;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.77rem;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.12;
    }
    .metric-value.compact {
        font-size: 1.2rem;
        line-height: 1.28;
    }
    .metric-helper {
        color: #475467;
        font-size: 0.9rem;
        margin-top: 0.45rem;
    }
    .decision-card {
        border-radius: 26px;
        padding: 1.8rem 1.5rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
        border: 2px solid;
        text-align: center;
        min-height: 345px;
        display: flex;
        justify-content: center;
        flex-direction: column;
    }
    .decision-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }
    .decision-label {
        font-size: 2.35rem;
        line-height: 1.15;
        font-weight: 900;
        margin-bottom: 1rem;
    }
    .decision-description {
        font-size: 1.03rem;
        color: #334155;
        margin-bottom: 1.2rem;
    }
    .decision-score {
        font-size: 3.0rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.8rem;
    }
    .decision-footer {
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .panel-card {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 24px;
        padding: 1.05rem 1.15rem 1rem;
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 0.95rem;
    }
    .panel-title {
        font-size: 1.28rem;
        font-weight: 800;
        color: #101828;
        margin-bottom: 0.25rem;
    }
    .panel-subtitle {
        color: #475467;
        margin-bottom: 0.8rem;
    }
    .sidebar-brand {
        margin-bottom: 1.2rem;
    }
    .sidebar-title {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.65rem 0 0.2rem;
    }
    .sidebar-caption {
        color: #c4cad7;
        font-size: 0.95rem;
    }
    .small-note {
        color: #667085;
        font-size: 0.9rem;
    }
    .warning-box {
        background: rgba(254, 242, 242, 0.9);
        border: 1px solid #fecaca;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        color: #991b1b;
        font-weight: 600;
    }
    .summary-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 0.25rem;
    }
    .summary-chip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.75rem 0.9rem;
        min-width: 150px;
    }
    .summary-chip-label {
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .summary-chip-value {
        font-size: 1.05rem;
        font-weight: 750;
        color: #0f172a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f8fafc;
        border-radius: 999px;
        border: 1px solid #dbe4f0;
        padding: 0.4rem 0.9rem;
        color: #334155;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        border-color: #bfdbfe !important;
    }
    .stTextArea textarea {
        color: #0f172a !important;
        background: rgba(255,255,255,0.9) !important;
    }
    .stTextArea textarea:disabled {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }
    .stMarkdown p, .stMarkdown li, .stCaption {
        color: #334155;
    }
</style>
"""


def inject_global_styles() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div style="font-size:4rem; line-height:1;">🧠</div>
            <div class="sidebar-title">Reddit MDD AI</div>
            <div class="sidebar-caption">Live severity inference dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badges: Iterable[str]) -> None:
    badge_html = "".join(
        f'<span class="status-badge">● {escape(str(badge))}</span>' for badge in badges
    )
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-subtitle">{escape(subtitle)}</div>
            <div class="badge-row">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, helper: str) -> None:
    value_class = "metric-value compact" if len(str(value)) > 18 else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="{value_class}">{escape(value)}</div>
            <div class="metric-helper">{escape(helper)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_card(label: str, confidence: float, description: str) -> None:
    theme = label_theme(label)
    st.markdown(
        f"""
        <div class="decision-card" style="background:{theme['soft']}; border-color:{theme['border']};">
            <div class="decision-title" style="color:{theme['accent']};">Classification Result</div>
            <div class="decision-label" style="color:{theme['accent']};">{escape(label)}</div>
            <div class="decision-description">{escape(description)}</div>
            <div class="decision-score" style="color:{theme['accent']};">{confidence * 100:.1f}%</div>
            <div class="decision-footer" style="color:{theme['badge']};">Academic Demo Use Only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def start_panel(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="panel-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{escape(title)}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def end_panel() -> None:
    return None


def probability_chart(probabilities: dict[str, float]):
    chart_df = pd.DataFrame(
        {"label": list(probabilities.keys()), "probability": list(probabilities.values())}
    ).sort_values("probability", ascending=True)
    color_map = {
        "Control": "#14b8a6",
        "Moderate MDD": "#f59e0b",
        "Severe Ideation": "#ef4444",
    }
    fig = px.bar(
        chart_df,
        x="probability",
        y="label",
        orientation="h",
        color="label",
        text=chart_df["probability"].map(lambda value: f"{value * 100:.1f}%"),
        color_discrete_map=color_map,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=13),
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability", gridcolor="#e5edf7", zeroline=False),
        yaxis=dict(title="", tickfont=dict(color="#475467")),
        height=260,
    )
    return fig


def render_summary_chips(items: list[tuple[str, str]]) -> None:
    chip_html = "".join(
        f"""
        <div class="summary-chip">
            <div class="summary-chip-label">{escape(label)}</div>
            <div class="summary-chip-value">{escape(value)}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(f'<div class="summary-chip-row">{chip_html}</div>', unsafe_allow_html=True)


def benchmark_table_row(benchmark_row: dict | None) -> pd.DataFrame:
    if not benchmark_row:
        return pd.DataFrame(
            [{"Metric": "Benchmark", "Value": "No saved benchmark row found yet"}]
        )
    return pd.DataFrame(
        [
            {"Metric": "Holdout Accuracy", "Value": f"{float(benchmark_row['accuracy_mean']):.4f}"},
            {"Metric": "Holdout Macro F1", "Value": f"{float(benchmark_row['macro_f1_mean']):.4f}"},
            {
                "Metric": "Severe Precision",
                "Value": f"{float(benchmark_row['precision_severe_ideation']):.4f}",
            },
            {
                "Metric": "Severe Recall",
                "Value": f"{float(benchmark_row['recall_severe_ideation']):.4f}",
            },
        ]
    )


def explanation_table(explanation_rows: list[dict]) -> pd.DataFrame:
    if not explanation_rows:
        return pd.DataFrame(
            [{"Token": "N/A", "Contribution": "Model-specific runtime token explanation not available"}]
        )
    return pd.DataFrame(
        [
            {
                "Token": row["token"],
                "Contribution": f"{row['contribution']:+.4f}",
                "Effect": row["direction"],
            }
            for row in explanation_rows
        ]
    )
