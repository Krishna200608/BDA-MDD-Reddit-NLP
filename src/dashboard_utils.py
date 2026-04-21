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
        background: linear-gradient(180deg, #f6f8fc 0%, #eef3fb 100%);
        color: #101828;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #23242d 0%, #1f2028 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #f4f6fb;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.92);
        border-left: 6px solid #1473e6;
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 18px 40px rgba(20, 33, 61, 0.08);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #123d7a;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #667085;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    .badge-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.35rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #eef6ff;
        color: #2459c4;
        border: 1px solid #d5e6ff;
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 22px;
        padding: 1.25rem 1rem;
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.06);
        min-height: 145px;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.77rem;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-helper {
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: 0.45rem;
    }
    .decision-card {
        border-radius: 26px;
        padding: 1.8rem 1.5rem;
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
        border: 2px solid;
        text-align: center;
        min-height: 380px;
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
        color: #475467;
        margin-bottom: 1.2rem;
    }
    .decision-score {
        font-size: 3.2rem;
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
        padding: 1.15rem 1.15rem 0.9rem;
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #101828;
        margin-bottom: 0.25rem;
    }
    .panel-subtitle {
        color: #667085;
        margin-bottom: 0.9rem;
    }
    .sidebar-brand {
        margin-bottom: 1.4rem;
    }
    .sidebar-title {
        font-size: 2rem;
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
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
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
        plot_bgcolor="rgba(248,250,252,0.65)",
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
        yaxis=dict(title=""),
        height=320,
    )
    return fig


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
