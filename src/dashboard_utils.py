from __future__ import annotations

from html import escape
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import label_theme


def inject_global_styles() -> None:
    is_dark = st.session_state.get("dark_mode", False)
    
    bg_col = "#0e1117" if is_dark else "#f4f6f9"
    text_col = "#e2e8f0" if is_dark else "#333333"
    hero_col = "#f8fafc" if is_dark else "#0f172a"
    sub_col = "#94a3b8" if is_dark else "#64748b"
    card_bg = "#262730" if is_dark else "#ffffff"
    card_border = "#3d3e45" if is_dark else "#e2e8f0"
    table_text = "#e2e8f0" if is_dark else "#334155"
    table_header = "#1e1e24" if is_dark else "#1e293b"
    table_row_border = "#3d3e45" if is_dark else "#f1f5f9"
    disabled_bg = "#1e1e24" if is_dark else "#f8fafc"
    warning_bg = "#451a1a" if is_dark else "#fef2f2"
    warning_border = "#7f1d1d" if is_dark else "#fecaca"
    warning_text = "#fca5a5" if is_dark else "#b91c1c"
    tab_bg = "#1e1e24" if is_dark else "#ffffff"

    css = f'''
<style>
    .stApp {{
        background-color: {bg_col};
        color: {text_col};
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] span {{
        color: {hero_col} !important; 
    }}
    .stDeployButton {{ 
        visibility: visible !important;
    }}
    
    [data-testid="stSidebar"] {{
        background: #1e1e24;
        border-right: 1px solid rgba(255,255,255,0.05);
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p {{
        color: #eef2ff !important;
        font-weight: 600;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stTextArea textarea {{
        background: rgba(255, 255, 255, 0.05) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
    }}
    [data-testid="stSidebar"] .stButton button[kind="primary"] {{
        background: #ef4444;
        color: white !important;
        border: none;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.25);
        font-weight: 700;
        border-radius: 8px;
    }}
    
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1400px;
    }}
    .app-title {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {hero_col};
        margin-top: 0;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }}
    .app-subtitle {{
        color: {sub_col};
        font-size: 1rem;
        margin-bottom: 2rem;
    }}
    .status-badge {{
        display: inline-flex;
        align-items: center;
        background: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }}
    .metric-card {{
        background-color: {card_bg};
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        border: 1px solid {card_border};
        text-align: center;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.06);
    }}
    .metric-label {{
        color: {sub_col} !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
        margin-bottom: 8px;
    }}
    .metric-value {{
        color: {hero_col} !important;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }}
    .metric-value.compact {{
        font-size: 1.3rem;
    }}
    .metric-helper {{
        color: {sub_col};
        font-size: 0.85rem;
        margin-top: 6px;
    }}
    
    .decision-card {{
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        border: 2px solid;
    }}
    .decision-title {{
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }}
    .decision-label {{
        font-size: 2.8rem;
        line-height: 1.2;
        font-weight: 900;
        margin-bottom: 1rem;
    }}
    .decision-description {{
        font-size: 1.05rem;
        color: {sub_col};
        margin-bottom: 1.2rem;
    }}
    .decision-score {{
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.8rem;
    }}
    .decision-footer {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    
    [data-testid="stDataFrame"] {{
        border: 1px solid {card_border} !important;
        border-radius: 12px;
        overflow: hidden;
    }}
    [data-testid="stDataFrame"] table th {{
        background-color: {table_header} !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: none !important;
        padding: 12px 16px !important;
    }}
    [data-testid="stDataFrame"] table td {{
        background-color: {card_bg} !important;
        color: {table_text} !important;
        border-bottom: 1px solid {table_row_border} !important;
        padding: 10px 16px !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {tab_bg};
        border-radius: 50px;
        border: 1px solid {card_border};
        padding: 0.5rem 1.2rem;
        color: {sub_col};
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    .stTabs [aria-selected="true"] {{
        background: {hero_col} !important;
        color: {card_bg} !important;
        border-color: {hero_col} !important;
    }}
    
    .stTextArea textarea {{
        background: {card_bg} !important;
        color: {hero_col} !important;
        border-radius: 8px;
        border: 1px solid {card_border} !important;
    }}
    .stTextArea textarea:disabled {{
        background: {disabled_bg} !important;
        opacity: 1 !important;
        color: {sub_col} !important;
        -webkit-text-fill-color: {sub_col} !important;
    }}
    
    .warning-box {{
        background: {warning_bg};
        border: 1px solid {warning_border};
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: {warning_text};
        font-weight: 600;
        margin-top: 1rem;
    }}
    
    /* ========== Theme Toggle Switch ========== */
    .st-key-theme_toggle_btn {{
        display: flex !important;
        justify-content: flex-end !important;
        padding-top: 4px !important;
    }}
    .st-key-theme_toggle_btn button {{
        background: {'#e8f0fe' if is_dark else '#e6f4ea'} !important;
        border: 1px solid {'#d2e3fc' if is_dark else '#ceead6'} !important;
        border-radius: 50px !important;
        width: 52px !important;
        height: 28px !important;
        min-height: unset !important;
        padding: 0 !important;
        cursor: pointer !important;
        position: relative !important;
        transition: all 0.35s ease !important;
        box-shadow: none !important;
    }}
    .st-key-theme_toggle_btn button:hover {{
        box-shadow: 0 2px 8px {'rgba(25,103,210,0.2)' if is_dark else 'rgba(19,115,51,0.2)'} !important;
    }}
    .st-key-theme_toggle_btn button:active {{
        transform: scale(0.96);
    }}
    .st-key-theme_toggle_btn button p {{
        font-size: 0 !important;
        line-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        color: transparent !important;
    }}
    .st-key-theme_toggle_btn button::before {{
        content: "{'☽' if is_dark else '☀'}";
        position: absolute;
        top: 3px;
        {'right: 3px; left: auto;' if is_dark else 'left: 3px;'}
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: {'#1967d2' if is_dark else '#137333'};
        color: white;
        font-size: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: visible;}} 
</style>
'''
    st.markdown(css, unsafe_allow_html=True)






def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <div style="font-size:3.5rem; line-height:1;">🧠</div>
            <h1 style="color: white; font-weight: 800; font-size: 1.8rem; margin: 10px 0 0 0;">Reddit MDD AI</h1>
            <p style="color: #abb2bf; font-size: 0.95rem; margin: 0;">Live severity inference dashboard</p>
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
        <div>
            <h1 class="app-title">{escape(title)}</h1>
            <div class="app-subtitle">{escape(subtitle)}</div>
            <div style="margin-bottom: 2.5rem;">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, helper: str) -> None:
    value_class = "metric-value compact" if len(str(value)) > 15 else "metric-value"
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
    is_dark = st.session_state.get("dark_mode", False)
    if label == "Severe Ideation":
        bg_col, border_col = ("#451a1a", "#7f1d1d") if is_dark else ("#fef2f2", "#fca5a5")
    elif label == "Moderate MDD":
        bg_col, border_col = ("#422006", "#78350f") if is_dark else ("#fffbeb", "#fcd34d")
    else:
        bg_col, border_col = ("#052e16", "#14532d") if is_dark else ("#f0fdf4", "#86efac")

    st.markdown(
        f"""
        <div class="decision-card" style="background:{bg_col}; border-color:{border_col};">
            <div class="decision-title" style="color:{theme['accent']};">⚠️ INFERRED STATE</div>
            <div class="decision-description" style="font-weight: 600; color: #64748b;">{escape(description)}</div>
            <div class="decision-score" style="color:{theme['accent']};">{confidence * 100:.1f}%</div>
            <div class="decision-label" style="color:{theme['accent']};">{escape(label)}</div>
            <div class="decision-footer" style="color:{theme['badge']};">{"IMMEDIATE REVIEW REQUIRED" if label == "Severe Ideation" else "ACADEMIC DEMO ONLY"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def start_panel(title: str, subtitle: str = "") -> None:
    # A cleaner native-looking header without the heavy panel-card background
    st.markdown(f"### {escape(title)}")
    if subtitle:
        st.markdown(f"<p style='color: #64748b; margin-bottom: 1rem;'>{escape(subtitle)}</p>", unsafe_allow_html=True)


def end_panel() -> None:
    pass


def probability_chart(probabilities: dict[str, float]):
    chart_df = pd.DataFrame(
        {"label": list(probabilities.keys()), "probability": list(probabilities.values())}
    ).sort_values("probability", ascending=True)
    color_map = {
        "Control": "#10b981", # Green
        "Moderate MDD": "#f59e0b", # Yellow
        "Severe Ideation": "#ef4444", # Red
    }
    # Standard NeuroFetal Bar chart layout
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
        margin=dict(l=20, r=40, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0" if st.session_state.get("dark_mode", False) else "#334155", size=13),
        xaxis=dict(range=[0, 1.1], tickformat=".0%", title="", visible=False),
        yaxis=dict(automargin=True, title="", tickfont=dict(color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", size=14, family="Inter, sans-serif")),
        height=260,
    )
    return fig


def render_summary_chips(items: list[tuple[str, str]]) -> None:
    # Stripped away the bulky HTML div blocks that clustered the UI to use a native dataframe/table instead.
    table_data = [{"Metric": k, "Value": v} for k, v in items]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


def benchmark_chart(benchmark_row: dict | None):
    if not benchmark_row:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text="Benchmark Unavailable", showarrow=False, font=dict(size=18, color="#64748b"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False), height=150)
        return fig
        
    df = pd.DataFrame([
        {"Metric": "Accuracy", "Value": float(benchmark_row['accuracy_mean'])},
        {"Metric": "Macro F1", "Value": float(benchmark_row['macro_f1_mean'])},
        {"Metric": "S. Precision", "Value": float(benchmark_row['precision_severe_ideation'])},
        {"Metric": "S. Recall", "Value": float(benchmark_row['recall_severe_ideation'])},
    ])
    fig = px.bar(
        df, x="Metric", y="Value", text=df["Value"].map(lambda x: f"{x:.3f}"),
        color="Metric", color_discrete_sequence=["#3b82f6", "#8b5cf6", "#ec4899", "#f43f5e"]
    )
    fig.update_traces(textposition="outside", marker_line_width=0, textfont=dict(color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", size=15, family="Inter, sans-serif"))
    fig.update_layout(
        showlegend=False, margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1.15], visible=False),
        xaxis=dict(automargin=True, title="", tickfont=dict(size=13, color="#f8fafc" if st.session_state.get("dark_mode", False) else "#334155")), height=250
    )
    return fig


def explanation_chart(explanation_rows: list[dict]):
    if not explanation_rows:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text="Detailed Sparse Token Explanation Unavailable", showarrow=False, font=dict(size=16, color="#64748b"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False), height=200)
        return fig
        
    df = pd.DataFrame(explanation_rows).sort_values("contribution", ascending=True)
    df["color"] = df["contribution"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")
    
    fig = px.bar(
        df, x="contribution", y="token", orientation="h",
        text=df["contribution"].map(lambda x: f"{x:+.3f}")
    )
    fig.update_traces(marker_color=df["color"], textposition="outside", textfont=dict(color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", size=14, family="Inter, sans-serif"))
    fig.update_layout(
        margin=dict(l=20, r=50, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(automargin=True, title="Log-Odds Contribution Impact", zeroline=True, zerolinecolor="#94a3b8", zerolinewidth=1),
        yaxis=dict(automargin=True, title="", tickfont=dict(size=14, color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", family="Inter, sans-serif")),
        height=max(300, 100 + (len(df)*30))
    )
    return fig
