import re

with open("src/dashboard_utils.py", "r", encoding="utf-8") as f:
    code = f.read()

css_start = code.find('GLOBAL_CSS = """')
css_end = code.find('"""', css_start + 16) + 3

new_css_func = """def inject_global_styles() -> None:
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
    
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: visible;}} 
</style>
'''
    st.markdown(css, unsafe_allow_html=True)
"""

code = code[:css_start] + new_css_func + code[css_end:]

code = code.replace(
'''def inject_global_styles() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)''', '')

code = code.replace(
'''        font=dict(color="#334155", size=13),''',
'''        font=dict(color="#e2e8f0" if st.session_state.get("dark_mode", False) else "#334155", size=13),'''
)

code = code.replace(
'''        yaxis=dict(automargin=True, title="", tickfont=dict(color="#0f172a", size=14, family="Inter, sans-serif")),''',
'''        yaxis=dict(automargin=True, title="", tickfont=dict(color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", size=14, family="Inter, sans-serif")),'''
)

code = code.replace(
'''    fig.update_traces(textposition="outside", marker_line_width=0, textfont=dict(color="#0f172a", size=15, family="Inter, sans-serif"))''',
'''    fig.update_traces(textposition="outside", marker_line_width=0, textfont=dict(color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", size=15, family="Inter, sans-serif"))'''
)

code = code.replace(
'''        xaxis=dict(automargin=True, title="", tickfont=dict(size=13, color="#334155")), height=250''',
'''        xaxis=dict(automargin=True, title="", tickfont=dict(size=13, color="#f8fafc" if st.session_state.get("dark_mode", False) else "#334155")), height=250'''
)

code = code.replace(
'''    fig.update_traces(marker_color=df["color"], textposition="outside", textfont=dict(color="#0f172a", size=14, family="Inter, sans-serif"))''',
'''    fig.update_traces(marker_color=df["color"], textposition="outside", textfont=dict(color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", size=14, family="Inter, sans-serif"))'''
)

code = code.replace(
'''        yaxis=dict(automargin=True, title="", tickfont=dict(size=14, color="#0f172a", family="Inter, sans-serif")),''',
'''        yaxis=dict(automargin=True, title="", tickfont=dict(size=14, color="#f8fafc" if st.session_state.get("dark_mode", False) else "#0f172a", family="Inter, sans-serif")),'''
)

# Decision card overrides for dark mode
code = code.replace(
'''    # Adjusting for NeuroFetal solid sleek backgrounds
    if label == "Severe Ideation":
        bg_col, border_col = "#fef2f2", "#fca5a5"
    elif label == "Moderate MDD":
        bg_col, border_col = "#fffbeb", "#fcd34d"
    else:
        bg_col, border_col = "#f0fdf4", "#86efac"''',
'''    is_dark = st.session_state.get("dark_mode", False)
    if label == "Severe Ideation":
        bg_col, border_col = ("#451a1a", "#7f1d1d") if is_dark else ("#fef2f2", "#fca5a5")
    elif label == "Moderate MDD":
        bg_col, border_col = ("#422006", "#78350f") if is_dark else ("#fffbeb", "#fcd34d")
    else:
        bg_col, border_col = ("#052e16", "#14532d") if is_dark else ("#f0fdf4", "#86efac")''')

with open("src/dashboard_utils.py", "w", encoding="utf-8") as f:
    f.write(code)

with open("app.py", "r", encoding="utf-8") as f:
    app_code = f.read()

toggle_code = '''
# Dark Mode Toggle placement at top right
col1, col2 = st.columns([0.85, 0.15])
with col2:
    st.toggle("🌙 Dark Mode", key="dark_mode")

inject_global_styles()'''

if "inject_global_styles()" in app_code and "st.toggle" not in app_code:
    app_code = app_code.replace("inject_global_styles()", toggle_code)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(app_code)
