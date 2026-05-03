"""Configuration Streamlit (page_config) + CSS global."""
import streamlit as st


def apply_page_config_and_style():
    """Applique la configuration de page et le CSS global. À appeler une seule fois en haut de l'app."""
    st.set_page_config(page_title="Variant Explorer", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        code, .stCode { font-family: 'JetBrains Mono', monospace; }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460; border-radius: 12px; padding: 16px 20px; color: white;
        }
        [data-testid="stMetric"] label { color: #a8b2d1 !important; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #64ffda !important; }
        [data-testid="stSidebar"] { background: #0a192f; }
        [data-testid="stSidebar"] .stMarkdown { color: #ccd6f6; }
        .main-title {
            background: linear-gradient(90deg, #64ffda, #48b1bf);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.4rem; font-weight: 700; margin-bottom: 0;
        }
        .sub-title { color: #8892b0; font-size: 1.05rem; margin-top: 0; }
        .feat-up { color: #ff6b6b; font-weight: 600; }
        .feat-down { color: #4ecdc4; font-weight: 600; }
        .ai-interpretation {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border: 1px solid #64ffda33; border-radius: 12px;
            padding: 24px; margin: 16px 0; line-height: 1.7;
        }
        .ai-interpretation h4 { color: #64ffda; }
    </style>
    """, unsafe_allow_html=True)
