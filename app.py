"""
╔══════════════════════════════════════════════════════════════╗
║                  VARIANT EXPLORER v4.0                       ║
║          Outil interactif d'exploration de variants          ║
║              génomiques pour données de séquençage           ║
╚══════════════════════════════════════════════════════════════╝

Point d'entrée — orchestration des onglets.
La logique est dans les modules core/, ui/, tabs/.
"""
import streamlit as st

from core.styling import apply_page_config_and_style
from ui.sidebar import render_sidebar
from tabs import overview, patient, acmg, oncoprint, complications, qc

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
apply_page_config_and_style()

# ─────────────────────────────────────────────
# SIDEBAR (upload, GMT, clé API, filtres globaux)
# ─────────────────────────────────────────────
df, df_f, pathways_dict, api_key = render_sidebar()

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab_ov, tab_pat, tab_acmg, tab_onco, tab_compl, tab_qc = st.tabs(
    ["📊 Vue d'ensemble", "👤 Patient", "🏷️ ACMG", "🧬 OncoPrint",
     "🎯 Complications", "⚖️ Homogénéité"]
)

with tab_ov:
    overview.render(df_f)

with tab_pat:
    patient.render(df_f)

with tab_acmg:
    acmg.render(df_f)

with tab_onco:
    oncoprint.render(df_f, df, pathways_dict, api_key)

with tab_compl:
    complications.render(df_f, df, pathways_dict, api_key)

with tab_qc:
    qc.render(df_f)
