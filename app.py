"""
╔══════════════════════════════════════════════════════════════╗
║                  VARIANT EXPLORER v4.1                       ║
║          Outil interactif d'exploration de variants          ║
║              genomiques pour donnees de sequencage           ║
╚══════════════════════════════════════════════════════════════╝

Point d'entree — orchestration des onglets.
La logique est dans les modules core/, ui/, tabs/.
"""
import streamlit as st

from core.styling import apply_page_config_and_style
from ui.sidebar import render_sidebar
from tabs import (
    overview, patient, acmg, oncoprint, complications, qc,
    clustering, predictors, network, hpo,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
apply_page_config_and_style()

# ─────────────────────────────────────────────
# SIDEBAR (upload, GMT, cle API, filtres globaux)
# ─────────────────────────────────────────────
df, df_f, pathways_dict, api_key = render_sidebar()

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
(tab_ov, tab_pat, tab_acmg, tab_onco, tab_compl, tab_qc,
 tab_clust, tab_pred, tab_net, tab_hpo) = st.tabs([
    "📊 Vue d'ensemble", "👤 Patient", "🏷️ ACMG", "🧬 OncoPrint",
    "🎯 Complications", "⚖️ Homogeneite",
    "🧮 Clustering", "🔮 Predicteurs", "🕸️ Reseau STRING", "🩺 HPO",
])

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

with tab_clust:
    clustering.render(df_f, df, pathways_dict, api_key)

with tab_pred:
    predictors.render(df_f, df, pathways_dict, api_key)

with tab_net:
    network.render(df_f, df, pathways_dict, api_key)

with tab_hpo:
    hpo.render(df_f, df, pathways_dict, api_key)
