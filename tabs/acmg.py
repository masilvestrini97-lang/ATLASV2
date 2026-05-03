"""Onglet : ACMG"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import ACMG_COLORS, ACMG_ORDER


def render(df_f):
    st.markdown("## 🏷️ Classification ACMG")
    st.markdown("> ⚠️ Classification **automatique**. Ne remplace pas une revue manuelle.")
    hm = pd.crosstab(df_f["Pseudo"], df_f["ACMG_class"]).reindex(columns=ACMG_ORDER, fill_value=0)
    fig = px.imshow(hm, color_continuous_scale=["#0a192f", "#64ffda"],
        labels=dict(x="ACMG", y="Patient", color="N"), aspect="auto")
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", height=max(400, len(hm)*18), margin=dict(l=100))
    st.plotly_chart(fig, use_container_width=True)

    dp_all = df_f[df_f["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
    if len(dp_all) > 0:
        st.markdown(f"### 🔴 {len(dp_all)} variants Pathogènes & LP")
        pc = ["Pseudo", "Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
              "ACMG_class", "Clinvar_significance", "CADD_phred", "gnomad_exomes_NFE_AF", "Allelic_ratio", "Depth"]
        st.dataframe(dp_all[[c for c in pc if c in dp_all.columns]].reset_index(drop=True),
                     use_container_width=True, height=500)

    hm_pct = hm.div(hm.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for cls in ACMG_ORDER:
        if cls in hm_pct.columns:
            fig.add_trace(go.Bar(name=cls, y=hm_pct.index, x=hm_pct[cls],
                orientation="h", marker_color=ACMG_COLORS[cls]))
    fig.update_layout(barmode="stack", template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(hm_pct)*22), xaxis_title="% variants", margin=dict(l=100),
        title="Profil ACMG par patient (%)")
    st.plotly_chart(fig, use_container_width=True)
