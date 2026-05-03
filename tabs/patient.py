"""Onglet : Patient"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import CLINICAL_COLS, ACMG_COLORS, ACMG_ORDER


def render(df_f):
    st.markdown("## 👤 Vue par patient")
    pat = st.selectbox("Patient", sorted(df_f["Pseudo"].unique()))
    dp = df_f[df_f["Pseudo"] == pat]
    st.markdown("### Données cliniques")
    ac_clin = [c for c in CLINICAL_COLS if c in dp.columns]
    cr_row = dp[ac_clin].dropna(how="all").head(1)
    if len(cr_row) > 0:
        cols = st.columns(len(ac_clin))
        for i, cn in enumerate(ac_clin):
            v = cr_row[cn].values[0]; cols[i].metric(cn, str(v) if pd.notna(v) else "—")
    st.markdown("---")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Variants", len(dp)); p2.metric("Gènes", dp["Gene_symbol"].nunique())
    p3.metric("Pathogènes/LP", len(dp[dp["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]))
    p4.metric("VUS", len(dp[dp["ACMG_class"] == "VUS"]))

    cl, cr = st.columns(2)
    with cl:
        pa = dp["ACMG_class"].value_counts().reindex(ACMG_ORDER).fillna(0)
        fig = go.Figure(go.Pie(labels=pa.index, values=pa.values,
            marker_colors=[ACMG_COLORS[c] for c in pa.index], hole=0.4, textinfo="label+value"))
        fig.update_layout(title=f"ACMG — {pat}", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        pe = dp["Variant_effect"].value_counts().head(10)
        fig = go.Figure(go.Bar(y=pe.index[::-1], x=pe.values[::-1], orientation="h", marker_color="#64ffda"))
        fig.update_layout(title=f"Types — {pat}", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚠️ Pathogènes & LP")
    dp_p = dp[dp["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
    if len(dp_p) > 0:
        pc = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect", "ACMG_class",
              "Clinvar_significance", "CADD_phred", "gnomad_exomes_NFE_AF", "Allelic_ratio", "Depth", "impact_score"]
        st.dataframe(dp_p[[c for c in pc if c in dp_p.columns]].reset_index(drop=True), use_container_width=True)
    else:
        st.success("Aucun variant pathogène / LP.")
    st.markdown("### Tous les variants")
    ac = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect", "Putative_impact",
          "ACMG_class", "CADD_phred", "gnomad_exomes_NFE_AF", "Allelic_ratio", "Depth", "impact_score"]
    st.dataframe(dp[[c for c in ac if c in dp.columns]].reset_index(drop=True),
                 use_container_width=True, height=400)
