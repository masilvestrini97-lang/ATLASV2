"""Onglet : Vue d'ensemble"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import IMPACT_COLORS, IMPACT_ORDER, ACMG_COLORS, ACMG_ORDER


def render(df_f):
    st.markdown("## Vue d'ensemble")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Variants", f"{len(df_f):,}")
    c2.metric("Patients", df_f["Pseudo"].nunique())
    c3.metric("Gènes", df_f["Gene_symbol"].nunique())
    c4.metric("Pathogènes/LP", len(df_f[df_f["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]))
    c5.metric("VUS", len(df_f[df_f["ACMG_class"] == "VUS"]))
    st.markdown("---")

    cl, cr = st.columns(2)
    with cl:
        ac = df_f["ACMG_class"].value_counts().reindex(ACMG_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=ac.index, y=ac.values,
            marker_color=[ACMG_COLORS[c] for c in ac.index],
            text=ac.values.astype(int), textposition="outside"))
        fig.update_layout(title="Distribution ACMG", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, width='stretch')
    with cr:
        ic = df_f["Putative_impact"].value_counts().reindex(IMPACT_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=ic.index, y=ic.values,
            marker_color=[IMPACT_COLORS[c] for c in ic.index],
            text=ic.values.astype(int), textposition="outside"))
        fig.update_layout(title="Distribution Impact", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, width='stretch')

    cl2, cr2 = st.columns(2)
    with cl2:
        tg = df_f["Gene_symbol"].value_counts().head(20)
        fig = go.Figure(go.Bar(y=tg.index[::-1], x=tg.values[::-1], orientation="h",
            marker_color="#64ffda", text=tg.values[::-1], textposition="outside"))
        fig.update_layout(title="Top 20 gènes", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=550, margin=dict(l=120))
        st.plotly_chart(fig, width='stretch')
    with cr2:
        co = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
        cc = df_f["Chromosome"].value_counts().reindex(co).dropna()
        fig = go.Figure(go.Bar(x=cc.index, y=cc.values, marker_color="#48b1bf",
            text=cc.values.astype(int), textposition="outside"))
        fig.update_layout(title="Variants par chromosome", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=550, xaxis_tickangle=-45, margin=dict(b=80))
        st.plotly_chart(fig, width='stretch')

    st.markdown("### CADD vs gnomAD NFE AF")
    sdf = df_f.dropna(subset=["CADD_phred", "gnomad_exomes_NFE_AF"])
    if len(sdf) > 0:
        fig = px.scatter(sdf, x="gnomad_exomes_NFE_AF", y="CADD_phred", color="ACMG_class",
            color_discrete_map=ACMG_COLORS, category_orders={"ACMG_class": ACMG_ORDER},
            hover_data=["Gene_symbol", "hgvs.c", "hgvs.p", "Pseudo"], opacity=0.6, log_x=True)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=500)
        fig.add_hline(y=20, line_dash="dash", line_color="#ffd93d", opacity=0.5, annotation_text="CADD=20")
        fig.add_hline(y=25, line_dash="dash", line_color="#ff6b6b", opacity=0.5, annotation_text="CADD=25")
        fig.add_vline(x=0.01, line_dash="dash", line_color="#888", opacity=0.4, annotation_text="AF=1%")
        st.plotly_chart(fig, width='stretch')
