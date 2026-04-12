"""
╔══════════════════════════════════════════════════════════════╗
║                  VARIANT EXPLORER v2.0                       ║
║          Outil interactif d'exploration de variants          ║
║              génomiques pour données de séquençage           ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
import umap

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Variant Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
IMPACT_COLORS = {"high": "#ff6b6b", "moderate": "#ffa500", "low": "#ffd93d", "modifier": "#4ecdc4"}
IMPACT_ORDER = ["high", "moderate", "low", "modifier"]
ACMG_COLORS = {
    "Pathogenic": "#ff6b6b", "Likely Pathogenic": "#ffa500", "VUS": "#ffd93d",
    "Likely Benign": "#6bcb77", "Benign": "#4ecdc4",
}
ACMG_ORDER = ["Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", "Benign"]
CLINICAL_COLS = ["Histo UCD", "Complication", "Chirurgie", "Auto Ac", "Recidive", "BO", "PNP", "MG", "FDSCS"]


# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────
def classify_acmg(row):
    clinvar = str(row.get("Clinvar_significance", "")).lower().strip()
    cadd = row.get("CADD_phred", 0)
    gnomad = row.get("gnomad_exomes_AF", None)
    impact = str(row.get("Putative_impact", "")).lower().strip()

    if clinvar in ("pathogenic", "pathogeniclikelypathogenic"):
        return "Pathogenic"
    if clinvar == "likelypathogenic":
        return "Likely Pathogenic"
    if clinvar in ("benign", "benignlikelybenign"):
        return "Benign"
    if clinvar == "likelybenign":
        return "Likely Benign"
    if impact == "high":
        if pd.notna(gnomad) and gnomad < 0.001:
            return "Likely Pathogenic"
        return "VUS"
    if pd.notna(gnomad) and gnomad > 0.05:
        return "Likely Benign"
    if pd.notna(cadd) and cadd > 25 and impact == "moderate":
        if pd.notna(gnomad) and gnomad < 0.01:
            return "VUS"
    return "VUS"


def extract_chrom(v):
    try: return f"chr{str(v).split(':')[0]}"
    except: return "Unknown"


def extract_pos(v):
    try: return int(str(v).split(':')[1])
    except: return 0


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig", low_memory=False)
    df["Chromosome"] = df["Variant"].apply(extract_chrom)
    df["Position"] = df["Variant"].apply(extract_pos)
    df["ACMG_class"] = df.apply(classify_acmg, axis=1)
    df.columns = df.columns.str.strip()
    return df


def build_patient_features(df, use_genomic=True, use_clinical=True, top_n_genes=30):
    """Matrice de features patient × features pour clustering."""
    patients = df["Pseudo"].unique()
    features = {}

    for pseudo in patients:
        dp = df[df["Pseudo"] == pseudo]
        row = {}

        if use_genomic:
            acmg_vc = dp["ACMG_class"].value_counts()
            for cls in ACMG_ORDER:
                row[f"n_{cls}"] = acmg_vc.get(cls, 0)

            impact_vc = dp["Putative_impact"].value_counts()
            for imp in IMPACT_ORDER:
                row[f"n_impact_{imp}"] = impact_vc.get(imp, 0)

            for eff in ["missensevariant", "frameshiftvariant", "stopgained",
                        "spliceregionvariant", "spliceacceptorvariant", "splicedonorvariant"]:
                row[f"n_{eff}"] = len(dp[dp["Variant_effect"] == eff])

            row["median_CADD"] = dp["CADD_phred"].median() if dp["CADD_phred"].notna().any() else 0
            row["median_AR"] = dp["Allelic_ratio"].median()
            row["median_depth"] = dp["Depth"].median()
            row["n_total_variants"] = len(dp)
            row["n_unique_genes"] = dp["Gene_symbol"].nunique()

        features[pseudo] = row

    df_feat = pd.DataFrame.from_dict(features, orient="index")

    if use_genomic:
        top_genes = df["Gene_symbol"].value_counts().head(top_n_genes).index.tolist()
        for gene in top_genes:
            carriers = df[df["Gene_symbol"] == gene]["Pseudo"].unique()
            df_feat[f"gene_{gene}"] = df_feat.index.isin(carriers).astype(int)

    if use_clinical:
        avail = [c for c in CLINICAL_COLS if c in df.columns]
        for pseudo in patients:
            dp = df[df["Pseudo"] == pseudo]
            for col in avail:
                vals = dp[col].dropna().unique()
                if len(vals) > 0:
                    val = vals[0]
                    if col == "Histo UCD":
                        df_feat.loc[pseudo, "Histo_HV"] = 1 if val == "HV" else 0
                        df_feat.loc[pseudo, "Histo_mixed"] = 1 if val == "mixed" else 0
                    else:
                        try:
                            df_feat.loc[pseudo, col] = float(val)
                        except (ValueError, TypeError):
                            df_feat.loc[pseudo, col] = 0
                else:
                    if col == "Histo UCD":
                        df_feat.loc[pseudo, "Histo_HV"] = 0
                        df_feat.loc[pseudo, "Histo_mixed"] = 0
                    else:
                        df_feat.loc[pseudo, col] = 0

    return df_feat.fillna(0).astype(float)


# ─────────────────────────────────────────────
# HEADER & CHARGEMENT
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🧬 Variant Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Exploration interactive de variants génomiques — Séquençage ciblé</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📁 Charger fichier variants (.csv)", type=["csv"])
if uploaded_file is None:
    st.info("👈 **Chargez votre fichier CSV** via la barre latérale pour commencer.")
    st.stop()

df = load_data(uploaded_file)

# ─────────────────────────────────────────────
# FILTRES
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Filtres globaux")

sel_patients = st.sidebar.multiselect("Patients", sorted(df["Pseudo"].unique()), placeholder="Tous")
sel_genes = st.sidebar.multiselect("Gènes", sorted(df["Gene_symbol"].unique()), placeholder="Tous")
sel_impacts = st.sidebar.multiselect("Impact", IMPACT_ORDER, placeholder="Tous")
sel_acmg = st.sidebar.multiselect("ACMG", ACMG_ORDER, placeholder="Toutes")
af_max = st.sidebar.slider("gnomAD AF max", 0.0, 1.0, 1.0, 0.001, format="%.3f")
cadd_min = st.sidebar.slider("CADD min", 0.0, 50.0, 0.0, 0.5)
ar_range = st.sidebar.slider("Allelic ratio", 0.0, 1.0, (0.0, 1.0), 0.01)
depth_min = st.sidebar.number_input("Profondeur min", min_value=0, value=0, step=10)

df_f = df.copy()
if sel_patients: df_f = df_f[df_f["Pseudo"].isin(sel_patients)]
if sel_genes: df_f = df_f[df_f["Gene_symbol"].isin(sel_genes)]
if sel_impacts: df_f = df_f[df_f["Putative_impact"].isin(sel_impacts)]
if sel_acmg: df_f = df_f[df_f["ACMG_class"].isin(sel_acmg)]
df_f = df_f[
    (df_f["gnomad_exomes_AF"].fillna(0) <= af_max) &
    (df_f["CADD_phred"].fillna(0) >= cadd_min) &
    (df_f["Allelic_ratio"].between(ar_range[0], ar_range[1])) &
    (df_f["Depth"] >= depth_min)
]

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab_ov, tab_var, tab_pat, tab_gene, tab_acmg, tab_clust = st.tabs(
    ["📊 Vue d'ensemble", "🔎 Variants", "👤 Patient", "🧬 Gène", "🏷️ ACMG", "🔬 Clustering"]
)

# ═══════ VUE D'ENSEMBLE ═══════
with tab_ov:
    st.markdown("## Vue d'ensemble")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Variants", f"{len(df_f):,}")
    c2.metric("Patients", df_f["Pseudo"].nunique())
    c3.metric("Gènes", df_f["Gene_symbol"].nunique())
    c4.metric("Pathogènes / LP", len(df_f[df_f["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]))
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
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        ic = df_f["Putative_impact"].value_counts().reindex(IMPACT_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=ic.index, y=ic.values,
            marker_color=[IMPACT_COLORS[c] for c in ic.index],
            text=ic.values.astype(int), textposition="outside"))
        fig.update_layout(title="Distribution Impact", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    cl2, cr2 = st.columns(2)
    with cl2:
        tg = df_f["Gene_symbol"].value_counts().head(20)
        fig = go.Figure(go.Bar(y=tg.index[::-1], x=tg.values[::-1], orientation="h",
            marker_color="#64ffda", text=tg.values[::-1], textposition="outside"))
        fig.update_layout(title="Top 20 gènes", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=550, margin=dict(l=120))
        st.plotly_chart(fig, use_container_width=True)

    with cr2:
        co = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
        cc = df_f["Chromosome"].value_counts().reindex(co).dropna()
        fig = go.Figure(go.Bar(x=cc.index, y=cc.values, marker_color="#48b1bf",
            text=cc.values.astype(int), textposition="outside"))
        fig.update_layout(title="Variants par chromosome", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=550, xaxis_tickangle=-45, margin=dict(b=80))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### CADD vs gnomAD AF")
    sdf = df_f.dropna(subset=["CADD_phred", "gnomad_exomes_AF"])
    if len(sdf) > 0:
        fig = px.scatter(sdf, x="gnomad_exomes_AF", y="CADD_phred", color="ACMG_class",
            color_discrete_map=ACMG_COLORS, category_orders={"ACMG_class": ACMG_ORDER},
            hover_data=["Gene_symbol", "hgvs.c", "hgvs.p", "Pseudo"], opacity=0.6, log_x=True)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=500)
        fig.add_hline(y=20, line_dash="dash", line_color="#ffd93d", opacity=0.5, annotation_text="CADD=20")
        fig.add_hline(y=25, line_dash="dash", line_color="#ff6b6b", opacity=0.5, annotation_text="CADD=25")
        fig.add_vline(x=0.01, line_dash="dash", line_color="#888", opacity=0.4, annotation_text="AF=1%")
        st.plotly_chart(fig, use_container_width=True)

# ═══════ VARIANTS ═══════
with tab_var:
    st.markdown("## 🔎 Explorateur de variants")
    search = st.text_input("🔍 Recherche", "")
    dv = df_f[df_f.apply(lambda r: search.lower() in str(r.values).lower(), axis=1)] if search else df_f
    st.markdown(f"**{len(dv):,} variants**")
    sc = ["Pseudo", "Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
          "Putative_impact", "ACMG_class", "Clinvar_significance", "CADD_phred",
          "gnomad_exomes_AF", "gnomad_exomes_NFE_AF", "Allelic_ratio", "Depth"]
    sc = [c for c in sc if c in dv.columns]
    st.dataframe(dv[sc].reset_index(drop=True), use_container_width=True, height=600,
        column_config={
            "CADD_phred": st.column_config.NumberColumn("CADD", format="%.1f"),
            "gnomad_exomes_AF": st.column_config.NumberColumn("gnomAD AF", format="%.5f"),
            "Allelic_ratio": st.column_config.ProgressColumn("AR", min_value=0, max_value=1, format="%.2f"),
        })
    st.download_button("📥 Télécharger (CSV)", dv[sc].to_csv(index=False, sep=";"),
                       "variants_filtered.csv", "text/csv")

# ═══════ PATIENT ═══════
with tab_pat:
    st.markdown("## 👤 Vue par patient")
    pat = st.selectbox("Patient", sorted(df_f["Pseudo"].unique()))
    dp = df_f[df_f["Pseudo"] == pat]

    st.markdown("### Données cliniques")
    ac_clin = [c for c in CLINICAL_COLS if c in dp.columns]
    cr = dp[ac_clin].dropna(how="all").head(1)
    if len(cr) > 0:
        cols = st.columns(len(ac_clin))
        for i, cn in enumerate(ac_clin):
            v = cr[cn].values[0]; cols[i].metric(cn, str(v) if pd.notna(v) else "—")

    st.markdown("---")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Variants", len(dp))
    p2.metric("Gènes", dp["Gene_symbol"].nunique())
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
        fig.update_layout(title=f"Variant types — {pat}", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚠️ Pathogènes & Likely Pathogenic")
    dp_p = dp[dp["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
    if len(dp_p) > 0:
        pc = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect", "ACMG_class",
              "Clinvar_significance", "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth"]
        st.dataframe(dp_p[[c for c in pc if c in dp_p.columns]].reset_index(drop=True), use_container_width=True)
    else:
        st.success("Aucun variant pathogène / LP.")

    st.markdown("### Tous les variants")
    ac = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect", "Putative_impact",
          "ACMG_class", "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth"]
    st.dataframe(dp[[c for c in ac if c in dp.columns]].reset_index(drop=True),
                 use_container_width=True, height=400)

# ═══════ GÈNE ═══════
with tab_gene:
    st.markdown("## 🧬 Vue par gène")
    gene = st.selectbox("Gène", sorted(df_f["Gene_symbol"].unique()))
    dg = df_f[df_f["Gene_symbol"] == gene]

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Variants", len(dg)); g2.metric("Patients", dg["Pseudo"].nunique())
    g3.metric("Uniques", dg["Variant"].nunique())
    mz = dg["mis_z"].iloc[0] if len(dg) > 0 and pd.notna(dg["mis_z"].iloc[0]) else None
    g4.metric("mis_z", f"{mz:.2f}" if mz else "N/A")

    cl, cr = st.columns(2)
    with cl:
        gs = dg.dropna(subset=["Position", "Allelic_ratio"])
        if len(gs) > 0:
            fig = px.scatter(gs, x="Position", y="Allelic_ratio", color="ACMG_class",
                color_discrete_map=ACMG_COLORS, size="Depth", size_max=15,
                hover_data=["hgvs.c", "hgvs.p", "Pseudo", "Variant_effect"],
                category_orders={"ACMG_class": ACMG_ORDER})
            fig.update_layout(title=f"{gene} — Position vs AR", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450)
            st.plotly_chart(fig, use_container_width=True)
    with cr:
        gb = dg.groupby("Pseudo")["Variant"].count().sort_values()
        fig = go.Figure(go.Bar(y=gb.index, x=gb.values, orientation="h",
            marker_color="#48b1bf", text=gb.values, textposition="outside"))
        fig.update_layout(title=f"{gene} — patients", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450, margin=dict(l=100))
        st.plotly_chart(fig, use_container_width=True)

    gc = ["Pseudo", "Variant", "hgvs.c", "hgvs.p", "Variant_effect", "Putative_impact",
          "ACMG_class", "Clinvar_significance", "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth"]
    st.dataframe(dg[[c for c in gc if c in dg.columns]].reset_index(drop=True),
                 use_container_width=True, height=400)

# ═══════ ACMG ═══════
with tab_acmg:
    st.markdown("## 🏷️ Classification ACMG")
    st.markdown("> ⚠️ Classification **automatique** (ClinVar + CADD + gnomAD). Ne remplace pas une revue manuelle.")

    st.markdown("### Heatmap ACMG × Patients")
    hm = pd.crosstab(df_f["Pseudo"], df_f["ACMG_class"]).reindex(columns=ACMG_ORDER, fill_value=0)
    fig = px.imshow(hm, color_continuous_scale=["#0a192f", "#64ffda"],
        labels=dict(x="ACMG", y="Patient", color="N"), aspect="auto")
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", height=max(400, len(hm)*18), margin=dict(l=100))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔴 Pathogènes & Likely Pathogenic")
    dp_all = df_f[df_f["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
    if len(dp_all) > 0:
        st.markdown(f"**{len(dp_all)} variants**")
        pc = ["Pseudo", "Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
              "ACMG_class", "Clinvar_significance", "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth"]
        st.dataframe(dp_all[[c for c in pc if c in dp_all.columns]].reset_index(drop=True),
                     use_container_width=True, height=500)

        pgc = dp_all["Gene_symbol"].value_counts().head(15)
        fig = go.Figure(go.Bar(x=pgc.index, y=pgc.values, marker_color="#ff6b6b",
            text=pgc.values, textposition="outside"))
        fig.update_layout(title="Gènes les plus souvent pathogènes", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Profil ACMG par patient (%)")
    hm_pct = hm.div(hm.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for cls in ACMG_ORDER:
        if cls in hm_pct.columns:
            fig.add_trace(go.Bar(name=cls, y=hm_pct.index, x=hm_pct[cls],
                orientation="h", marker_color=ACMG_COLORS[cls]))
    fig.update_layout(barmode="stack", template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(hm_pct)*22), xaxis_title="% variants", margin=dict(l=100))
    st.plotly_chart(fig, use_container_width=True)

# ═══════ CLUSTERING ═══════
with tab_clust:
    st.markdown("## 🔬 Clustering des patients")
    st.markdown(
        "Identification de groupes de patients partageant des **signatures génomiques et/ou cliniques** "
        "similaires via **UMAP** + clustering."
    )

    if df_f["Pseudo"].nunique() < 5:
        st.warning("Au moins 5 patients nécessaires. Élargissez vos filtres.")
        st.stop()

    st.markdown("### Paramètres")
    cp1, cp2, cp3 = st.columns(3)
    with cp1: use_gen = st.checkbox("Features génomiques", True)
    with cp2: use_clin = st.checkbox("Features cliniques", True)
    with cp3: top_n = st.slider("Top N gènes", 10, 50, 30, 5)

    cc1, cc2 = st.columns(2)
    with cc1: method = st.selectbox("Méthode", ["Hiérarchique (Ward)", "K-Means"])
    with cc2: n_clust = st.slider("Nb clusters", 2, 8, 3)

    cu1, cu2 = st.columns(2)
    with cu1: n_neigh = st.slider("UMAP n_neighbors", 3, 30, 10,
                                   help="Petit = plus de détails locaux")
    with cu2: m_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.3, 0.05,
                                  help="Petit = clusters plus compacts")

    if not use_gen and not use_clin:
        st.warning("Sélectionnez au moins un type de features."); st.stop()

    with st.spinner("Construction matrice de features..."):
        df_feat = build_patient_features(df_f, use_gen, use_clin, top_n)

    st.markdown(f"**{df_feat.shape[0]} patients × {df_feat.shape[1]} features**")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat)

    with st.spinner("UMAP..."):
        nn = min(n_neigh, len(df_feat) - 1)
        emb = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=m_dist,
                         random_state=42).fit_transform(X)

    with st.spinner("Clustering..."):
        if method == "K-Means":
            mdl = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        else:
            mdl = AgglomerativeClustering(n_clusters=n_clust, linkage="ward")
        labs = mdl.fit_predict(X)
        sil = silhouette_score(X, labs) if len(set(labs)) > 1 else 0

    df_u = pd.DataFrame({
        "UMAP_1": emb[:, 0], "UMAP_2": emb[:, 1],
        "Cluster": [f"Cluster {l}" for l in labs],
        "Patient": df_feat.index,
    })
    for col in df_feat.columns:
        if col.startswith("n_") or col.startswith("median_") or col in [
            "Histo_HV", "Histo_mixed", "Complication", "Chirurgie",
            "Recidive", "BO", "PNP", "MG", "FDSCS"]:
            df_u[col] = df_feat[col].values

    ccols = px.colors.qualitative.Bold[:n_clust]

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clusters", n_clust)
    m2.metric("Silhouette", f"{sil:.3f}")
    m3.metric("Features", df_feat.shape[1])

    # UMAP plot
    st.markdown("### Projection UMAP")
    fig = px.scatter(df_u, x="UMAP_1", y="UMAP_2", color="Cluster", text="Patient",
        hover_data=["Patient", "Cluster"] +
            [c for c in ["n_Pathogenic", "n_Likely Pathogenic", "n_VUS", "n_total_variants"] if c in df_u.columns],
        color_discrete_sequence=ccols)
    fig.update_traces(textposition="top center", textfont_size=10, marker_size=12)
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Dendrogramme
    if method == "Hiérarchique (Ward)":
        st.markdown("### Dendrogramme")
        Z = linkage(X, method="ward")
        fig_d = ff.create_dendrogram(X, labels=df_feat.index.tolist(), linkagefun=lambda x: Z,
            color_threshold=Z[-(n_clust-1), 2] if n_clust > 1 else 0)
        fig_d.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=400, xaxis_tickangle=-90, margin=dict(b=120))
        st.plotly_chart(fig_d, use_container_width=True)

    # Profil clusters
    st.markdown("### Profil des clusters")
    df_fc = df_feat.copy()
    df_fc["Cluster"] = [f"Cluster {l}" for l in labs]

    key_gen = [c for c in df_fc.columns if c in [
        "n_Pathogenic", "n_Likely Pathogenic", "n_VUS", "n_Likely Benign", "n_Benign",
        "n_impact_high", "n_impact_moderate", "n_impact_low",
        "n_missensevariant", "n_frameshiftvariant", "n_stopgained",
        "median_CADD", "median_AR", "n_total_variants", "n_unique_genes"]]
    key_clin = [c for c in df_fc.columns if c in [
        "Histo_HV", "Histo_mixed", "Complication", "Chirurgie",
        "Recidive", "BO", "PNP", "MG", "FDSCS"]]
    key_feat = (key_gen if use_gen else []) + (key_clin if use_clin else [])

    if key_feat:
        cp = df_fc.groupby("Cluster")[key_feat].mean().round(2)
        fig = px.imshow(cp.T, color_continuous_scale="YlOrRd", aspect="auto",
            labels=dict(x="Cluster", y="Feature", color="Moyenne"),
            text_auto=".2f")
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=max(400, len(key_feat)*25),
            margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    # Barplot clinique
    if use_clin and key_clin:
        st.markdown("### Profil clinique par cluster")
        cm = df_fc.groupby("Cluster")[key_clin].mean()
        fig = px.bar(cm.reset_index().melt(id_vars="Cluster"), x="variable", y="value",
            color="Cluster", barmode="group", color_discrete_sequence=ccols,
            labels={"variable": "Variable clinique", "value": "Proportion moyenne"})
        fig.update_layout(title="Profil clinique moyen", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Barplot ACMG
    if use_gen:
        st.markdown("### Profil ACMG par cluster")
        acmg_c = [f"n_{c}" for c in ACMG_ORDER if f"n_{c}" in df_fc.columns]
        if acmg_c:
            am = df_fc.groupby("Cluster")[acmg_c].mean()
            am.columns = [c.replace("n_", "") for c in am.columns]
            fig = go.Figure()
            for cls in am.columns:
                fig.add_trace(go.Bar(name=cls, x=am.index, y=am[cls],
                    marker_color=ACMG_COLORS.get(cls, "#888")))
            fig.update_layout(barmode="stack", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450,
                title="ACMG moyen par cluster", yaxis_title="Variants (moy)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Export")
    exp = df_u[["Patient", "Cluster", "UMAP_1", "UMAP_2"]].to_csv(index=False, sep=";")
    st.download_button("📥 Clusters (CSV)", exp, "clusters.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;color:#4a5568;font-size:0.85rem;">'
    '🧬 Variant Explorer v2.0 — Données locales uniquement</p>', unsafe_allow_html=True)
