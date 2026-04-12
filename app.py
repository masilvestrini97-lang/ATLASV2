"""
╔══════════════════════════════════════════════════════════════╗
║                  VARIANT EXPLORER v1.0                       ║
║          Outil interactif d'exploration de variants          ║
║              génomiques pour données de séquençage           ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Variant Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLE CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Police principale */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace; }
    
    /* Métriques stylisées */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
    }
    [data-testid="stMetric"] label { color: #a8b2d1 !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #64ffda !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0a192f; }
    [data-testid="stSidebar"] .stMarkdown { color: #ccd6f6; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    /* Titre principal */
    .main-title {
        background: linear-gradient(90deg, #64ffda, #48b1bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-title { color: #8892b0; font-size: 1.05rem; margin-top: 0; }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-pathogenic { background: #ff6b6b33; color: #ff6b6b; border: 1px solid #ff6b6b55; }
    .badge-likely-pathogenic { background: #ffa50033; color: #ffa500; border: 1px solid #ffa50055; }
    .badge-vus { background: #ffd93d33; color: #ffd93d; border: 1px solid #ffd93d55; }
    .badge-likely-benign { background: #6bcb7733; color: #6bcb77; border: 1px solid #6bcb7755; }
    .badge-benign { background: #4ecdc433; color: #4ecdc4; border: 1px solid #4ecdc455; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────

# Palette de couleurs pour ClinVar
CLINVAR_COLORS = {
    "pathogenic": "#ff6b6b",
    "pathogeniclikelypathogenic": "#ff8e53",
    "likelypathogenic": "#ffa500",
    "uncertainsignificance": "#ffd93d",
    "conflictinginterpretationsofpathogenicity": "#a8a8a8",
    "notprovided": "#888888",
    "likelybenign": "#6bcb77",
    "benignlikelybenign": "#4ecdc4",
    "benign": "#4ecdc4",
    "drugresponse": "#9b59b6",
    "riskfactor": "#e67e22",
    "other": "#cccccc",
}

CLINVAR_LABELS = {
    "pathogenic": "Pathogenic",
    "pathogeniclikelypathogenic": "Pathogenic / Likely Pathogenic",
    "likelypathogenic": "Likely Pathogenic",
    "uncertainsignificance": "VUS",
    "conflictinginterpretationsofpathogenicity": "Conflicting",
    "notprovided": "Not Provided",
    "likelybenign": "Likely Benign",
    "benignlikelybenign": "Benign / Likely Benign",
    "benign": "Benign",
    "drugresponse": "Drug Response",
    "riskfactor": "Risk Factor",
    "other": "Other",
}

IMPACT_COLORS = {
    "high": "#ff6b6b",
    "moderate": "#ffa500",
    "low": "#ffd93d",
    "modifier": "#4ecdc4",
}

IMPACT_ORDER = ["high", "moderate", "low", "modifier"]


def classify_acmg_simple(row):
    """
    Classification ACMG simplifiée basée sur ClinVar, CADD et fréquences alléliques.
    C'est un proxy — la vraie classification ACMG nécessite une revue manuelle.
    """
    clinvar = str(row.get("Clinvar_significance", "")).lower()
    cadd = row.get("CADD_phred", 0)
    gnomad = row.get("gnomad_exomes_AF", None)
    impact = str(row.get("Putative_impact", "")).lower()

    # ClinVar direct
    if "pathogenic" in clinvar and "likely" not in clinvar and "benign" not in clinvar and "conflicting" not in clinvar:
        return "Pathogenic"
    if clinvar == "pathogeniclikelypathogenic":
        return "Pathogenic"
    if clinvar == "likelypathogenic":
        return "Likely Pathogenic"
    if clinvar == "benign" or clinvar == "benignlikelybenign":
        return "Benign"
    if clinvar == "likelybenign":
        return "Likely Benign"

    # Heuristique si ClinVar absent ou VUS
    if impact == "high":
        if pd.notna(gnomad) and gnomad < 0.001:
            return "Likely Pathogenic"
        elif pd.isna(gnomad):
            return "VUS"
        else:
            return "VUS"
    if pd.notna(cadd) and cadd > 25 and impact == "moderate":
        if pd.notna(gnomad) and gnomad < 0.01:
            return "VUS"
    if pd.notna(gnomad) and gnomad > 0.05:
        return "Likely Benign"

    return "VUS"


ACMG_COLORS = {
    "Pathogenic": "#ff6b6b",
    "Likely Pathogenic": "#ffa500",
    "VUS": "#ffd93d",
    "Likely Benign": "#6bcb77",
    "Benign": "#4ecdc4",
}
ACMG_ORDER = ["Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", "Benign"]


def extract_chromosome(variant_str):
    """Extrait le chromosome depuis la colonne Variant (format '1:2488153:A>G')."""
    try:
        chrom = str(variant_str).split(":")[0]
        return f"chr{chrom}"
    except:
        return "Unknown"


def extract_position(variant_str):
    """Extrait la position depuis la colonne Variant."""
    try:
        return int(str(variant_str).split(":")[1])
    except:
        return 0


@st.cache_data
def load_data(uploaded_file):
    """Charge et pré-traite le fichier CSV."""
    df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig", low_memory=False)
    df["Chromosome"] = df["Variant"].apply(extract_chromosome)
    df["Position"] = df["Variant"].apply(extract_position)
    df["ACMG_class"] = df.apply(classify_acmg_simple, axis=1)

    # Nettoyage des noms de colonnes cliniques
    df.columns = df.columns.str.strip()
    return df


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🧬 Variant Explorer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Exploration interactive de variants génomiques — Séquençage ciblé</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader(
    "📁 Charger votre fichier de variants (.csv)",
    type=["csv"],
    help="Fichier CSV séparé par des points-virgules contenant les variants annotés.",
)

if uploaded_file is None:
    st.info(
        "👈 **Chargez votre fichier CSV** via la barre latérale pour commencer l'exploration.\n\n"
        "Le fichier attendu est un CSV séparé par `;` avec les colonnes : "
        "`Sample_id`, `Gene_symbol`, `Variant`, `Clinvar_significance`, `CADD_phred`, etc."
    )
    st.stop()

# Chargement
df = load_data(uploaded_file)

# ─────────────────────────────────────────────
# SIDEBAR — FILTRES GLOBAUX
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Filtres globaux")

# Filtre patients
all_patients = sorted(df["Pseudo"].unique())
selected_patients = st.sidebar.multiselect(
    "Patients", all_patients, default=[], placeholder="Tous les patients"
)

# Filtre gènes
all_genes = sorted(df["Gene_symbol"].unique())
selected_genes = st.sidebar.multiselect(
    "Gènes", all_genes, default=[], placeholder="Tous les gènes"
)

# Filtre impact
selected_impacts = st.sidebar.multiselect(
    "Impact (Putative)",
    IMPACT_ORDER,
    default=[],
    placeholder="Tous les impacts",
)

# Filtre classification ACMG
selected_acmg = st.sidebar.multiselect(
    "Classification ACMG",
    ACMG_ORDER,
    default=[],
    placeholder="Toutes les classes",
)

# Filtre fréquence allélique gnomAD
st.sidebar.markdown("**Fréquence gnomAD exomes**")
af_max = st.sidebar.slider(
    "AF max (gnomAD exomes)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.001,
    format="%.3f",
)

# Filtre CADD
st.sidebar.markdown("**Score CADD**")
cadd_min = st.sidebar.slider(
    "CADD phred min",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=0.5,
)

# Filtre ratio allélique
st.sidebar.markdown("**Ratio allélique**")
ar_range = st.sidebar.slider(
    "Allelic ratio",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01,
)

# Filtre profondeur
depth_min = st.sidebar.number_input("Profondeur min (Depth)", min_value=0, value=0, step=10)


# ─── Application des filtres ───
df_filtered = df.copy()

if selected_patients:
    df_filtered = df_filtered[df_filtered["Pseudo"].isin(selected_patients)]
if selected_genes:
    df_filtered = df_filtered[df_filtered["Gene_symbol"].isin(selected_genes)]
if selected_impacts:
    df_filtered = df_filtered[df_filtered["Putative_impact"].isin(selected_impacts)]
if selected_acmg:
    df_filtered = df_filtered[df_filtered["ACMG_class"].isin(selected_acmg)]

df_filtered = df_filtered[
    (df_filtered["gnomad_exomes_AF"].fillna(0) <= af_max)
    & (df_filtered["CADD_phred"].fillna(0) >= cadd_min)
    & (df_filtered["Allelic_ratio"] >= ar_range[0])
    & (df_filtered["Allelic_ratio"] <= ar_range[1])
    & (df_filtered["Depth"] >= depth_min)
]


# ─────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────
tab_overview, tab_variants, tab_patient, tab_gene, tab_acmg, tab_clinical = st.tabs(
    ["📊 Vue d'ensemble", "🔎 Variants", "👤 Patient", "🧬 Gène", "🏷️ ACMG", "🏥 Clinique"]
)


# ═════════════════════════════════════════════
# TAB 1 : VUE D'ENSEMBLE
# ═════════════════════════════════════════════
with tab_overview:
    st.markdown("## Vue d'ensemble")

    # Métriques
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Variants", f"{len(df_filtered):,}")
    col2.metric("Patients", df_filtered["Pseudo"].nunique())
    col3.metric("Gènes", df_filtered["Gene_symbol"].nunique())
    n_patho = len(df_filtered[df_filtered["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])])
    col4.metric("Pathogènes", n_patho)
    col5.metric("CADD médian", f"{df_filtered['CADD_phred'].median():.1f}" if df_filtered['CADD_phred'].notna().any() else "N/A")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Distribution ACMG
    with col_left:
        acmg_counts = df_filtered["ACMG_class"].value_counts().reindex(ACMG_ORDER).fillna(0)
        fig_acmg = go.Figure(
            data=[
                go.Bar(
                    x=acmg_counts.index,
                    y=acmg_counts.values,
                    marker_color=[ACMG_COLORS.get(c, "#888") for c in acmg_counts.index],
                    text=acmg_counts.values.astype(int),
                    textposition="outside",
                )
            ]
        )
        fig_acmg.update_layout(
            title="Distribution ACMG",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            yaxis_title="Nombre de variants",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_acmg, use_container_width=True)

    # Distribution par impact
    with col_right:
        impact_counts = df_filtered["Putative_impact"].value_counts().reindex(IMPACT_ORDER).fillna(0)
        fig_impact = go.Figure(
            data=[
                go.Bar(
                    x=impact_counts.index,
                    y=impact_counts.values,
                    marker_color=[IMPACT_COLORS.get(c, "#888") for c in impact_counts.index],
                    text=impact_counts.values.astype(int),
                    textposition="outside",
                )
            ]
        )
        fig_impact.update_layout(
            title="Distribution par Impact",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            yaxis_title="Nombre de variants",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_impact, use_container_width=True)

    # Top gènes mutés
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        top_genes = df_filtered["Gene_symbol"].value_counts().head(20)
        fig_genes = go.Figure(
            data=[
                go.Bar(
                    y=top_genes.index[::-1],
                    x=top_genes.values[::-1],
                    orientation="h",
                    marker_color="#64ffda",
                    text=top_genes.values[::-1],
                    textposition="outside",
                )
            ]
        )
        fig_genes.update_layout(
            title="Top 20 gènes les plus mutés",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=550,
            xaxis_title="Nombre de variants",
            margin=dict(l=120, t=50, b=40),
        )
        st.plotly_chart(fig_genes, use_container_width=True)

    # Variants par chromosome
    with col_right2:
        chrom_order = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
        chrom_counts = df_filtered["Chromosome"].value_counts().reindex(chrom_order).dropna()
        fig_chrom = go.Figure(
            data=[
                go.Bar(
                    x=chrom_counts.index,
                    y=chrom_counts.values,
                    marker_color="#48b1bf",
                    text=chrom_counts.values.astype(int),
                    textposition="outside",
                )
            ]
        )
        fig_chrom.update_layout(
            title="Variants par chromosome",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=550,
            yaxis_title="Nombre de variants",
            xaxis_tickangle=-45,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_chrom, use_container_width=True)

    # Scatter : CADD vs gnomAD AF
    st.markdown("### CADD score vs Fréquence gnomAD")
    scatter_df = df_filtered.dropna(subset=["CADD_phred", "gnomad_exomes_AF"])
    if len(scatter_df) > 0:
        fig_scatter = px.scatter(
            scatter_df,
            x="gnomad_exomes_AF",
            y="CADD_phred",
            color="ACMG_class",
            color_discrete_map=ACMG_COLORS,
            category_orders={"ACMG_class": ACMG_ORDER},
            hover_data=["Gene_symbol", "hgvs.c", "hgvs.p", "Pseudo"],
            opacity=0.6,
            log_x=True,
            labels={
                "gnomad_exomes_AF": "gnomAD Exomes AF (log)",
                "CADD_phred": "CADD Phred Score",
            },
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
        )
        # Lignes de référence
        fig_scatter.add_hline(y=20, line_dash="dash", line_color="#ffd93d", opacity=0.5,
                              annotation_text="CADD=20")
        fig_scatter.add_hline(y=25, line_dash="dash", line_color="#ff6b6b", opacity=0.5,
                              annotation_text="CADD=25")
        fig_scatter.add_vline(x=0.01, line_dash="dash", line_color="#888", opacity=0.4,
                              annotation_text="AF=1%")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Pas assez de données avec CADD et gnomAD renseignés pour ce scatter plot.")


# ═════════════════════════════════════════════
# TAB 2 : TABLEAU DES VARIANTS
# ═════════════════════════════════════════════
with tab_variants:
    st.markdown("## 🔎 Explorateur de variants")

    # Recherche libre
    search_term = st.text_input("🔍 Recherche (gène, variant, transcript...)", "")
    if search_term:
        mask = df_filtered.apply(
            lambda row: search_term.lower() in str(row.values).lower(), axis=1
        )
        df_display = df_filtered[mask]
    else:
        df_display = df_filtered

    st.markdown(f"**{len(df_display):,} variants** affichés")

    # Colonnes à afficher
    display_cols = [
        "Pseudo", "Gene_symbol", "Variant", "hgvs.c", "hgvs.p",
        "Variant_effect", "Putative_impact", "ACMG_class",
        "Clinvar_significance", "CADD_phred",
        "gnomad_exomes_AF", "gnomad_exomes_NFE_AF",
        "Allelic_ratio", "Depth", "patho_score",
    ]
    display_cols = [c for c in display_cols if c in df_display.columns]

    st.dataframe(
        df_display[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=600,
        column_config={
            "CADD_phred": st.column_config.NumberColumn("CADD", format="%.1f"),
            "gnomad_exomes_AF": st.column_config.NumberColumn("gnomAD AF", format="%.5f"),
            "gnomad_exomes_NFE_AF": st.column_config.NumberColumn("gnomAD NFE AF", format="%.5f"),
            "Allelic_ratio": st.column_config.ProgressColumn("Allelic Ratio", min_value=0, max_value=1, format="%.2f"),
            "patho_score": st.column_config.NumberColumn("Patho Score", format="%.2f"),
        },
    )

    # Export CSV
    csv_export = df_display[display_cols].to_csv(index=False, sep=";")
    st.download_button(
        "📥 Télécharger les variants filtrés (CSV)",
        csv_export,
        file_name="variants_filtered.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════
# TAB 3 : VUE PAR PATIENT
# ═════════════════════════════════════════════
with tab_patient:
    st.markdown("## 👤 Vue par patient")

    patient_choice = st.selectbox("Sélectionnez un patient", sorted(df_filtered["Pseudo"].unique()))
    df_pat = df_filtered[df_filtered["Pseudo"] == patient_choice]

    # Infos cliniques
    st.markdown("### Informations cliniques")
    clinical_cols = ["Histo UCD", "Complication", "Chirurgie", "Auto Ac", "Recidive", "BO", "PNP", "MG", "FDSCS"]
    clinical_cols = [c for c in clinical_cols if c in df_pat.columns]
    if len(clinical_cols) > 0:
        clinical_row = df_pat[clinical_cols].dropna(how="all").head(1)
        if len(clinical_row) > 0:
            cols = st.columns(len(clinical_cols))
            for i, col_name in enumerate(clinical_cols):
                val = clinical_row[col_name].values[0]
                val_display = str(val) if pd.notna(val) else "—"
                cols[i].metric(col_name, val_display)
        else:
            st.info("Pas de données cliniques disponibles pour ce patient.")

    st.markdown("---")

    # Métriques patient
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Variants", len(df_pat))
    pc2.metric("Gènes", df_pat["Gene_symbol"].nunique())
    n_pat_patho = len(df_pat[df_pat["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])])
    pc3.metric("Pathogènes / LP", n_pat_patho)
    pc4.metric("VUS", len(df_pat[df_pat["ACMG_class"] == "VUS"]))

    # Distribution ACMG du patient
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pat_acmg = df_pat["ACMG_class"].value_counts().reindex(ACMG_ORDER).fillna(0)
        fig_pat_acmg = go.Figure(
            go.Pie(
                labels=pat_acmg.index,
                values=pat_acmg.values,
                marker_colors=[ACMG_COLORS.get(c, "#888") for c in pat_acmg.index],
                hole=0.4,
                textinfo="label+value",
            )
        )
        fig_pat_acmg.update_layout(
            title=f"Classification ACMG — {patient_choice}",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_pat_acmg, use_container_width=True)

    with col_p2:
        pat_effects = df_pat["Variant_effect"].value_counts().head(10)
        fig_pat_eff = go.Figure(
            go.Bar(
                y=pat_effects.index[::-1],
                x=pat_effects.values[::-1],
                orientation="h",
                marker_color="#64ffda",
            )
        )
        fig_pat_eff.update_layout(
            title=f"Types de variants — {patient_choice}",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=200),
        )
        st.plotly_chart(fig_pat_eff, use_container_width=True)

    # Variants pathogènes / LP du patient
    st.markdown("### ⚠️ Variants Pathogènes & Likely Pathogenic")
    df_pat_patho = df_pat[df_pat["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
    if len(df_pat_patho) > 0:
        patho_cols = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
                      "ACMG_class", "Clinvar_significance", "CADD_phred",
                      "gnomad_exomes_AF", "Allelic_ratio", "Depth", "patho_score"]
        patho_cols = [c for c in patho_cols if c in df_pat_patho.columns]
        st.dataframe(df_pat_patho[patho_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.success("Aucun variant pathogène ou likely pathogenic pour ce patient.")

    # Tous les variants du patient
    st.markdown("### Tous les variants")
    all_pat_cols = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
                    "Putative_impact", "ACMG_class", "CADD_phred", "gnomad_exomes_AF",
                    "Allelic_ratio", "Depth"]
    all_pat_cols = [c for c in all_pat_cols if c in df_pat.columns]
    st.dataframe(df_pat[all_pat_cols].reset_index(drop=True), use_container_width=True, height=400)


# ═════════════════════════════════════════════
# TAB 4 : VUE PAR GÈNE
# ═════════════════════════════════════════════
with tab_gene:
    st.markdown("## 🧬 Vue par gène")

    gene_choice = st.selectbox("Sélectionnez un gène", sorted(df_filtered["Gene_symbol"].unique()))
    df_gene = df_filtered[df_filtered["Gene_symbol"] == gene_choice]

    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.metric("Variants", len(df_gene))
    gc2.metric("Patients porteurs", df_gene["Pseudo"].nunique())
    gc3.metric("Variants uniques", df_gene["Variant"].nunique())
    mis_z_val = df_gene["mis_z"].iloc[0] if len(df_gene) > 0 and pd.notna(df_gene["mis_z"].iloc[0]) else None
    gc4.metric("mis_z score", f"{mis_z_val:.2f}" if mis_z_val else "N/A")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Lollipop-like : position vs ratio allélique
        gene_scatter = df_gene.dropna(subset=["Position", "Allelic_ratio"])
        if len(gene_scatter) > 0:
            fig_lollipop = px.scatter(
                gene_scatter,
                x="Position",
                y="Allelic_ratio",
                color="ACMG_class",
                color_discrete_map=ACMG_COLORS,
                size="Depth",
                size_max=15,
                hover_data=["hgvs.c", "hgvs.p", "Pseudo", "Variant_effect"],
                category_orders={"ACMG_class": ACMG_ORDER},
            )
            fig_lollipop.update_layout(
                title=f"Variants sur {gene_choice} (Position vs Allelic Ratio)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Position génomique",
                yaxis_title="Ratio allélique",
            )
            st.plotly_chart(fig_lollipop, use_container_width=True)

    with col_g2:
        # Patients porteurs
        gene_by_patient = df_gene.groupby("Pseudo")["Variant"].count().sort_values(ascending=True)
        fig_gene_pat = go.Figure(
            go.Bar(
                y=gene_by_patient.index,
                x=gene_by_patient.values,
                orientation="h",
                marker_color="#48b1bf",
                text=gene_by_patient.values,
                textposition="outside",
            )
        )
        fig_gene_pat.update_layout(
            title=f"Nombre de variants {gene_choice} par patient",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=100),
        )
        st.plotly_chart(fig_gene_pat, use_container_width=True)

    # Tableau des variants du gène
    st.markdown(f"### Tous les variants de **{gene_choice}**")
    gene_cols = ["Pseudo", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
                 "Putative_impact", "ACMG_class", "Clinvar_significance",
                 "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth", "patho_score"]
    gene_cols = [c for c in gene_cols if c in df_gene.columns]
    st.dataframe(df_gene[gene_cols].reset_index(drop=True), use_container_width=True, height=400)


# ═════════════════════════════════════════════
# TAB 5 : CLASSIFICATION ACMG
# ═════════════════════════════════════════════
with tab_acmg:
    st.markdown("## 🏷️ Classification ACMG")
    st.markdown(
        "> ⚠️ **Note** : La classification présentée ici est une **estimation automatique** "
        "basée sur ClinVar, CADD et les fréquences gnomAD. Elle ne remplace pas une classification "
        "ACMG manuelle par un biologiste ou un généticien."
    )

    # Heatmap ACMG par patient
    st.markdown("### Heatmap ACMG par patient")
    acmg_heatmap = pd.crosstab(df_filtered["Pseudo"], df_filtered["ACMG_class"])
    acmg_heatmap = acmg_heatmap.reindex(columns=ACMG_ORDER, fill_value=0)

    fig_heatmap = px.imshow(
        acmg_heatmap,
        color_continuous_scale=["#0a192f", "#64ffda"],
        labels=dict(x="Classification ACMG", y="Patient", color="Nb variants"),
        aspect="auto",
    )
    fig_heatmap.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(acmg_heatmap) * 18),
        margin=dict(l=100),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Focus variants pathogènes
    st.markdown("### 🔴 Focus : Variants Pathogènes & Likely Pathogenic")
    df_patho_all = df_filtered[df_filtered["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
    if len(df_patho_all) > 0:
        st.markdown(f"**{len(df_patho_all)} variants** identifiés comme Pathogenic ou Likely Pathogenic")
        patho_display_cols = [
            "Pseudo", "Gene_symbol", "Variant", "hgvs.c", "hgvs.p",
            "Variant_effect", "ACMG_class", "Clinvar_significance",
            "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth", "patho_score",
        ]
        patho_display_cols = [c for c in patho_display_cols if c in df_patho_all.columns]
        st.dataframe(df_patho_all[patho_display_cols].sort_values("patho_score", ascending=False).reset_index(drop=True),
                     use_container_width=True, height=500)

        # Récurrence des gènes pathogènes
        st.markdown("### Récurrence des gènes pathogènes")
        patho_gene_counts = df_patho_all["Gene_symbol"].value_counts().head(15)
        fig_patho_genes = go.Figure(
            go.Bar(
                x=patho_gene_counts.index,
                y=patho_gene_counts.values,
                marker_color="#ff6b6b",
                text=patho_gene_counts.values,
                textposition="outside",
            )
        )
        fig_patho_genes.update_layout(
            title="Gènes avec le plus de variants pathogènes",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig_patho_genes, use_container_width=True)
    else:
        st.info("Aucun variant pathogène ou likely pathogenic avec les filtres actuels.")

    # Distribution Patho Score
    st.markdown("### Distribution du Patho Score")
    fig_patho_dist = px.histogram(
        df_filtered,
        x="patho_score",
        color="ACMG_class",
        color_discrete_map=ACMG_COLORS,
        category_orders={"ACMG_class": ACMG_ORDER},
        nbins=50,
        barmode="overlay",
        opacity=0.7,
    )
    fig_patho_dist.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        xaxis_title="Patho Score",
        yaxis_title="Nombre de variants",
    )
    st.plotly_chart(fig_patho_dist, use_container_width=True)


# ═════════════════════════════════════════════
# TAB 6 : DONNÉES CLINIQUES
# ═════════════════════════════════════════════
with tab_clinical:
    st.markdown("## 🏥 Analyse clinique")
    st.markdown("Exploration des variants en lien avec les données cliniques des patients.")

    # Tableau résumé clinique par patient
    clinical_features = ["Histo UCD", "Complication", "Chirurgie", "Auto Ac", "Recidive", "BO", "PNP", "MG", "FDSCS"]
    clinical_features = [c for c in clinical_features if c in df_filtered.columns]

    # Construire un résumé par patient
    patient_summary = []
    for pseudo in df_filtered["Pseudo"].unique():
        df_p = df_filtered[df_filtered["Pseudo"] == pseudo]
        row = {"Patient": pseudo}
        row["N_variants"] = len(df_p)
        row["N_genes"] = df_p["Gene_symbol"].nunique()
        row["N_pathogenic"] = len(df_p[df_p["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])])
        row["N_VUS"] = len(df_p[df_p["ACMG_class"] == "VUS"])
        row["Max_patho_score"] = df_p["patho_score"].max()

        for feat in clinical_features:
            val = df_p[feat].dropna().unique()
            row[feat] = val[0] if len(val) > 0 else None

        patient_summary.append(row)

    df_summary = pd.DataFrame(patient_summary)
    st.markdown("### Résumé par patient")
    st.dataframe(df_summary.sort_values("N_pathogenic", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=500)

    # Comparaison clinique
    st.markdown("### Comparaison clinique")
    if "Histo UCD" in df_filtered.columns:
        histo_vals = df_filtered["Histo UCD"].dropna().unique()
        if len(histo_vals) > 1:
            col_cl1, col_cl2 = st.columns(2)
            with col_cl1:
                histo_acmg = pd.crosstab(
                    df_filtered["Histo UCD"].fillna("Non renseigné"),
                    df_filtered["ACMG_class"],
                )
                histo_acmg = histo_acmg.reindex(columns=ACMG_ORDER, fill_value=0)
                fig_histo = px.bar(
                    histo_acmg.reset_index(),
                    x="Histo UCD",
                    y=ACMG_ORDER,
                    barmode="group",
                    color_discrete_map={c: ACMG_COLORS[c] for c in ACMG_ORDER},
                )
                fig_histo.update_layout(
                    title="ACMG par Histologie",
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400,
                )
                st.plotly_chart(fig_histo, use_container_width=True)

            with col_cl2:
                # Box plot patho_score par histologie
                df_histo = df_filtered.dropna(subset=["Histo UCD"])
                fig_box = px.box(
                    df_histo,
                    x="Histo UCD",
                    y="patho_score",
                    color="Histo UCD",
                    points="outliers",
                )
                fig_box.update_layout(
                    title="Patho Score par Histologie",
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400,
                )
                st.plotly_chart(fig_box, use_container_width=True)

    # Complication vs variants pathogènes
    if "Complication" in df_summary.columns:
        st.markdown("### Complications vs Charge mutationnelle pathogène")
        df_comp = df_summary.dropna(subset=["Complication"])
        if len(df_comp) > 0:
            fig_comp = px.box(
                df_comp,
                x="Complication",
                y="N_pathogenic",
                points="all",
                color="Complication",
            )
            fig_comp.update_layout(
                title="Nombre de variants pathogènes selon les complications",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                xaxis_title="Complication (0=Non, 1=Oui)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#4a5568; font-size:0.85rem;">'
    '🧬 Variant Explorer v1.0 — Données chargées localement, rien n\'est stocké sur le serveur.'
    "</p>",
    unsafe_allow_html=True,
)
