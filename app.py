"""
╔══════════════════════════════════════════════════════════════╗
║                  VARIANT EXPLORER v3.0                       ║
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
import json

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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
    .cluster-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 12px;
        padding: 20px; margin: 10px 0;
    }
    .cluster-card h4 { color: #64ffda; margin-top: 0; }
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

# Noms lisibles des features pour l'interprétation
FEATURE_LABELS = {
    "pct_Pathogenic": "% Variants Pathogènes",
    "pct_Likely Pathogenic": "% Variants Likely Pathogenic",
    "pct_VUS": "% VUS",
    "pct_Likely Benign": "% Likely Benign",
    "pct_Benign": "% Benign",
    "pct_impact_high": "% Impact HIGH",
    "pct_impact_moderate": "% Impact MODERATE",
    "pct_impact_low": "% Impact LOW",
    "pct_impact_modifier": "% MODIFIER",
    "pct_missensevariant": "% Missense",
    "pct_frameshiftvariant": "% Frameshift",
    "pct_stopgained": "% Stop Gained",
    "pct_spliceregionvariant": "% Splice Region",
    "pct_spliceacceptorvariant": "% Splice Acceptor",
    "pct_splicedonorvariant": "% Splice Donor",
    "pct_disruptiveinframedeletion": "% Délétion in-frame disruptive",
    "pct_startlost": "% Start Lost",
    "impact_score_mean": "Score d'impact moyen",
    "impact_score_max": "Score d'impact maximal",
    "impact_score_p75": "Score d'impact P75",
    "pct_high_impact_score": "% Variants à haut impact (score≥6)",
    "median_CADD": "Score CADD médian",
    "median_AR": "Ratio allélique médian",
    "mean_gnomAD_AF": "Fréquence gnomAD moyenne",
    "n_unique_genes": "Gènes uniques touchés",
    "Histo_HV": "Histologie HV",
    "Histo_mixed": "Histologie mixed",
    "Complication": "Complications",
    "Chirurgie": "Chirurgie",
    "Recidive": "Récidive",
    "BO": "Bronchiolite oblitérante (BO)",
    "PNP": "Polyneuropathie (PNP)",
    "MG": "Myasthénie (MG)",
    "FDSCS": "FDSCS",
}


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


# Variant effects considérés comme non-fonctionnels (exclure du clustering)
NON_FUNCTIONAL_EFFECTS = {
    "synonymousvariant",
    "intronvariant",
    "3primeutrvariant",
    "5primeutrvariant",
    "upstreamgenevariant",
    "downstreamgenevariant",
    "intragenicvariant",
    "5primeutrprematurestartcodongainvariant",
}


def compute_variant_impact_score(row):
    """
    Score d'impact composite par variant (0-10).
    
    Combine 4 dimensions :
    - Impact fonctionnel prédit : HIGH=4, MODERATE=2, LOW=0.5
    - Score CADD normalisé (0-2) : délétèrité in silico
    - Rareté allélique (0-2) : plus rare = plus suspect
    - ClinVar (0-2) : classification clinique connue
    """
    score = 0.0
    
    # 1. Impact fonctionnel (0-4)
    impact = str(row.get("Putative_impact", "")).lower()
    score += {"high": 4.0, "moderate": 2.0, "low": 0.5, "modifier": 0.0}.get(impact, 0)
    
    # 2. CADD normalisé (0-2)
    cadd = row.get("CADD_phred", None)
    if pd.notna(cadd):
        score += min(cadd / 20.0, 2.0)
    
    # 3. Rareté allélique (0-2)
    gnomad = row.get("gnomad_exomes_AF", None)
    if pd.isna(gnomad) or gnomad == 0:
        score += 2.0
    elif gnomad < 0.0001:
        score += 1.8
    elif gnomad < 0.001:
        score += 1.5
    elif gnomad < 0.01:
        score += 1.0
    elif gnomad < 0.05:
        score += 0.3
    
    # 4. ClinVar (0-2)
    clinvar = str(row.get("Clinvar_significance", "")).lower()
    clinvar_w = {
        "pathogenic": 2.0, "pathogeniclikelypathogenic": 2.0,
        "likelypathogenic": 1.5, "uncertainsignificance": 0.5,
        "conflictinginterpretationsofpathogenicity": 0.3, "riskfactor": 0.5,
    }
    score += clinvar_w.get(clinvar, 0)
    
    return round(score, 2)


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
    df["impact_score"] = df.apply(compute_variant_impact_score, axis=1)
    df.columns = df.columns.str.strip()
    return df


def build_patient_features(df, use_genomic=True, use_clinical=True, top_n_genes=30):
    """
    Matrice de features par patient pour clustering.
    
    Stratégie :
    - PROPORTIONS (%) au lieu de comptages bruts (neutralise le biais de quantité)
    - Features BINAIRES pour les gènes (muté oui/non)
    - SCORE D'IMPACT agrégé par patient (mean, max, somme pondérée)
    - Pas de features corrélées à la qualité ADN (n_total, depth)
    """
    patients = df["Pseudo"].unique()
    features = {}

    for pseudo in patients:
        dp = df[df["Pseudo"] == pseudo]
        row = {}
        n_total = len(dp)

        if use_genomic and n_total > 0:
            # ── Proportions ACMG (%) ──
            acmg_vc = dp["ACMG_class"].value_counts()
            for cls in ACMG_ORDER:
                row[f"pct_{cls}"] = (acmg_vc.get(cls, 0) / n_total) * 100

            # ── Proportions par impact (%) ──
            impact_vc = dp["Putative_impact"].value_counts()
            for imp in IMPACT_ORDER:
                row[f"pct_impact_{imp}"] = (impact_vc.get(imp, 0) / n_total) * 100

            # ── Proportions par type de variant fonctionnel (%) ──
            for eff in ["missensevariant", "frameshiftvariant", "stopgained",
                        "spliceregionvariant", "spliceacceptorvariant", "splicedonorvariant",
                        "disruptiveinframedeletion", "startlost"]:
                row[f"pct_{eff}"] = (len(dp[dp["Variant_effect"] == eff]) / n_total) * 100

            # ── Score d'impact agrégé ──
            scores = dp["impact_score"]
            row["impact_score_mean"] = scores.mean()
            row["impact_score_max"] = scores.max()
            row["impact_score_p75"] = scores.quantile(0.75)
            # Nb de variants à haut impact (score >= 6)
            row["pct_high_impact_score"] = (len(scores[scores >= 6]) / n_total) * 100

            # ── Métriques continues ──
            row["median_CADD"] = dp["CADD_phred"].median() if dp["CADD_phred"].notna().any() else 0
            row["median_AR"] = dp["Allelic_ratio"].median()
            row["mean_gnomAD_AF"] = dp["gnomad_exomes_AF"].mean() if dp["gnomad_exomes_AF"].notna().any() else 0
            row["n_unique_genes"] = dp["Gene_symbol"].nunique()

        features[pseudo] = row

    df_feat = pd.DataFrame.from_dict(features, orient="index")

    # ── Gènes : features BINAIRES (muté oui/non) ──
    # On pondère par le max impact_score du gène chez ce patient
    if use_genomic:
        top_genes = df["Gene_symbol"].value_counts().head(top_n_genes).index.tolist()
        for gene in top_genes:
            gene_data = df[df["Gene_symbol"] == gene]
            # Binaire : le patient a-t-il une mutation fonctionnelle dans ce gène ?
            carriers = gene_data["Pseudo"].unique()
            df_feat[f"gene_{gene}"] = df_feat.index.isin(carriers).astype(int)
            
            # Score max du gène chez chaque patient (0 si pas muté)
            gene_max_scores = gene_data.groupby("Pseudo")["impact_score"].max()
            df_feat[f"genescore_{gene}"] = df_feat.index.map(
                lambda p, gms=gene_max_scores: gms.get(p, 0)
            )

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
                        try: df_feat.loc[pseudo, col] = float(val)
                        except (ValueError, TypeError): df_feat.loc[pseudo, col] = 0
                else:
                    if col == "Histo UCD":
                        df_feat.loc[pseudo, "Histo_HV"] = 0
                        df_feat.loc[pseudo, "Histo_mixed"] = 0
                    else:
                        df_feat.loc[pseudo, col] = 0

    return df_feat.fillna(0).astype(float)


def compute_cluster_interpretation(df_feat, labels, key_features, top_n=5):
    """
    Interprétation statistique : pour chaque cluster, identifie les features
    les plus discriminantes par rapport à la moyenne globale (z-score).
    """
    df_fc = df_feat.copy()
    df_fc["Cluster"] = labels

    global_mean = df_feat[key_features].mean()
    global_std = df_feat[key_features].std().replace(0, 1)  # évite division par 0

    interpretations = {}
    for cluster_id in sorted(df_fc["Cluster"].unique()):
        cluster_data = df_fc[df_fc["Cluster"] == cluster_id]
        cluster_mean = cluster_data[key_features].mean()

        # Z-scores par rapport à la moyenne globale
        z_scores = (cluster_mean - global_mean) / global_std

        # Top features enrichies (z > 0) et appauvries (z < 0)
        z_sorted = z_scores.sort_values()
        top_up = z_sorted.tail(top_n).iloc[::-1]  # plus enrichies
        top_down = z_sorted.head(top_n)            # plus appauvries

        # Caractéristiques marquantes (|z| > 0.5)
        significant = z_scores[z_scores.abs() > 0.5].sort_values(ascending=False)

        interpretations[cluster_id] = {
            "n_patients": len(cluster_data),
            "patients": list(cluster_data.index),
            "top_up": top_up,
            "top_down": top_down,
            "significant": significant,
            "cluster_mean": cluster_mean,
            "global_mean": global_mean,
        }

    return interpretations


def get_gene_signature_per_cluster(df, labels_map):
    """Identifie les gènes les plus spécifiques à chaque cluster."""
    signatures = {}
    for cluster_id, patients in labels_map.items():
        df_cl = df[df["Pseudo"].isin(patients)]
        df_rest = df[~df["Pseudo"].isin(patients)]

        # Fréquence du gène dans le cluster vs hors cluster
        gene_freq_cl = df_cl.groupby("Gene_symbol")["Pseudo"].nunique() / len(patients)
        gene_freq_rest = df_rest.groupby("Gene_symbol")["Pseudo"].nunique() / max(1, df_rest["Pseudo"].nunique())

        # Ratio d'enrichissement
        enrichment = (gene_freq_cl / gene_freq_rest.reindex(gene_freq_cl.index).fillna(0.01)).sort_values(ascending=False)
        # Ne garder que les gènes présents chez >30% du cluster
        enrichment = enrichment[gene_freq_cl > 0.3]

        # Top gènes pathogènes dans le cluster
        patho_genes = df_cl[df_cl["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]["Gene_symbol"].value_counts()

        signatures[cluster_id] = {
            "enriched_genes": enrichment.head(10),
            "pathogenic_genes": patho_genes.head(10),
        }
    return signatures


def build_ai_prompt(interpretations, gene_signatures, n_clusters):
    """Construit le prompt pour l'interprétation par IA."""
    prompt = """Tu es un expert en génétique clinique et en bioinformatique spécialisé dans 
l'analyse de variants génomiques somatiques issus de séquençage ciblé (panel de gènes). 
Le contexte est celui de l'étude de tissus FFPE (thymomes / pathologies thymiques liées aux 
maladies auto-immunes : myasthénie, polyneuropathie, bronchiolite oblitérante, etc.).

Un clustering non supervisé a été réalisé sur les profils génomiques et cliniques des patients.

MÉTHODOLOGIE IMPORTANTE :
- Les variants ont été filtrés par qualité avant le clustering (profondeur, ratio allélique, 
  fréquence gnomAD) pour éliminer les artéfacts FFPE et les polymorphismes fréquents.
- Les features génomiques utilisent des PROPORTIONS (%) et non des comptages bruts, 
  afin de ne pas biaiser le clustering par la qualité de l'ADN ou le nombre total de variants.
- Les features de gènes sont binaires (gène muté oui/non dans le patient).

Voici les résultats détaillés pour chaque cluster. Analyse-les et fournis une interprétation 
clinico-génomique structurée.

"""
    for cluster_id, interp in interpretations.items():
        prompt += f"\n{'='*60}\n"
        prompt += f"## {cluster_id} ({interp['n_patients']} patients)\n"
        prompt += f"Patients : {', '.join(interp['patients'])}\n\n"

        prompt += "### Features significativement enrichies (z-score > 0.5) :\n"
        for feat, z in interp['significant'].items():
            if z > 0:
                label = FEATURE_LABELS.get(feat, feat)
                mean_val = interp['cluster_mean'][feat]
                glob_val = interp['global_mean'][feat]
                prompt += f"  - {label}: {mean_val:.2f} (moyenne globale: {glob_val:.2f}, z={z:.2f})\n"

        prompt += "\n### Features significativement réduites :\n"
        for feat, z in interp['significant'].items():
            if z < 0:
                label = FEATURE_LABELS.get(feat, feat)
                mean_val = interp['cluster_mean'][feat]
                glob_val = interp['global_mean'][feat]
                prompt += f"  - {label}: {mean_val:.2f} (moyenne globale: {glob_val:.2f}, z={z:.2f})\n"

        if cluster_id in gene_signatures:
            gs = gene_signatures[cluster_id]
            if len(gs["pathogenic_genes"]) > 0:
                prompt += "\n### Gènes avec variants pathogènes dans ce cluster :\n"
                for gene, count in gs["pathogenic_genes"].head(5).items():
                    prompt += f"  - {gene}: {count} variants pathogènes\n"

            if len(gs["enriched_genes"]) > 0:
                prompt += "\n### Gènes enrichis dans ce cluster (vs autres) :\n"
                for gene, ratio in gs["enriched_genes"].head(5).items():
                    prompt += f"  - {gene}: enrichissement x{ratio:.1f}\n"

    prompt += f"""

{'='*60}
## Instructions pour ton analyse :

Pour chaque cluster, fournis :

1. **Signature dominante** : Quel est le profil génomique et clinique principal de ce groupe ?
2. **Interprétation clinique** : Quelles implications cliniques peut-on déduire ? 
   Lien avec la pathologie thymique, les complications auto-immunes ?
3. **Gènes d'intérêt** : Les gènes enrichis ou porteurs de variants pathogènes ont-ils 
   un rôle connu dans les thymomes ou les maladies auto-immunes ?
4. **Comparaison inter-clusters** : Qu'est-ce qui distingue fondamentalement ces groupes ?

Termine par une **synthèse globale** qui résume les axes de stratification des patients 
et les pistes cliniques/biologiques à explorer.

Réponds en français. Sois précis et cliniquement pertinent.
"""
    return prompt


def call_anthropic_api(prompt, api_key):
    """Appelle l'API Anthropic pour obtenir une interprétation IA."""
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ─────────────────────────────────────────────
# HEADER & CHARGEMENT
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🧬 Variant Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Exploration interactive de variants génomiques — Séquençage ciblé</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📁 Charger fichier variants (.csv)", type=["csv"])

# API Key dans la sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Interprétation IA")
api_key = st.sidebar.text_input(
    "Clé API Anthropic",
    type="password",
    help="Optionnel. Permet une interprétation narrative des clusters par Claude. "
         "Obtenez une clé sur console.anthropic.com",
)

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
          "gnomad_exomes_AF", "gnomad_exomes_NFE_AF", "Allelic_ratio", "Depth", "impact_score"]
    sc = [c for c in sc if c in dv.columns]
    st.dataframe(dv[sc].reset_index(drop=True), use_container_width=True, height=600,
        column_config={
            "CADD_phred": st.column_config.NumberColumn("CADD", format="%.1f"),
            "gnomad_exomes_AF": st.column_config.NumberColumn("gnomAD AF", format="%.5f"),
            "Allelic_ratio": st.column_config.ProgressColumn("AR", min_value=0, max_value=1, format="%.2f"),
            "impact_score": st.column_config.NumberColumn("Impact Score", format="%.1f"),
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
    cr_row = dp[ac_clin].dropna(how="all").head(1)
    if len(cr_row) > 0:
        cols = st.columns(len(ac_clin))
        for i, cn in enumerate(ac_clin):
            v = cr_row[cn].values[0]; cols[i].metric(cn, str(v) if pd.notna(v) else "—")

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
              "Clinvar_significance", "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth", "impact_score"]
        st.dataframe(dp_p[[c for c in pc if c in dp_p.columns]].reset_index(drop=True), use_container_width=True)
    else:
        st.success("Aucun variant pathogène / LP.")

    st.markdown("### Tous les variants")
    ac = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect", "Putative_impact",
          "ACMG_class", "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth", "impact_score"]
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
          "ACMG_class", "Clinvar_significance", "CADD_phred", "gnomad_exomes_AF", "Allelic_ratio", "Depth", "impact_score"]
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


# ═══════════════════════════════════════════════════════
# CLUSTERING + INTERPRÉTATION
# ═══════════════════════════════════════════════════════
with tab_clust:
    st.markdown("## 🔬 Clustering des patients")
    st.markdown(
        "Identification de groupes de patients partageant des **signatures génomiques et/ou cliniques** "
        "similaires via **UMAP** + clustering, avec interprétation statistique et IA."
    )

    if df_f["Pseudo"].nunique() < 5:
        st.warning("Au moins 5 patients nécessaires."); st.stop()

    # ── FILTRES QUALITÉ PRÉ-CLUSTERING ──
    st.markdown("### 🧹 Filtres qualité (pré-clustering)")
    st.markdown(
        "> Ces filtres s'appliquent **uniquement au clustering** pour ne conserver que les variants "
        "fiables et informatifs. En FFPE, la qualité de l'ADN varie entre échantillons : "
        "sans filtrage, le clustering regrouperait les patients par qualité d'ADN plutôt que par biologie."
    )

    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        cl_depth_min = st.number_input("Profondeur min", min_value=0, value=100, step=50,
            key="cl_depth", help="Exclut les variants à faible couverture (artéfacts FFPE)")
    with qc2:
        cl_ar_min = st.number_input("Ratio allélique min", min_value=0.0, value=0.05, step=0.01,
            format="%.2f", key="cl_ar", help="Exclut le bruit de fond (AR très bas)")
    with qc3:
        cl_af_max = st.number_input("gnomAD AF max", min_value=0.0, value=0.01, step=0.005,
            format="%.3f", key="cl_af", help="Exclut les polymorphismes fréquents (non informatifs)")
    with qc4:
        cl_exclude_benign = st.checkbox("Exclure Benign / Likely Benign", value=True,
            help="Focus sur les variants potentiellement pathogènes")

    # Filtrage des effets non fonctionnels
    st.markdown("**Exclusion des variants non fonctionnels**")
    excluded_effects = st.multiselect(
        "Types de variants à exclure",
        sorted(NON_FUNCTIONAL_EFFECTS),
        default=sorted(NON_FUNCTIONAL_EFFECTS),
        help="Les variants synonymes, introniques, UTR, etc. n'ont en général pas d'impact "
             "fonctionnel et ajoutent du bruit au clustering.",
    )

    # Appliquer les filtres qualité
    df_clust = df_f[
        (df_f["Depth"] >= cl_depth_min) &
        (df_f["Allelic_ratio"] >= cl_ar_min) &
        (df_f["gnomad_exomes_AF"].fillna(0) <= cl_af_max) &
        (~df_f["Variant_effect"].isin(excluded_effects))
    ].copy()

    if cl_exclude_benign:
        df_clust = df_clust[~df_clust["ACMG_class"].isin(["Benign", "Likely Benign"])]

    # Métriques post-filtre
    n_before = len(df_f)
    n_after = len(df_clust)
    n_patients_after = df_clust["Pseudo"].nunique()

    qm1, qm2, qm3, qm4 = st.columns(4)
    qm1.metric("Variants avant filtre", f"{n_before:,}")
    qm2.metric("Variants après filtre", f"{n_after:,}")
    qm3.metric("% conservés", f"{n_after/max(n_before,1)*100:.1f}%")
    qm4.metric("Patients conservés", n_patients_after)

    if n_patients_after < 5:
        st.error("Moins de 5 patients après filtrage. Assouplissez les filtres qualité.")
        st.stop()
    if n_after < 50:
        st.warning("Très peu de variants conservés. Les résultats pourraient être instables.")

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres du clustering")
    st.markdown(
        "> Le clustering utilise des **proportions** (%), des **scores d'impact composites** "
        "(combinant CADD, rareté, ClinVar et impact fonctionnel), et des **features binaires** "
        "par gène. Les variants non fonctionnels (synonymes, introniques, UTR) sont exclus."
    )
    cp1, cp2, cp3 = st.columns(3)
    with cp1: use_gen = st.checkbox("Features génomiques", True)
    with cp2: use_clin = st.checkbox("Features cliniques", True)
    with cp3: top_n = st.slider("Top N gènes", 10, 50, 30, 5)

    cc1, cc2 = st.columns(2)
    with cc1: method = st.selectbox("Méthode", ["Hiérarchique (Ward)", "K-Means"])
    with cc2: n_clust = st.slider("Nb clusters", 2, 8, 3)

    cu1, cu2 = st.columns(2)
    with cu1: n_neigh = st.slider("UMAP n_neighbors", 3, 30, 10)
    with cu2: m_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.3, 0.05)

    if not use_gen and not use_clin:
        st.warning("Sélectionnez au moins un type de features."); st.stop()

    with st.spinner("Construction matrice de features (proportions + binaire)..."):
        df_feat = build_patient_features(df_clust, use_gen, use_clin, top_n)

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

    cluster_labels = [f"Cluster {l}" for l in labs]

    df_u = pd.DataFrame({
        "UMAP_1": emb[:, 0], "UMAP_2": emb[:, 1],
        "Cluster": cluster_labels, "Patient": df_feat.index,
    })
    for col in df_feat.columns:
        if col.startswith("pct_") or col.startswith("median_") or col.startswith("mean_") or col.startswith("impact_score") or col in [
            "n_unique_genes", "Histo_HV", "Histo_mixed", "Complication", "Chirurgie",
            "Recidive", "BO", "PNP", "MG", "FDSCS"]:
            df_u[col] = df_feat[col].values

    ccols = px.colors.qualitative.Bold[:n_clust]

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clusters", n_clust)
    m2.metric("Silhouette", f"{sil:.3f}")
    m3.metric("Features", df_feat.shape[1])

    # UMAP
    st.markdown("### Projection UMAP")
    hover_extra = [c for c in ["pct_Pathogenic", "pct_Likely Pathogenic", "pct_VUS",
                               "impact_score_mean", "pct_high_impact_score", "n_unique_genes"]
                   if c in df_u.columns]
    fig = px.scatter(df_u, x="UMAP_1", y="UMAP_2", color="Cluster", text="Patient",
        hover_data=["Patient", "Cluster"] + hover_extra,
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

    # Heatmap profil
    st.markdown("### Profil des clusters")
    df_fc = df_feat.copy()
    df_fc["Cluster"] = cluster_labels

    key_gen = [c for c in df_fc.columns if c in [
        "pct_Pathogenic", "pct_Likely Pathogenic", "pct_VUS", "pct_Likely Benign", "pct_Benign",
        "pct_impact_high", "pct_impact_moderate", "pct_impact_low",
        "pct_missensevariant", "pct_frameshiftvariant", "pct_stopgained",
        "pct_disruptiveinframedeletion", "pct_startlost",
        "impact_score_mean", "impact_score_max", "impact_score_p75",
        "pct_high_impact_score",
        "median_CADD", "median_AR", "mean_gnomAD_AF", "n_unique_genes"]]
    key_clin = [c for c in df_fc.columns if c in [
        "Histo_HV", "Histo_mixed", "Complication", "Chirurgie",
        "Recidive", "BO", "PNP", "MG", "FDSCS"]]
    key_feat = (key_gen if use_gen else []) + (key_clin if use_clin else [])

    if key_feat:
        cp = df_fc.groupby("Cluster")[key_feat].mean().round(2)
        fig = px.imshow(cp.T, color_continuous_scale="YlOrRd", aspect="auto",
            labels=dict(x="Cluster", y="Feature", color="Moyenne"), text_auto=".2f")
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=max(400, len(key_feat)*25), margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    # Barplots
    if use_clin and key_clin:
        st.markdown("### Profil clinique par cluster")
        cm = df_fc.groupby("Cluster")[key_clin].mean()
        fig = px.bar(cm.reset_index().melt(id_vars="Cluster"), x="variable", y="value",
            color="Cluster", barmode="group", color_discrete_sequence=ccols,
            labels={"variable": "Variable clinique", "value": "Proportion moyenne"})
        fig.update_layout(title="Profil clinique moyen", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    if use_gen:
        st.markdown("### Profil ACMG par cluster")
        acmg_c = [f"pct_{c}" for c in ACMG_ORDER if f"pct_{c}" in df_fc.columns]
        if acmg_c:
            am = df_fc.groupby("Cluster")[acmg_c].mean()
            am.columns = [c.replace("pct_", "") for c in am.columns]
            fig = go.Figure()
            for cls in am.columns:
                fig.add_trace(go.Bar(name=cls, x=am.index, y=am[cls],
                    marker_color=ACMG_COLORS.get(cls, "#888")))
            fig.update_layout(barmode="stack", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450,
                title="Répartition ACMG par cluster (%)", yaxis_title="% moyen")
            st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # INTERPRÉTATION STATISTIQUE
    # ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🧠 Interprétation des clusters")

    if key_feat:
        interpretations = compute_cluster_interpretation(df_feat, cluster_labels, key_feat)
        labels_map = {}
        for cl_id, interp in interpretations.items():
            labels_map[cl_id] = interp["patients"]

        gene_signatures = get_gene_signature_per_cluster(df_f, labels_map)

        st.markdown("### 📊 Interprétation statistique")
        st.markdown(
            "Pour chaque cluster, les features sont comparées à la **moyenne globale** via un z-score. "
            "Les features avec |z| > 0.5 sont considérées comme **discriminantes**."
        )

        for cluster_id in sorted(interpretations.keys()):
            interp = interpretations[cluster_id]

            with st.expander(f"🔹 {cluster_id} — {interp['n_patients']} patients ({', '.join(interp['patients'])})", expanded=True):

                # Features discriminantes
                sig = interp["significant"]
                if len(sig) > 0:
                    enriched = sig[sig > 0]
                    depleted = sig[sig < 0]

                    col_e, col_d = st.columns(2)

                    with col_e:
                        st.markdown("**🔺 Enrichi par rapport à la cohorte**")
                        for feat, z in enriched.items():
                            label = FEATURE_LABELS.get(feat, feat)
                            val = interp["cluster_mean"][feat]
                            glob = interp["global_mean"][feat]
                            st.markdown(
                                f'<span class="feat-up">▲ {label}</span> : '
                                f'{val:.2f} vs {glob:.2f} (z={z:+.2f})',
                                unsafe_allow_html=True,
                            )

                    with col_d:
                        st.markdown("**🔻 Réduit par rapport à la cohorte**")
                        for feat, z in depleted.items():
                            label = FEATURE_LABELS.get(feat, feat)
                            val = interp["cluster_mean"][feat]
                            glob = interp["global_mean"][feat]
                            st.markdown(
                                f'<span class="feat-down">▼ {label}</span> : '
                                f'{val:.2f} vs {glob:.2f} (z={z:+.2f})',
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("Aucune feature significativement discriminante pour ce cluster.")

                # Gènes spécifiques
                if cluster_id in gene_signatures:
                    gs = gene_signatures[cluster_id]
                    col_pg, col_eg = st.columns(2)
                    with col_pg:
                        if len(gs["pathogenic_genes"]) > 0:
                            st.markdown("**🧬 Gènes avec variants pathogènes :**")
                            for gene_name, count in gs["pathogenic_genes"].head(5).items():
                                st.markdown(f"- **{gene_name}** : {count} variant(s)")
                    with col_eg:
                        if len(gs["enriched_genes"]) > 0:
                            st.markdown("**📈 Gènes enrichis (vs autres clusters) :**")
                            for gene_name, ratio in gs["enriched_genes"].head(5).items():
                                st.markdown(f"- **{gene_name}** : ×{ratio:.1f}")

        # ─────────────────────────────────────────────────────
        # INTERPRÉTATION IA
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Interprétation par Intelligence Artificielle")

        if not ANTHROPIC_AVAILABLE:
            st.warning(
                "Le package `anthropic` n'est pas installé. "
                "Ajoutez `anthropic` à votre `requirements.txt` pour activer cette fonctionnalité."
            )
        elif not api_key:
            st.info(
                "💡 Pour obtenir une interprétation narrative par IA, entrez votre **clé API Anthropic** "
                "dans la barre latérale. L'IA analysera les profils de chaque cluster et fournira "
                "une interprétation clinico-génomique détaillée.\n\n"
                "👉 Obtenez une clé sur [console.anthropic.com](https://console.anthropic.com)"
            )
        else:
            if st.button("🧠 Lancer l'interprétation IA", type="primary", use_container_width=True):
                with st.spinner("Claude analyse vos clusters... (30-60 secondes)"):
                    try:
                        prompt = build_ai_prompt(interpretations, gene_signatures, n_clust)
                        ai_response = call_anthropic_api(prompt, api_key)

                        st.markdown(
                            f'<div class="ai-interpretation">'
                            f'<h4>🤖 Analyse par Claude</h4>'
                            f'{ai_response}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Stocker en session pour ne pas refaire l'appel
                        st.session_state["ai_interpretation"] = ai_response

                    except anthropic.AuthenticationError:
                        st.error("❌ Clé API invalide. Vérifiez votre clé dans la barre latérale.")
                    except anthropic.RateLimitError:
                        st.error("⏳ Limite de requêtes atteinte. Réessayez dans quelques instants.")
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel API : {str(e)}")

            # Afficher l'interprétation précédente si elle existe
            elif "ai_interpretation" in st.session_state:
                st.markdown(
                    f'<div class="ai-interpretation">'
                    f'<h4>🤖 Analyse par Claude (précédente)</h4>'
                    f'{st.session_state["ai_interpretation"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Export
    st.markdown("### Export")
    exp = df_u[["Patient", "Cluster", "UMAP_1", "UMAP_2"]].to_csv(index=False, sep=";")
    st.download_button("📥 Clusters (CSV)", exp, "clusters.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;color:#4a5568;font-size:0.85rem;">'
    '🧬 Variant Explorer v3.0 — Données locales uniquement</p>', unsafe_allow_html=True)
