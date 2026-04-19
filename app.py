"""
╔══════════════════════════════════════════════════════════════╗
║                  VARIANT EXPLORER v4.0                       ║
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
from scipy.stats import fisher_exact
import umap
import json
import io
import tempfile
import os

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
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

NON_FUNCTIONAL_EFFECTS = {
    "synonymousvariant", "intronvariant", "3primeutrvariant", "5primeutrvariant",
    "upstreamgenevariant", "downstreamgenevariant", "intragenicvariant",
    "5primeutrprematurestartcodongainvariant",
}

FEATURE_LABELS = {
    "pct_Pathogenic": "% Pathogènes", "pct_Likely Pathogenic": "% Likely Pathogenic",
    "pct_VUS": "% VUS", "pct_Likely Benign": "% Likely Benign", "pct_Benign": "% Benign",
    "pct_impact_high": "% Impact HIGH", "pct_impact_moderate": "% Impact MODERATE",
    "pct_impact_low": "% Impact LOW", "pct_impact_modifier": "% MODIFIER",
    "pct_missensevariant": "% Missense", "pct_frameshiftvariant": "% Frameshift",
    "pct_stopgained": "% Stop Gained", "pct_spliceregionvariant": "% Splice Region",
    "pct_spliceacceptorvariant": "% Splice Acceptor", "pct_splicedonorvariant": "% Splice Donor",
    "pct_disruptiveinframedeletion": "% Dél. in-frame disruptive", "pct_startlost": "% Start Lost",
    "impact_score_mean": "Score impact moyen", "impact_score_max": "Score impact max",
    "impact_score_p75": "Score impact P75", "pct_high_impact_score": "% Haut impact (≥6)",
    "median_CADD": "CADD médian",
    "mean_gnomAD_AF": "gnomAD NFE moyen", "n_unique_genes": "Gènes uniques",
    "vaf_median": "VAF médiane", "vaf_mean": "VAF moyenne", "vaf_max": "VAF max",
    "vaf_iqr": "VAF IQR (dispersion)", "pct_clonal": "% clonal (VAF≥0.25)",
    "pct_subclonal": "% sous-clonal (0.1-0.25)", "pct_minor": "% mineur (VAF<0.1)",
    "clonal_ratio": "Ratio clonal/mineur", "tmb_score": "Score TMB",
    "Histo_HV": "Histo HV", "Histo_mixed": "Histo mixed",
    "Complication": "Complications", "Chirurgie": "Chirurgie", "Recidive": "Récidive",
    "BO": "BO", "PNP": "PNP", "MG": "MG", "FDSCS": "FDSCS",
}

# ─────────────────────────────────────────────
# FONCTIONS CORE
# ─────────────────────────────────────────────
def classify_acmg(row):
    clinvar = str(row.get("Clinvar_significance", "")).lower().strip()
    gnomad = row.get("gnomad_exomes_NFE_AF", None)
    impact = str(row.get("Putative_impact", "")).lower().strip()
    cadd = row.get("CADD_phred", 0)
    if clinvar in ("pathogenic", "pathogeniclikelypathogenic"): return "Pathogenic"
    if clinvar == "likelypathogenic": return "Likely Pathogenic"
    if clinvar in ("benign", "benignlikelybenign"): return "Benign"
    if clinvar == "likelybenign": return "Likely Benign"
    if impact == "high":
        return "Likely Pathogenic" if pd.notna(gnomad) and gnomad < 0.001 else "VUS"
    if pd.notna(gnomad) and gnomad > 0.05: return "Likely Benign"
    if pd.notna(cadd) and cadd > 25 and impact == "moderate":
        if pd.notna(gnomad) and gnomad < 0.01: return "VUS"
    return "VUS"


def compute_impact_score(row):
    score = {"high": 4.0, "moderate": 2.0, "low": 0.5, "modifier": 0.0}.get(
        str(row.get("Putative_impact", "")).lower(), 0)
    cadd = row.get("CADD_phred", None)
    if pd.notna(cadd): score += min(cadd / 20.0, 2.0)
    gnomad = row.get("gnomad_exomes_NFE_AF", None)
    if pd.isna(gnomad) or gnomad == 0: score += 2.0
    elif gnomad < 0.0001: score += 1.8
    elif gnomad < 0.001: score += 1.5
    elif gnomad < 0.01: score += 1.0
    elif gnomad < 0.05: score += 0.3
    clinvar = str(row.get("Clinvar_significance", "")).lower()
    score += {"pathogenic": 2.0, "pathogeniclikelypathogenic": 2.0,
              "likelypathogenic": 1.5, "uncertainsignificance": 0.5,
              "conflictinginterpretationsofpathogenicity": 0.3, "riskfactor": 0.5}.get(clinvar, 0)
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
    df["impact_score"] = df.apply(compute_impact_score, axis=1)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
@st.cache_data
def load_gmt(gmt_file):
    """Charge un fichier GMT depuis un upload Streamlit et retourne un dict."""
    pathways = {}
    content = gmt_file.read().decode("utf-8") if hasattr(gmt_file, 'read') else open(gmt_file).read()
    for line in content.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            pathways[parts[0]] = set(parts[2:])
    return pathways


@st.cache_data
def load_gmt_from_path(path):
    """Charge un fichier GMT depuis un chemin local (dans le repo)."""
    import os
    if not os.path.exists(path):
        return None
    pathways = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = set(parts[2:])
    return pathways


@st.cache_data
def load_gmt_from_url(url):
    """Télécharge et charge un fichier GMT depuis une URL."""
    import urllib.request
    pathways = {}
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")
        for line in content.strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = set(parts[2:])
        return pathways
    except Exception as e:
        return None


def get_relevant_pathways(pathways, panel_genes, min_genes=2):
    """Filtre les pathways qui contiennent au moins min_genes gènes du panel."""
    return {k: v & panel_genes for k, v in pathways.items() if len(v & panel_genes) >= min_genes}

# ─────────────────────────────────────────────
# CO-OCCURRENCE / CO-EXCLUSION
# ─────────────────────────────────────────────
def compute_cooccurrence_matrix(df, entity_col, patient_col="Pseudo", min_patients=3):
    """
    Matrice de co-occurrence entre entités (gènes ou variants).
    Retourne la matrice binaire patient×entité et les résultats Fisher.
    """
    # Matrice binaire patient × entité
    entities = df.groupby(patient_col)[entity_col].apply(set)
    all_entities = set()
    for s in entities: all_entities.update(s)
    
    # Filtrer les entités présentes chez au moins min_patients
    entity_counts = df.groupby(entity_col)[patient_col].nunique()
    freq_entities = entity_counts[entity_counts >= min_patients].index.tolist()
    
    patients = entities.index.tolist()
    binary = pd.DataFrame(0, index=patients, columns=freq_entities)
    for pat in patients:
        for ent in entities[pat]:
            if ent in freq_entities:
                binary.loc[pat, ent] = 1
    
    return binary


def compute_pairwise_fisher(binary_matrix, max_pairs=50):
    """
    Test de Fisher exact pour chaque paire d'entités.
    Retourne un DataFrame avec odds_ratio, p_value, type (co-occ / co-excl).
    """
    entities = binary_matrix.columns.tolist()
    n = len(entities)
    n_patients = len(binary_matrix)
    
    results = []
    for i in range(min(n, max_pairs)):
        for j in range(i + 1, min(n, max_pairs)):
            a = binary_matrix.iloc[:, i]
            b = binary_matrix.iloc[:, j]
            # Table de contingence 2×2
            both = int((a & b).sum())
            a_only = int((a & ~b).sum())
            b_only = int((~a & b).sum())
            neither = int((~a & ~b).sum())
            
            table = [[both, a_only], [b_only, neither]]
            try:
                odds_ratio, p_value = fisher_exact(table)
            except:
                odds_ratio, p_value = 1.0, 1.0
            
            results.append({
                "Entity_1": entities[i],
                "Entity_2": entities[j],
                "Both": both,
                "Only_1": a_only,
                "Only_2": b_only,
                "Neither": neither,
                "Odds_Ratio": odds_ratio,
                "P_value": p_value,
                "Type": "Co-occurrence" if odds_ratio > 1 else "Co-exclusion",
                "Freq_1": int(a.sum()),
                "Freq_2": int(b.sum()),
            })
    
    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        # Correction Bonferroni
        df_res["P_adjusted"] = df_res["P_value"] * len(df_res)
        df_res["P_adjusted"] = df_res["P_adjusted"].clip(upper=1.0)
        df_res = df_res.sort_values("P_value")
    return df_res


# ─────────────────────────────────────────────
# CLUSTERING FEATURES
# ─────────────────────────────────────────────
def build_patient_features(df, use_genomic=True, use_clinical=True, use_pathways=True,
                           top_n_genes=30, pathways_dict=None):
    """
    Matrice patient × features pour clustering.
    Proportions + scores d'impact + VAF/clonalité + pathways + clinique.
    """
    patients = df["Pseudo"].unique()
    features = {}

    for pseudo in patients:
        dp = df[df["Pseudo"] == pseudo]
        row = {}
        n_total = len(dp)

        if use_genomic and n_total > 0:
            acmg_vc = dp["ACMG_class"].value_counts()
            for cls in ACMG_ORDER:
                row[f"pct_{cls}"] = (acmg_vc.get(cls, 0) / n_total) * 100
            impact_vc = dp["Putative_impact"].value_counts()
            for imp in IMPACT_ORDER:
                row[f"pct_impact_{imp}"] = (impact_vc.get(imp, 0) / n_total) * 100
            for eff in ["missensevariant", "frameshiftvariant", "stopgained",
                        "spliceregionvariant", "spliceacceptorvariant", "splicedonorvariant",
                        "disruptiveinframedeletion", "startlost"]:
                row[f"pct_{eff}"] = (len(dp[dp["Variant_effect"] == eff]) / n_total) * 100
            scores = dp["impact_score"]
            row["impact_score_mean"] = scores.mean()
            row["impact_score_max"] = scores.max()
            row["impact_score_p75"] = scores.quantile(0.75)
            row["pct_high_impact_score"] = (len(scores[scores >= 6]) / n_total) * 100
            row["median_CADD"] = dp["CADD_phred"].median() if dp["CADD_phred"].notna().any() else 0
            row["mean_gnomAD_AF"] = dp["gnomad_exomes_NFE_AF"].mean() if dp["gnomad_exomes_NFE_AF"].notna().any() else 0
            row["n_unique_genes"] = dp["Gene_symbol"].nunique()

            # ── VAF / CLONALITÉ (nouvelles features) ──
            vafs = dp["Allelic_ratio"]
            row["vaf_median"] = vafs.median()
            row["vaf_mean"] = vafs.mean()
            row["vaf_max"] = vafs.max()
            row["vaf_iqr"] = vafs.quantile(0.75) - vafs.quantile(0.25)
            row["pct_clonal"] = (vafs >= 0.25).sum() / n_total * 100
            row["pct_subclonal"] = ((vafs >= 0.1) & (vafs < 0.25)).sum() / n_total * 100
            row["pct_minor"] = (vafs < 0.1).sum() / n_total * 100
            row["clonal_ratio"] = (vafs >= 0.25).sum() / max((vafs < 0.1).sum(), 1)
            # Score TMB composite
            row["tmb_score"] = n_total * vafs.mean()

        features[pseudo] = row

    df_feat = pd.DataFrame.from_dict(features, orient="index")

    # Gènes binaires + score max
    if use_genomic:
        top_genes = df["Gene_symbol"].value_counts().head(top_n_genes).index.tolist()
        for gene in top_genes:
            gene_data = df[df["Gene_symbol"] == gene]
            carriers = gene_data["Pseudo"].unique()
            df_feat[f"gene_{gene}"] = df_feat.index.isin(carriers).astype(int)
            gene_max = gene_data.groupby("Pseudo")["impact_score"].max()
            df_feat[f"genescore_{gene}"] = df_feat.index.map(lambda p, g=gene_max: g.get(p, 0))

    # Pathways
    if use_pathways and pathways_dict:
        panel_genes = set(df["Gene_symbol"].unique())
        relevant = get_relevant_pathways(pathways_dict, panel_genes, min_genes=3)
        # Garder les top pathways les plus pertinents (par nb de gènes du panel)
        top_pw = sorted(relevant.items(), key=lambda x: len(x[1]), reverse=True)[:50]
        
        for pw_name, pw_genes in top_pw:
            pw_key = pw_name[:40]  # Tronquer les noms longs
            for pseudo in patients:
                dp = df[df["Pseudo"] == pseudo]
                patient_genes = set(dp["Gene_symbol"].unique())
                mutated_in_pw = patient_genes & pw_genes
                # Proportion du pathway touchée
                df_feat.loc[pseudo, f"pw_pct_{pw_key}"] = (len(mutated_in_pw) / len(pw_genes)) * 100
                # Score d'impact max dans ce pathway
                pw_variants = dp[dp["Gene_symbol"].isin(pw_genes)]
                df_feat.loc[pseudo, f"pw_score_{pw_key}"] = pw_variants["impact_score"].max() if len(pw_variants) > 0 else 0

    # Clinique
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
                        except: df_feat.loc[pseudo, col] = 0
                else:
                    if col == "Histo UCD":
                        df_feat.loc[pseudo, "Histo_HV"] = 0
                        df_feat.loc[pseudo, "Histo_mixed"] = 0
                    else:
                        df_feat.loc[pseudo, col] = 0

    return df_feat.fillna(0).astype(float)


def compute_cluster_interpretation(df_feat, labels, key_features, top_n=5):
    df_fc = df_feat.copy()
    df_fc["Cluster"] = labels
    global_mean = df_feat[key_features].mean()
    global_std = df_feat[key_features].std().replace(0, 1)
    interpretations = {}
    for cid in sorted(df_fc["Cluster"].unique()):
        cd = df_fc[df_fc["Cluster"] == cid]
        cm = cd[key_features].mean()
        z = (cm - global_mean) / global_std
        sig = z[z.abs() > 0.5].sort_values(ascending=False)
        interpretations[cid] = {
            "n_patients": len(cd), "patients": list(cd.index),
            "significant": sig, "cluster_mean": cm, "global_mean": global_mean,
        }
    return interpretations


def get_gene_signature_per_cluster(df, labels_map):
    signatures = {}
    for cid, patients in labels_map.items():
        df_cl = df[df["Pseudo"].isin(patients)]
        df_rest = df[~df["Pseudo"].isin(patients)]
        gf_cl = df_cl.groupby("Gene_symbol")["Pseudo"].nunique() / max(1, len(patients))
        gf_rest = df_rest.groupby("Gene_symbol")["Pseudo"].nunique() / max(1, df_rest["Pseudo"].nunique())
        enrichment = (gf_cl / gf_rest.reindex(gf_cl.index).fillna(0.01)).sort_values(ascending=False)
        enrichment = enrichment[gf_cl > 0.3]
        patho_genes = df_cl[df_cl["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]["Gene_symbol"].value_counts()
        signatures[cid] = {"enriched_genes": enrichment.head(10), "pathogenic_genes": patho_genes.head(10)}
    return signatures


def build_ai_prompt(interpretations, gene_signatures, n_clusters):
    prompt = """Tu es un expert en génétique clinique spécialisé dans les variants somatiques 
issus de séquençage ciblé FFPE (thymomes / pathologies thymiques / maladies auto-immunes).

Méthodologie : variants filtrés par qualité (profondeur, AR, gnomAD NFE), exclusion des 
synonymes/introniques/UTR. Features en proportions (%) + scores d'impact composites + 
analyse de pathways (MSigDB). Gènes en features binaires.

"""
    for cid, interp in interpretations.items():
        prompt += f"\n{'='*50}\n## {cid} ({interp['n_patients']} patients: {', '.join(interp['patients'])})\n"
        prompt += "### Enrichi:\n"
        for f, z in interp['significant'].items():
            if z > 0:
                label = FEATURE_LABELS.get(f, f)
                prompt += f"  - {label}: {interp['cluster_mean'][f]:.2f} (glob: {interp['global_mean'][f]:.2f}, z={z:.2f})\n"
        prompt += "### Réduit:\n"
        for f, z in interp['significant'].items():
            if z < 0:
                label = FEATURE_LABELS.get(f, f)
                prompt += f"  - {label}: {interp['cluster_mean'][f]:.2f} (glob: {interp['global_mean'][f]:.2f}, z={z:.2f})\n"
        if cid in gene_signatures:
            gs = gene_signatures[cid]
            if len(gs["pathogenic_genes"]) > 0:
                prompt += "### Gènes pathogènes:\n"
                for g, c in gs["pathogenic_genes"].head(5).items(): prompt += f"  - {g}: {c}\n"
            if len(gs["enriched_genes"]) > 0:
                prompt += "### Gènes enrichis:\n"
                for g, r in gs["enriched_genes"].head(5).items(): prompt += f"  - {g}: x{r:.1f}\n"

    prompt += f"""
{'='*50}
Pour chaque cluster: 1) Signature dominante 2) Interprétation clinique (thymome, auto-immunité)
3) Gènes/pathways d'intérêt 4) Comparaison inter-clusters. Termine par une synthèse globale.
Réponds en français, sois précis et cliniquement pertinent."""
    return prompt


def call_anthropic_api(prompt, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=4000,
                                  messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def generate_cluster_report(df_clust, df_feat, cluster_labels, interpretations,
                            gene_sigs, sil_score, method_name, key_feat,
                            excluded_patients=None, pathways_dict=None):
    """Genere un rapport PDF complet sur le clustering."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, PageBreak, Image, HRFlowable)
    import datetime

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MainTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e")))
    styles.add(ParagraphStyle(name='SubTitle2', parent=styles['Normal'],
        fontSize=12, textColor=colors.HexColor("#666666"), spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionTitle', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor("#0f3460"), spaceBefore=20, spaceAfter=10))
    styles.add(ParagraphStyle(name='SubSection', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name='BodyJ', parent=styles['Normal'],
        fontSize=9, leading=13, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name='ClusterName2', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor("#ff6b6b"), spaceBefore=16, spaceAfter=6))
    styles.add(ParagraphStyle(name='FeatureUp', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor("#cc0000")))
    styles.add(ParagraphStyle(name='FeatureDown', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor("#008080")))

    story = []

    def make_table(data, col_widths=None, header_color="#0f3460"):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        return t

    df_fc = df_feat.copy()
    df_fc["Cluster"] = cluster_labels
    n_clusters = len(set(cluster_labels))

    # ========== PAGE 1 : TITRE + RESUME ==========
    story.append(Spacer(1, 30))
    story.append(Paragraph("Variant Explorer - Rapport de Clustering", styles['MainTitle']))
    story.append(Paragraph(f"Date : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['SubTitle2']))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Resume executif", styles['SectionTitle']))

    summary = [
        ["Parametre", "Valeur"],
        ["Patients analyses", str(len(df_feat))],
        ["Patients exclus (FFPE/techniques)", str(len(excluded_patients)) if excluded_patients else "0"],
        ["Variants (apres filtrage)", f"{len(df_clust):,}"],
        ["Genes", str(df_clust["Gene_symbol"].nunique())],
        ["Nombre de clusters", str(n_clusters)],
        ["Methode", method_name],
        ["Score silhouette", f"{sil_score:.3f}"],
        ["Features utilisees", str(len(key_feat))],
    ]
    story.append(make_table(summary, col_widths=[160, 220]))
    story.append(Spacer(1, 10))

    if excluded_patients:
        story.append(Paragraph("Patients exclus du clustering biologique", styles['SubSection']))
        story.append(Paragraph(
            f"Retires avant clustering (VAF mediane trop basse / trop peu de variants) : "
            f"<b>{', '.join(sorted(excluded_patients))}</b>", styles['BodyJ']))
        story.append(Spacer(1, 6))

    sil_txt = ("Bonne separation." if sil_score > 0.5
               else "Separation moderee." if sil_score > 0.3
               else "Faible separation, chevauchement partiel.")
    story.append(Paragraph(f"<b>Silhouette ({sil_score:.3f})</b> : {sil_txt}", styles['BodyJ']))

    # ========== PAGE 2 : PROFILS GLOBAUX ==========
    story.append(PageBreak())
    story.append(Paragraph("Profil global des clusters", styles['SectionTitle']))

    # Tableau comparatif clusters
    comp_cols = [("vaf_median", "VAF med."), ("vaf_mean", "VAF moy."),
                 ("pct_clonal", "% clonal"), ("pct_minor", "% mineur"),
                 ("clonal_ratio", "Ratio cl/min"), ("tmb_score", "Score TMB"),
                 ("impact_score_mean", "Impact moy"), ("n_unique_genes", "N genes"),
                 ("pct_Pathogenic", "% Patho"), ("pct_VUS", "% VUS"),
                 ("pct_impact_high", "% HIGH"), ("median_CADD", "CADD med")]

    avail_comp = [(c, l) for c, l in comp_cols if c in df_fc.columns]
    if avail_comp:
        header = ["Cluster", "N pat."] + [l for _, l in avail_comp]
        comp_data = [header]
        for cl in sorted(df_fc["Cluster"].unique()):
            sub = df_fc[df_fc["Cluster"] == cl]
            row_data = [str(cl), str(len(sub))]
            for c, _ in avail_comp:
                v = sub[c].mean()
                row_data.append(f"{v:.3f}" if v < 1 else f"{v:.1f}")
            comp_data.append(row_data)
        # Ligne globale
        glob_row = ["GLOBAL", str(len(df_fc))]
        for c, _ in avail_comp:
            v = df_fc[c].mean()
            glob_row.append(f"{v:.3f}" if v < 1 else f"{v:.1f}")
        comp_data.append(glob_row)

        story.append(Paragraph("Comparaison inter-clusters (moyennes)", styles['SubSection']))
        cw = [65, 35] + [42] * len(avail_comp)
        t_comp = make_table(comp_data, col_widths=cw)
        story.append(t_comp)
    story.append(Spacer(1, 12))

    # Heatmap features cles
    if key_feat:
        story.append(Paragraph("Heatmap des features cles", styles['SubSection']))
        display_kf = [f for f in key_feat if not f.startswith("pw_")][:20]
        hm_data = [["Feature", "Global"] + sorted(df_fc["Cluster"].unique())]
        for f in display_kf:
            label = FEATURE_LABELS.get(f, f.replace("pct_", "").replace("_", " "))[:25]
            row_hm = [label, f"{df_fc[f].mean():.2f}"]
            for cl in sorted(df_fc["Cluster"].unique()):
                v = df_fc[df_fc["Cluster"] == cl][f].mean()
                row_hm.append(f"{v:.2f}")
            hm_data.append(row_hm)
        cw_hm = [100, 45] + [50] * n_clusters
        story.append(make_table(hm_data, col_widths=cw_hm, header_color="#16213e"))

    # ========== PAGES 3+ : DETAIL PAR CLUSTER ==========
    for cid in sorted(interpretations.keys()):
        interp = interpretations[cid]
        story.append(PageBreak())
        story.append(Paragraph(f"{cid}", styles['ClusterName2']))
        story.append(Paragraph(
            f"<b>{interp['n_patients']} patients</b> : {', '.join(sorted(interp['patients']))}",
            styles['BodyJ']))
        story.append(Spacer(1, 8))

        cluster_data = df_fc[df_fc["Cluster"] == cid]

        # Stats descriptives
        story.append(Paragraph("Statistiques descriptives", styles['SubSection']))
        stat_items = [("vaf_median", "VAF mediane"), ("pct_clonal", "% clonal"),
                      ("pct_minor", "% mineur"), ("tmb_score", "Score TMB"),
                      ("impact_score_mean", "Score impact moy"), ("n_unique_genes", "Genes uniques"),
                      ("pct_Pathogenic", "% Pathogenes"), ("pct_VUS", "% VUS"),
                      ("pct_impact_high", "% Impact HIGH"), ("median_CADD", "CADD median"),
                      ("clonal_ratio", "Ratio clonal/mineur")]
        stat_data = [["Metrique", "Moyenne", "Mediane", "Ecart-type", "Min", "Max"]]
        for col, label in stat_items:
            if col in cluster_data.columns:
                vals = cluster_data[col]
                stat_data.append([label, f"{vals.mean():.2f}", f"{vals.median():.2f}",
                    f"{vals.std():.2f}", f"{vals.min():.2f}", f"{vals.max():.2f}"])
        story.append(make_table(stat_data, col_widths=[90, 55, 55, 55, 55, 55]))
        story.append(Spacer(1, 8))

        # Features discriminantes
        sig = interp["significant"]
        if len(sig) > 0:
            story.append(Paragraph("Features discriminantes (|z| &gt; 0.5)", styles['SubSection']))
            enriched = sig[sig > 0]
            if len(enriched) > 0:
                story.append(Paragraph("<b>Enrichies :</b>", styles['BodyJ']))
                for feat, z in enriched.items():
                    label = FEATURE_LABELS.get(feat, feat.replace("pw_pct_", "PW: ").replace("pct_", "").replace("_", " "))
                    val = interp['cluster_mean'][feat]
                    glob = interp['global_mean'][feat]
                    story.append(Paragraph(
                        f"  &#9650; {label} : {val:.2f} vs {glob:.2f} (z = {z:+.2f})", styles['FeatureUp']))
            depleted = sig[sig < 0]
            if len(depleted) > 0:
                story.append(Spacer(1, 3))
                story.append(Paragraph("<b>Reduites :</b>", styles['BodyJ']))
                for feat, z in depleted.items():
                    label = FEATURE_LABELS.get(feat, feat.replace("pw_pct_", "PW: ").replace("pct_", "").replace("_", " "))
                    val = interp['cluster_mean'][feat]
                    glob = interp['global_mean'][feat]
                    story.append(Paragraph(
                        f"  &#9660; {label} : {val:.2f} vs {glob:.2f} (z = {z:+.2f})", styles['FeatureDown']))
        story.append(Spacer(1, 8))

        # Genes
        if cid in gene_sigs:
            gs = gene_sigs[cid]
            if len(gs["pathogenic_genes"]) > 0:
                story.append(Paragraph("Genes avec variants pathogenes", styles['SubSection']))
                gd = [["Gene", "N variants"]]
                for g, c in gs["pathogenic_genes"].head(10).items():
                    gd.append([str(g), str(c)])
                story.append(make_table(gd, col_widths=[120, 100], header_color="#cc0000"))
                story.append(Spacer(1, 4))
            if len(gs["enriched_genes"]) > 0:
                story.append(Paragraph("Genes enrichis (vs autres clusters)", styles['SubSection']))
                ed = [["Gene", "Enrichissement"]]
                for g, r in gs["enriched_genes"].head(10).items():
                    ed.append([str(g), f"x{r:.1f}"])
                story.append(make_table(ed, col_widths=[120, 100], header_color="#008080"))
        story.append(Spacer(1, 6))

        # Tableau patients
        story.append(Paragraph("Detail des patients", styles['SubSection']))
        pat_data = [["Patient", "VAF med.", "% clonal", "TMB", "Impact", "Genes", "% Patho", "% VUS"]]
        for pat in sorted(interp["patients"]):
            if pat in df_feat.index:
                r = df_feat.loc[pat]
                pat_data.append([pat,
                    f"{r.get('vaf_median', 0):.3f}", f"{r.get('pct_clonal', 0):.1f}%",
                    f"{r.get('tmb_score', 0):.1f}", f"{r.get('impact_score_mean', 0):.2f}",
                    f"{int(r.get('n_unique_genes', 0))}", f"{r.get('pct_Pathogenic', 0):.1f}%",
                    f"{r.get('pct_VUS', 0):.1f}%"])
        story.append(make_table(pat_data, col_widths=[55, 45, 45, 40, 40, 40, 45, 40]))

    # ========== ANNEXE ==========
    story.append(PageBreak())
    story.append(Paragraph("Annexe : Matrice complete des features cles", styles['SectionTitle']))
    display_feats = [c for c in key_feat if not c.startswith("pw_") and not c.startswith("gene")][:15]
    if display_feats:
        header = ["Patient", "Cluster"] + [FEATURE_LABELS.get(f, f)[:16] for f in display_feats]
        annex = [header]
        for pat in sorted(df_feat.index):
            cl = df_fc.loc[pat, "Cluster"]
            row = [pat, str(cl)]
            for f in display_feats:
                v = df_feat.loc[pat].get(f, 0)
                row.append(f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}")
            annex.append(row)
        cw_a = [50, 50] + [35] * len(display_feats)
        story.append(make_table(annex, col_widths=cw_a, header_color="#0f3460"))

    # ========== METHODOLOGIE ==========
    story.append(PageBreak())
    story.append(Paragraph("Methodologie", styles['SectionTitle']))
    story.append(Paragraph(
        "Filtrage qualite : profondeur minimale, ratio allelique minimum, frequence gnomAD NFE maximale. "
        "Exclusion des variants synonymes, introniques et UTR. Features genomiques en proportions (%) "
        "pour neutraliser le biais lie a la qualite FFPE. Features VAF (mediane, % clonal, ratio "
        "clonal/mineur, score TMB) integrees pour capturer la structure clonale. Features de genes "
        "binaires ponderees par score d'impact maximal. Pathways (GMT, MSigDB) representes par le "
        "pourcentage de genes touches et le score d'impact maximal.", styles['BodyJ']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Score d'impact composite (0-10) par variant : impact fonctionnel (HIGH=4, MODERATE=2, "
        "LOW=0.5), CADD normalise (0-2), rarete allelique gnomAD NFE (0-2), ClinVar (0-2).",
        styles['BodyJ']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Clustering : {method_name} apres standardisation z-score. "
        f"Score silhouette : {sil_score:.3f}. UMAP pour visualisation 2D. "
        f"Mode 2 etapes : exclusion prealable des suspects FFPE (VAF mediane basse).",
        styles['BodyJ']))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()




def generate_complications_report(df_c, df_pat_elig, df_pat_clin,
                                   df_res_var, df_res_gene,
                                   n_elig, n_compl, n_no_compl, n_excluded,
                                   compl_counts, avail_compl,
                                   compl_depth, compl_ar, compl_af, compl_excl_ben,
                                   excluded_effects_compl,
                                   n_variants_base, n_variants_filtered, pct_kept,
                                   min_carriers, correction_method,
                                   exclude_no_clinical, excluded_list):
    """Genere un rapport PDF complet de l'analyse des complications."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, PageBreak, Image, HRFlowable)
    import datetime

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    # Helper: add style only if it doesn't exist
    def add_style(name, **kwargs):
        if name not in styles:
            styles.add(ParagraphStyle(name=name, **kwargs))

    add_style('CompMainTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    add_style('CompSubTitle', parent=styles['Normal'],
        fontSize=12, textColor=colors.HexColor("#666666"), spaceAfter=20)
    add_style('CompSection', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor("#0f3460"), spaceBefore=16, spaceAfter=10)
    add_style('CompSub', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=12, spaceAfter=8)
    add_style('CompBody', parent=styles['Normal'],
        fontSize=9, leading=13, alignment=TA_JUSTIFY)
    add_style('CompSmall', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    add_style('CompSuccess', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#006600"),
        backColor=colors.HexColor("#e6f7e6"), borderPadding=6, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=6)
    add_style('CompWarning', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#b07700"),
        backColor=colors.HexColor("#fff5e0"), borderPadding=6, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=6)
    add_style('CompInfo', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#0a4466"),
        backColor=colors.HexColor("#e0f0fa"), borderPadding=6, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=6)

    story = []

    def make_table(data, col_widths=None, header_color="#0f3460", fontsize=8):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), fontsize),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        return t

    def fig_to_image(matplotlib_fig, width_mm=170):
        """Convert a matplotlib Figure to a reportlab Image via in-memory PNG."""
        import matplotlib.pyplot as plt
        img_buf = io.BytesIO()
        matplotlib_fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight",
                               facecolor="white", edgecolor="none")
        plt.close(matplotlib_fig)
        img_buf.seek(0)
        # Calculate height from the figure dims
        w_in, h_in = matplotlib_fig.get_size_inches()
        height_mm = width_mm * (h_in / w_in)
        return Image(img_buf, width=width_mm*mm, height=height_mm*mm)

    # ========== PAGE 1 : TITRE + RESUME ==========
    story.append(Spacer(1, 30))
    story.append(Paragraph("Analyse des complications - Rapport", styles['CompMainTitle']))
    story.append(Paragraph(f"Date : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
                           styles['CompSubTitle']))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Question scientifique", styles['CompSection']))
    story.append(Paragraph(
        "Existe-t-il une signature genomique particuliere (au niveau du variant ou du gene) "
        "associee a l'apparition d'une complication (BO, PNP, MG, FDSCS) chez les patients "
        "de la cohorte ? Analyse supervisee par test exact de Fisher.",
        styles['CompBody']))
    story.append(Spacer(1, 12))

    # Tableau recap
    story.append(Paragraph("Resume de l'analyse", styles['CompSub']))
    recap = [
        ["Parametre", "Valeur"],
        ["Patients eligibles", str(n_elig)],
        ["Avec complication", str(n_compl)],
        ["Sans complication", str(n_no_compl)],
        ["Exclus (aucune info clinique)", str(n_excluded)],
        ["Variants avant filtrage", f"{n_variants_base:,}"],
        ["Variants apres filtrage", f"{n_variants_filtered:,}"],
        ["Pourcentage conserve", f"{pct_kept:.1f}%"],
        ["Seuil patients porteurs min", str(min_carriers)],
        ["Correction multi-tests", correction_method],
    ]
    story.append(make_table(recap, col_widths=[180, 180]))
    story.append(Spacer(1, 12))

    # Filtres detailles
    story.append(Paragraph("Filtres qualite appliques", styles['CompSub']))
    filters_tbl = [
        ["Filtre", "Valeur"],
        ["Profondeur minimale (Depth)", f"≥ {compl_depth}"],
        ["Ratio allelique minimum (AR)", f"≥ {compl_ar:.2f}"],
        ["gnomAD NFE frequence max", f"≤ {compl_af:.3f}"],
        ["Benign / Likely Benign", "Exclus" if compl_excl_ben else "Inclus"],
        ["Types exclus", ", ".join(sorted(excluded_effects_compl)) if excluded_effects_compl else "Aucun"],
        ["Patients sans info clinique", "Exclus" if exclude_no_clinical else "Inclus"],
    ]
    story.append(make_table(filters_tbl, col_widths=[200, 260]))

    if exclude_no_clinical and n_excluded > 0 and excluded_list:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"<b>Patients exclus ({len(excluded_list)})</b> : {', '.join(sorted(excluded_list))}",
            styles['CompSmall']))

    # ========== PAGE 2 : REPARTITION DES COMPLICATIONS ==========
    story.append(PageBreak())
    story.append(Paragraph("Repartition des complications", styles['CompSection']))

    # Tableau comptage
    compl_tbl = [["Type de complication", "Nb patients", "% de la cohorte"]]
    for col in avail_compl:
        n = compl_counts.get(col, 0)
        compl_tbl.append([col, str(n), f"{n/max(n_elig,1)*100:.1f}%"])
    story.append(make_table(compl_tbl, col_widths=[150, 100, 150]))
    story.append(Spacer(1, 10))

    # Figure : bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig_mpl, ax = plt.subplots(figsize=(8, 3.5))
        bar_colors = ["#ff6b6b", "#ffa500", "#ffd93d", "#9b59b6"][:len(compl_counts)]
        bars = ax.bar(list(compl_counts.keys()), list(compl_counts.values()),
                      color=bar_colors, edgecolor="#333", linewidth=0.5)
        for bar, val in zip(bars, compl_counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   str(val), ha="center", fontsize=10)
        ax.set_ylabel("Patients", fontsize=10)
        ax.set_title("Nombre de patients par type de complication", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
        fig_mpl.tight_layout()
        story.append(fig_to_image(fig_mpl, width_mm=170))
    except Exception as e:
        story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))
    story.append(Spacer(1, 8))

    # Figure : pie chart compl vs non-compl
    try:
        fig_mpl, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            [n_compl, n_no_compl],
            labels=["Complique", "Non complique"],
            colors=["#ff6b6b", "#4ecdc4"],
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100 * (n_compl+n_no_compl)))})",
            startangle=90, wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10))
        ax.set_title("Repartition compliques vs non compliques", fontsize=11)
        fig_mpl.tight_layout()
        story.append(fig_to_image(fig_mpl, width_mm=120))
    except Exception as e:
        story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))

    # ========== PAGE 3+ : NIVEAU VARIANT ==========
    story.append(PageBreak())
    story.append(Paragraph("Analyse par variant", styles['CompSection']))

    if df_res_var is None or len(df_res_var) == 0:
        story.append(Paragraph(
            "Aucun variant n'est present chez suffisamment de patients pour etre teste. "
            "Abaissez le seuil de porteurs minimum ou elargissez la cohorte.",
            styles['CompWarning']))
    else:
        sig_var = df_res_var[df_res_var["P_adjusted"] < 0.05]
        nominal_sig_var = df_res_var[df_res_var["P_value"] < 0.05]

        # Conclusion statistique
        story.append(Paragraph("Conclusion statistique", styles['CompSub']))
        if len(sig_var) > 0:
            top_with_gene = sig_var.head(5).apply(
                lambda r: f"{r.get('Gene', 'N/A')} ({r['Entity']})" if pd.notna(r.get('Gene')) else str(r['Entity']),
                axis=1
            ).tolist()
            story.append(Paragraph(
                f"<b>{len(sig_var)} variant(s) significativement associe(s)</b> a la complication "
                f"apres correction {correction_method} (p ajustee < 0.05). "
                f"Top : {', '.join(top_with_gene)}. Validation sur cohorte independante necessaire.",
                styles['CompSuccess']))
        elif len(nominal_sig_var) > 0:
            story.append(Paragraph(
                f"<b>Aucun variant significatif apres correction {correction_method}</b> "
                f"(p ajustee &lt; 0.05). {len(nominal_sig_var)} variant(s) ont une p brute &lt; 0.05, "
                f"suggerant des signaux exploratoires non concluants. "
                f"Attendu avec une petite cohorte et beaucoup de tests multiples.",
                styles['CompWarning']))
        else:
            story.append(Paragraph(
                "<b>Aucun variant n'est associe</b> a la complication. "
                "Resultat attendu : il est rare que plusieurs patients partagent exactement "
                "le meme variant. L'analyse par gene est plus appropriee.",
                styles['CompInfo']))

        # Volcano plot variants
        story.append(Spacer(1, 6))
        story.append(Paragraph("Volcano plot", styles['CompSub']))
        df_res_var_plot = df_res_var.copy()
        df_res_var_plot["log2_OR"] = np.log2(df_res_var_plot["Odds_Ratio"].clip(lower=0.01, upper=100))
        df_res_var_plot["log10_p"] = -np.log10(df_res_var_plot["P_adjusted"].clip(lower=1e-10))
        df_res_var_plot["Significant"] = df_res_var_plot["P_adjusted"] < 0.05

        try:
            fig_mpl, ax = plt.subplots(figsize=(9, 5))
            # Non-significatifs en gris
            ns = df_res_var_plot[~df_res_var_plot["Significant"]]
            s = df_res_var_plot[df_res_var_plot["Significant"]]
            ax.scatter(ns["log2_OR"], ns["log10_p"], c="#555555", s=25, alpha=0.6,
                       edgecolor="none", label="Non significatif")
            if len(s) > 0:
                ax.scatter(s["log2_OR"], s["log10_p"], c="#ff6b6b", s=40, alpha=0.8,
                           edgecolor="#cc0000", linewidth=0.5, label="Significatif")
            ax.axvline(0, linestyle="--", color="#888", linewidth=0.8)
            ax.axhline(-np.log10(0.05), linestyle="--", color="#ffa500", linewidth=0.8)
            ax.text(ax.get_xlim()[1] * 0.95, -np.log10(0.05) + 0.05, "p ajustee = 0.05",
                    fontsize=8, color="#ffa500", ha="right")
            ax.set_xlabel("log2(Odds Ratio)", fontsize=10)
            ax.set_ylabel("-log10(P ajustee)", fontsize=10)
            ax.set_title("Volcano plot : variants", fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=9, loc="upper right")
            ax.tick_params(labelsize=9)
            fig_mpl.tight_layout()
            story.append(fig_to_image(fig_mpl, width_mm=170))
        except Exception as e:
            story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))
        story.append(Spacer(1, 8))

        # Tableau top 20 variants
        story.append(PageBreak())
        story.append(Paragraph("Top 20 variants associes", styles['CompSub']))
        gene_col = "Gene" if "Gene" in df_res_var.columns else ("Gène" if "Gène" in df_res_var.columns else None)
        hgvs_col = "hgvs.p" if "hgvs.p" in df_res_var.columns else None
        eff_col = "Variant_effect" if "Variant_effect" in df_res_var.columns else None

        headers = ["Gene", "Variant", "HGVS p.", "Effet", "N port.", "Comp+", "Comp-", "OR", "P brute", "P adj."]
        var_tbl = [headers]
        for _, r in df_res_var.head(20).iterrows():
            var_tbl.append([
                str(r.get(gene_col, "-"))[:10] if gene_col else "-",
                str(r["Entity"])[:18],
                str(r.get(hgvs_col, "-") if hgvs_col else "-")[:14],
                str(r.get(eff_col, "-") if eff_col else "-")[:12],
                str(int(r["N_carriers"])),
                str(int(r["Carriers_with_compl"])),
                str(int(r["Carriers_without_compl"])),
                f"{r['Odds_Ratio']:.2f}",
                f"{r['P_value']:.4f}",
                f"{r['P_adjusted']:.4f}",
            ])
        col_widths_var = [50, 85, 65, 55, 32, 32, 32, 32, 40, 40]
        story.append(make_table(var_tbl, col_widths=col_widths_var, fontsize=7))

    # ========== NIVEAU GENE ==========
    story.append(PageBreak())
    story.append(Paragraph("Analyse par gene (Pathogenic / LP / VUS)", styles['CompSection']))
    story.append(Paragraph(
        "Test sur les genes ou au moins une mutation Pathogenic, Likely Pathogenic ou VUS "
        "est presente chez au moins N patients. Les variants Benign/LB sont exclus.",
        styles['CompBody']))
    story.append(Spacer(1, 6))

    if df_res_gene is None or len(df_res_gene) == 0:
        story.append(Paragraph(
            "Aucun gene n'est mute chez suffisamment de patients pour etre teste.",
            styles['CompWarning']))
    else:
        sig_gene = df_res_gene[df_res_gene["P_adjusted"] < 0.05]
        nominal_sig_gene = df_res_gene[df_res_gene["P_value"] < 0.05]

        story.append(Paragraph("Conclusion statistique", styles['CompSub']))
        if len(sig_gene) > 0:
            top_genes = sig_gene.head(5)["Entity"].tolist()
            story.append(Paragraph(
                f"<b>{len(sig_gene)} gene(s) significativement associe(s)</b> a la complication "
                f"apres correction {correction_method} (p ajustee &lt; 0.05). "
                f"Top genes : {', '.join(top_genes)}. "
                f"Ces genes constituent des pistes biologiques interessantes.",
                styles['CompSuccess']))
        elif len(nominal_sig_gene) > 0:
            top_nominal = nominal_sig_gene.head(5)["Entity"].tolist()
            story.append(Paragraph(
                f"<b>Aucun gene significatif apres correction {correction_method}</b> "
                f"(p ajustee &lt; 0.05). {len(nominal_sig_gene)} gene(s) ont une p brute &lt; 0.05 : "
                f"{', '.join(top_nominal)}. Signaux exploratoires a valider. "
                f"Essayez la correction FDR (moins stricte).",
                styles['CompWarning']))
        else:
            story.append(Paragraph(
                "<b>Aucun gene n'est associe</b> a la complication, meme avant correction. "
                "Soit le signal n'existe pas a l'echelle individuelle du gene, soit la cohorte "
                "est trop petite. L'analyse par pathway pourrait reveler des associations.",
                styles['CompInfo']))

        # Volcano plot genes
        story.append(Spacer(1, 6))
        story.append(Paragraph("Volcano plot", styles['CompSub']))
        df_res_gene_plot = df_res_gene.copy()
        df_res_gene_plot["log2_OR"] = np.log2(df_res_gene_plot["Odds_Ratio"].clip(lower=0.01, upper=100))
        df_res_gene_plot["log10_p"] = -np.log10(df_res_gene_plot["P_adjusted"].clip(lower=1e-10))
        df_res_gene_plot["Significant"] = df_res_gene_plot["P_adjusted"] < 0.05

        try:
            fig_mpl, ax = plt.subplots(figsize=(9, 5.5))
            ns = df_res_gene_plot[~df_res_gene_plot["Significant"]]
            s = df_res_gene_plot[df_res_gene_plot["Significant"]]
            ax.scatter(ns["log2_OR"], ns["log10_p"], c="#555555", s=25, alpha=0.6,
                       edgecolor="none", label="Non significatif")
            if len(s) > 0:
                ax.scatter(s["log2_OR"], s["log10_p"], c="#ff6b6b", s=45, alpha=0.85,
                           edgecolor="#cc0000", linewidth=0.5, label="Significatif")
            # Annoter les top hits (p brute < 0.1)
            to_label = df_res_gene_plot[df_res_gene_plot["P_value"] < 0.1].head(10)
            for _, r in to_label.iterrows():
                ax.annotate(str(r["Entity"]), xy=(r["log2_OR"], r["log10_p"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.axvline(0, linestyle="--", color="#888", linewidth=0.8)
            ax.axhline(-np.log10(0.05), linestyle="--", color="#ffa500", linewidth=0.8)
            ax.text(ax.get_xlim()[1] * 0.95, -np.log10(0.05) + 0.05, "p ajustee = 0.05",
                    fontsize=8, color="#ffa500", ha="right")
            ax.set_xlabel("log2(Odds Ratio)", fontsize=10)
            ax.set_ylabel("-log10(P ajustee)", fontsize=10)
            ax.set_title("Volcano plot : genes", fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=9, loc="upper right")
            ax.tick_params(labelsize=9)
            fig_mpl.tight_layout()
            story.append(fig_to_image(fig_mpl, width_mm=170))
        except Exception as e:
            story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))

        # Tableau top 20 genes
        story.append(PageBreak())
        story.append(Paragraph("Top 20 genes associes", styles['CompSub']))
        gene_headers = ["Gene", "N port.", "Comp+", "Comp-",
                        "Freq compl porteurs", "Freq compl non-port.",
                        "OR", "P brute", "P ajust."]
        gene_tbl = [gene_headers]
        for _, r in df_res_gene.head(20).iterrows():
            gene_tbl.append([
                str(r["Entity"])[:15],
                str(int(r["N_carriers"])),
                str(int(r["Carriers_with_compl"])),
                str(int(r["Carriers_without_compl"])),
                f"{r['Freq_compl_carriers']:.2f}",
                f"{r['Freq_compl_non_carriers']:.2f}",
                f"{r['Odds_Ratio']:.2f}",
                f"{r['P_value']:.4f}",
                f"{r['P_adjusted']:.4f}",
            ])
        col_widths_gene = [70, 40, 40, 40, 70, 70, 40, 50, 50]
        story.append(make_table(gene_tbl, col_widths=col_widths_gene, fontsize=7))

    # ========== METHODOLOGIE ==========
    story.append(PageBreak())
    story.append(Paragraph("Methodologie et interpretation", styles['CompSection']))

    story.append(Paragraph("Principe de l'analyse", styles['CompSub']))
    story.append(Paragraph(
        "Pour chaque variant (ou gene) present chez au moins N patients, un test exact de Fisher "
        "est applique sur une table de contingence 2x2 croisant la presence du variant (porteur "
        "oui/non) et le statut complication (Complication_any = 1 si au moins une des colonnes BO, "
        "PNP, MG ou FDSCS est a 1). L'odds ratio (OR) mesure la force de l'association : OR &gt; 1 "
        "= enrichissement chez les compliques, OR &lt; 1 = depletion.",
        styles['CompBody']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Correction multi-tests", styles['CompSub']))
    if correction_method == "Bonferroni":
        corr_txt = ("Correction de Bonferroni : p ajustee = p brute * N tests. Methode stricte "
                    "qui minimise les faux positifs mais reduit la puissance. Adaptee quand "
                    "on cherche des signaux robustes.")
    elif correction_method == "FDR (Benjamini-Hochberg)":
        corr_txt = ("Correction FDR (Benjamini-Hochberg) : controle le taux de faux positifs "
                    "attendu. Plus permissive que Bonferroni, adaptee a l'exploration quand on "
                    "genere des hypotheses sur de nombreux tests.")
    else:
        corr_txt = ("Aucune correction appliquee. Les p-values sont brutes. A utiliser uniquement "
                    "pour explorer des tendances ou sur un nombre tres limite de tests.")
    story.append(Paragraph(corr_txt, styles['CompBody']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Limitations et puissance statistique", styles['CompSub']))
    power_note = ""
    if n_elig < 30:
        power_note = ("<b>Cohorte tres petite ({} patients)</b> : la puissance statistique est "
                     "fortement limitee. Les tests multiples apres correction peuvent ne rien "
                     "retenir meme si des signaux biologiques existent.").format(n_elig)
    elif n_elig < 50:
        power_note = ("<b>Cohorte modeste ({} patients)</b> : puissance limitee pour les "
                     "correction strictes. Envisagez une validation independante.").format(n_elig)
    else:
        power_note = ("Cohorte de taille acceptable ({} patients).").format(n_elig)

    if n_compl < 5 or n_no_compl < 5:
        power_note += (" <b>ATTENTION</b> : un des groupes contient moins de 5 patients "
                      "(compliques={}, non-compliques={}), la puissance est critique.").format(n_compl, n_no_compl)

    story.append(Paragraph(power_note, styles['CompBody']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Niveaux d'analyse", styles['CompSub']))
    story.append(Paragraph(
        "<b>Niveau variant</b> : identifie les variants exacts associes. Peu de signal attendu "
        "car il est rare que plusieurs patients partagent exactement le meme variant dans un "
        "contexte somatique FFPE.",
        styles['CompBody']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Niveau gene</b> : identifie les genes ou une mutation delertere (Patho/LP/VUS) est "
        "enrichie chez les compliques. Plus sensible car il agrege les variants par gene. Les "
        "variants Benign/Likely Benign sont exclus car ils diluent le signal.",
        styles['CompBody']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Niveau pathway</b> (a venir) : identifie les voies biologiques enrichies. Plus "
        "sensible encore car agrege plusieurs genes d'une meme voie.",
        styles['CompBody']))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Rapport genere par Variant Explorer v4.0 - {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles['CompSmall']))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────
# HEADER & CHARGEMENT
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🧬 Variant Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Exploration interactive de variants génomiques — Séquençage ciblé</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📁 Fichier variants (.csv)", type=["csv"])

# ── CHARGEMENT DES PATHWAYS (multi-sources) ──
st.sidebar.markdown("### 🧬 Pathways (optionnel)")

# 1. Essayer de trouver un fichier GMT local dans le repo (détection flexible)
import os
import glob

def find_local_gmt():
    """Cherche un fichier GMT dans le repo en étant flexible sur le nom/extension."""
    # Chemins prioritaires (noms canoniques)
    priority_paths = ["pathways.gmt", "data/pathways.gmt", "gmt/pathways.gmt"]
    for p in priority_paths:
        if os.path.exists(p):
            return p

    # Fallback : n'importe quel fichier .gmt ou contenant "gmt" dans le nom à la racine ou sous-dossier
    patterns = ["*.gmt", "*gmt*.txt", "data/*.gmt", "gmt/*.gmt", "data/*gmt*.txt"]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Retourner le plus petit (souvent le plus pertinent) ou le premier
            return matches[0]
    return None

local_gmt_path = find_local_gmt()

# 2. Interface
pathways_dict = None
gmt_source = "Aucun"

if local_gmt_path:
    # Fichier trouvé automatiquement dans le repo
    use_local_gmt = st.sidebar.checkbox(
        f"✅ Utiliser `{local_gmt_path}` (repo)", value=True,
        help="Fichier GMT détecté automatiquement dans le repo GitHub."
    )
    if use_local_gmt:
        pathways_dict = load_gmt_from_path(local_gmt_path)
        gmt_source = f"Local: {local_gmt_path}"

# Option alternative : uploader un fichier
gmt_file = st.sidebar.file_uploader(
    "📁 Ou uploader un GMT (.gmt)", type=["gmt"],
    help="Alternative : uploader manuellement un fichier GMT (MSigDB)."
)
if gmt_file:
    pathways_dict = load_gmt(gmt_file)
    gmt_source = f"Upload: {gmt_file.name}"

# Option alternative : URL
with st.sidebar.expander("🌐 Ou depuis une URL"):
    gmt_url = st.text_input("URL du fichier GMT", value="",
        placeholder="https://...")
    if st.button("Télécharger", key="dl_gmt") and gmt_url:
        with st.spinner("Téléchargement..."):
            pw = load_gmt_from_url(gmt_url)
            if pw:
                pathways_dict = pw
                gmt_source = f"URL ({len(pw)} pathways)"
                st.success(f"✅ {len(pw)} pathways chargés")
            else:
                st.error("Échec du téléchargement. Vérifiez l'URL.")

if pathways_dict:
    st.sidebar.markdown(f"📊 **{len(pathways_dict)} pathways** chargés")
    st.sidebar.caption(f"Source : {gmt_source}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 IA")
api_key = st.sidebar.text_input("Clé API Anthropic", type="password",
    help="Optionnel. Pour l'interprétation IA des clusters.")

if uploaded_file is None:
    st.info("👈 **Chargez votre fichier CSV** via la barre latérale."); st.stop()

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
af_max = st.sidebar.slider("gnomAD NFE AF max", 0.0, 1.0, 1.0, 0.001, format="%.3f")
cadd_min = st.sidebar.slider("CADD min", 0.0, 50.0, 0.0, 0.5)
ar_range = st.sidebar.slider("Allelic ratio", 0.0, 1.0, (0.0, 1.0), 0.01)
depth_min = st.sidebar.number_input("Profondeur min", min_value=0, value=0, step=10)

df_f = df.copy()
if sel_patients: df_f = df_f[df_f["Pseudo"].isin(sel_patients)]
if sel_genes: df_f = df_f[df_f["Gene_symbol"].isin(sel_genes)]
if sel_impacts: df_f = df_f[df_f["Putative_impact"].isin(sel_impacts)]
if sel_acmg: df_f = df_f[df_f["ACMG_class"].isin(sel_acmg)]
df_f = df_f[
    (df_f["gnomad_exomes_NFE_AF"].fillna(0) <= af_max) &
    (df_f["CADD_phred"].fillna(0) >= cadd_min) &
    (df_f["Allelic_ratio"].between(ar_range[0], ar_range[1])) &
    (df_f["Depth"] >= depth_min)
]

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab_ov, tab_var, tab_pat, tab_gene, tab_acmg, tab_vaf, tab_coocc, tab_compl, tab_qc, tab_clust = st.tabs(
    ["📊 Vue d'ensemble", "🔎 Variants", "👤 Patient", "🧬 Gène", "🏷️ ACMG",
     "📈 VAF & Clonalité", "🔗 Co-occurrence", "🎯 Complications",
     "⚖️ Homogénéité", "🔬 Clustering"]
)

# ═══════ VUE D'ENSEMBLE ═══════
with tab_ov:
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
        st.plotly_chart(fig, use_container_width=True)

# ═══════ VARIANTS ═══════
with tab_var:
    st.markdown("## 🔎 Explorateur de variants")
    search = st.text_input("🔍 Recherche", "")
    dv = df_f[df_f.apply(lambda r: search.lower() in str(r.values).lower(), axis=1)] if search else df_f
    st.markdown(f"**{len(dv):,} variants**")
    sc = ["Pseudo", "Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
          "Putative_impact", "ACMG_class", "Clinvar_significance", "CADD_phred",
          "gnomad_exomes_NFE_AF", "Allelic_ratio", "Depth", "impact_score"]
    sc = [c for c in sc if c in dv.columns]
    st.dataframe(dv[sc].reset_index(drop=True), use_container_width=True, height=600,
        column_config={
            "CADD_phred": st.column_config.NumberColumn("CADD", format="%.1f"),
            "gnomad_exomes_NFE_AF": st.column_config.NumberColumn("gnomAD NFE", format="%.5f"),
            "Allelic_ratio": st.column_config.ProgressColumn("AR", min_value=0, max_value=1, format="%.2f"),
            "impact_score": st.column_config.NumberColumn("Impact", format="%.1f"),
        })
    st.download_button("📥 CSV", dv[sc].to_csv(index=False, sep=";"), "variants_filtered.csv", "text/csv")

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
          "ACMG_class", "Clinvar_significance", "CADD_phred", "gnomad_exomes_NFE_AF", "Allelic_ratio", "Depth", "impact_score"]
    st.dataframe(dg[[c for c in gc if c in dg.columns]].reset_index(drop=True),
                 use_container_width=True, height=400)

# ═══════ ACMG ═══════
with tab_acmg:
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

# ═══════════════════════════════════════════════════════
# VAF & CLONALITÉ
# ═══════════════════════════════════════════════════════
with tab_vaf:
    st.markdown("## 📈 VAF & Charge Tumorale")
    st.markdown(
        "Analyse de la **fréquence allélique des variants (VAF)** par patient. "
        "La distribution des VAFs reflète la structure clonale de la tumeur : "
        "les VAFs élevées correspondent à des mutations **clonales** (événements précoces), "
        "les VAFs basses à des clones mineurs (hétérogénéité intra-tumorale)."
    )
    st.markdown(
        "> ⚠️ En FFPE sans contrôle de pureté tumorale, les VAFs absolues sont modulées "
        "par le ratio tumeur/normal. La **distribution relative** reste informative."
    )

    # Filtres qualité pour VAF
    st.markdown("### Filtres")
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        vaf_depth = st.number_input("Profondeur min", value=100, step=50, key="vaf_d")
    with vc2:
        vaf_af_max = st.number_input("gnomAD NFE max", value=0.01, step=0.005, format="%.3f", key="vaf_af")
    with vc3:
        vaf_excl_nf = st.checkbox("Exclure non fonctionnels", value=True, key="vaf_nf")

    df_vaf = df_f[(df_f["Depth"] >= vaf_depth) &
                  (df_f["gnomad_exomes_NFE_AF"].fillna(0) <= vaf_af_max)].copy()
    if vaf_excl_nf:
        df_vaf = df_vaf[~df_vaf["Variant_effect"].isin(NON_FUNCTIONAL_EFFECTS)]

    # Exclure benign pour focus tumoral
    df_vaf = df_vaf[~df_vaf["ACMG_class"].isin(["Benign", "Likely Benign"])]

    st.markdown(f"**{len(df_vaf):,} variants** après filtrage")

    if len(df_vaf) > 0:
        # ── RÉSUMÉ TMB PAR PATIENT ──
        st.markdown("### Charge mutationnelle par patient")

        tmb_summary = []
        for pseudo in sorted(df_vaf["Pseudo"].unique()):
            dp = df_vaf[df_vaf["Pseudo"] == pseudo]
            vafs = dp["Allelic_ratio"]
            tmb_summary.append({
                "Patient": pseudo,
                "N_variants": len(dp),
                "N_gènes": dp["Gene_symbol"].nunique(),
                "VAF_médiane": round(vafs.median(), 3),
                "VAF_max": round(vafs.max(), 3),
                "N_clonal (VAF≥0.25)": int((vafs >= 0.25).sum()),
                "N_sous-clonal (0.1-0.25)": int(((vafs >= 0.1) & (vafs < 0.25)).sum()),
                "N_mineur (VAF<0.1)": int((vafs < 0.1).sum()),
                "% clonal": round((vafs >= 0.25).sum() / max(len(vafs), 1) * 100, 1),
                "Score TMB": round(len(dp) * vafs.mean(), 2),  # nb variants × VAF moyenne
            })
        df_tmb = pd.DataFrame(tmb_summary).sort_values("Score TMB", ascending=False)

        # Bar chart TMB
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_tmb["Patient"], y=df_tmb["N_clonal (VAF≥0.25)"],
            name="Clonal (VAF ≥ 0.25)", marker_color="#ff6b6b"))
        fig.add_trace(go.Bar(
            x=df_tmb["Patient"], y=df_tmb["N_sous-clonal (0.1-0.25)"],
            name="Sous-clonal (0.1–0.25)", marker_color="#ffa500"))
        fig.add_trace(go.Bar(
            x=df_tmb["Patient"], y=df_tmb["N_mineur (VAF<0.1)"],
            name="Mineur (VAF < 0.1)", marker_color="#4ecdc4"))
        fig.update_layout(barmode="stack", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=450, xaxis_tickangle=-45, margin=dict(b=100),
            title="Structure clonale par patient",
            yaxis_title="Nombre de variants", legend_title="Clonalité")
        st.plotly_chart(fig, use_container_width=True)

        # Tableau TMB
        st.dataframe(df_tmb.reset_index(drop=True), use_container_width=True, height=400,
            column_config={
                "VAF_médiane": st.column_config.NumberColumn("VAF méd.", format="%.3f"),
                "VAF_max": st.column_config.NumberColumn("VAF max", format="%.3f"),
                "Score TMB": st.column_config.NumberColumn("Score TMB", format="%.1f"),
                "% clonal": st.column_config.ProgressColumn("% clonal", min_value=0, max_value=100, format="%.1f%%"),
            })

        st.markdown("---")

        # ── VUE PAR PATIENT : SPECTRE VAF ──
        st.markdown("### Spectre VAF par patient")
        vaf_patient = st.selectbox("Patient", df_tmb["Patient"].tolist(), key="vaf_pat")
        dp_vaf = df_vaf[df_vaf["Pseudo"] == vaf_patient]

        vl, vr = st.columns(2)

        with vl:
            # Histogramme VAF
            fig = px.histogram(dp_vaf, x="Allelic_ratio", nbins=30,
                color="ACMG_class", color_discrete_map=ACMG_COLORS,
                category_orders={"ACMG_class": ACMG_ORDER},
                barmode="overlay", opacity=0.7)
            fig.add_vline(x=0.25, line_dash="dash", line_color="#ff6b6b", opacity=0.7,
                          annotation_text="Clonal (0.25)")
            fig.add_vline(x=0.1, line_dash="dash", line_color="#ffa500", opacity=0.5,
                          annotation_text="Sous-clonal (0.1)")
            fig.update_layout(title=f"Distribution VAF — {vaf_patient}",
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=400,
                xaxis_title="VAF (Allelic Ratio)", yaxis_title="Variants")
            st.plotly_chart(fig, use_container_width=True)

        with vr:
            # VAF vs Impact Score
            fig = px.scatter(dp_vaf, x="Allelic_ratio", y="impact_score",
                color="Putative_impact", color_discrete_map=IMPACT_COLORS,
                size="Depth", size_max=15,
                hover_data=["Gene_symbol", "hgvs.c", "hgvs.p", "ACMG_class"],
                category_orders={"Putative_impact": IMPACT_ORDER})
            fig.add_vline(x=0.25, line_dash="dash", line_color="#ff6b6b", opacity=0.5)
            fig.update_layout(title=f"VAF vs Impact Score — {vaf_patient}",
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=400,
                xaxis_title="VAF", yaxis_title="Impact Score")
            st.plotly_chart(fig, use_container_width=True)

        # Lollipop : gène × VAF
        st.markdown("### Gènes par VAF")
        gene_vaf = dp_vaf.groupby("Gene_symbol").agg(
            max_VAF=("Allelic_ratio", "max"),
            mean_VAF=("Allelic_ratio", "mean"),
            n_variants=("Variant", "count"),
            max_impact=("impact_score", "max"),
            best_acmg=("ACMG_class", "first"),
        ).sort_values("max_VAF", ascending=False).head(30)

        fig = go.Figure()
        # Lignes verticales (lollipop sticks)
        for i, (gene, row) in enumerate(gene_vaf.iterrows()):
            color = ACMG_COLORS.get(row["best_acmg"], "#888")
            fig.add_trace(go.Scatter(
                x=[row["max_VAF"]], y=[gene],
                mode="markers", marker=dict(size=row["max_impact"] * 2.5 + 4, color=color),
                showlegend=False,
                hovertemplate=f"<b>{gene}</b><br>Max VAF: {row['max_VAF']:.3f}<br>"
                    f"Impact: {row['max_impact']:.1f}<br>{row['n_variants']} variant(s)<extra></extra>",
            ))

        fig.add_vline(x=0.25, line_dash="dash", line_color="#ff6b6b", opacity=0.5,
                      annotation_text="Clonal")
        fig.add_vline(x=0.1, line_dash="dash", line_color="#ffa500", opacity=0.4,
                      annotation_text="Sous-clonal")
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=max(400, len(gene_vaf) * 22),
            xaxis_title="VAF max", yaxis_title="",
            title=f"Gènes mutés — {vaf_patient} (taille = impact score, couleur = ACMG)",
            margin=dict(l=120))
        st.plotly_chart(fig, use_container_width=True)

        # Tableau des variants clonaux
        st.markdown("### 🔴 Variants clonaux (VAF ≥ 0.25)")
        clonal = dp_vaf[dp_vaf["Allelic_ratio"] >= 0.25].sort_values("Allelic_ratio", ascending=False)
        if len(clonal) > 0:
            cl_cols = ["Gene_symbol", "Variant", "hgvs.c", "hgvs.p", "Variant_effect",
                       "ACMG_class", "Allelic_ratio", "Depth", "CADD_phred", "impact_score"]
            st.dataframe(clonal[[c for c in cl_cols if c in clonal.columns]].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "Allelic_ratio": st.column_config.ProgressColumn("VAF", min_value=0, max_value=1, format="%.3f"),
                    "impact_score": st.column_config.NumberColumn("Impact", format="%.1f"),
                })
        else:
            st.info("Aucun variant clonal (VAF ≥ 0.25) pour ce patient.")

        st.markdown("---")

        # ── COMPARAISON MULTI-PATIENTS ──
        st.markdown("### Comparaison VAF entre patients")

        # Boxplot VAF par patient
        fig = px.box(df_vaf, x="Pseudo", y="Allelic_ratio", color="Pseudo",
            points="outliers", notched=True)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=450,
            title="Distribution VAF par patient",
            xaxis_tickangle=-45, margin=dict(b=100),
            xaxis_title="", yaxis_title="VAF",
            showlegend=False)
        fig.add_hline(y=0.25, line_dash="dash", line_color="#ff6b6b", opacity=0.5)
        fig.add_hline(y=0.1, line_dash="dash", line_color="#ffa500", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

        # Scatter TMB score vs % clonal
        st.markdown("### Score TMB vs Proportion clonale")
        fig = px.scatter(df_tmb, x="Score TMB", y="% clonal",
            text="Patient", size="N_variants", size_max=20,
            color="VAF_médiane", color_continuous_scale=["#4ecdc4", "#ffa500", "#ff6b6b"],
            hover_data=["N_gènes", "N_clonal (VAF≥0.25)", "VAF_max"])
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=500,
            xaxis_title="Score TMB (N variants × VAF moyenne)",
            yaxis_title="% de variants clonaux (VAF ≥ 0.25)")
        st.plotly_chart(fig, use_container_width=True)

        # Export
        csv_tmb = df_tmb.to_csv(index=False, sep=";")
        st.download_button("📥 Exporter TMB (CSV)", csv_tmb, "tmb_summary.csv", "text/csv")
    else:
        st.warning("Pas assez de variants après filtrage.")

# ═══════════════════════════════════════════════════════
# CO-OCCURRENCE / CO-EXCLUSION
# ═══════════════════════════════════════════════════════
with tab_coocc:
    st.markdown("## 🔗 Co-occurrence & Co-exclusion")
    st.markdown(
        "Analyse des associations entre mutations : quels **gènes** ou **variants** "
        "tendent à être mutés ensemble (co-occurrence) ou à s'exclure mutuellement ?"
    )

    coocc_level = st.radio("Niveau d'analyse", ["Par gène", "Par variant"],
        horizontal=True, help="Gène : un patient porte-t-il une mutation dans le gène ? "
                              "Variant : un patient porte-t-il exactement ce variant ?")

    st.markdown("### Filtres")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        coocc_depth = st.number_input("Profondeur min", value=100, step=50, key="coocc_d")
    with cc2:
        coocc_ar = st.number_input("AR min", value=0.05, step=0.01, format="%.2f", key="coocc_ar")
    with cc3:
        coocc_min_pat = st.number_input("Fréquence min (patients)", value=3, min_value=2, step=1,
            help="L'entité doit être présente chez au moins N patients pour être analysée.")

    coocc_exclude_benign = st.checkbox("Exclure Benign / Likely Benign", value=True, key="coocc_ben")
    coocc_exclude_nonfunc = st.checkbox("Exclure variants non fonctionnels", value=True, key="coocc_nf")

    # Filtrage
    df_coocc = df_f[(df_f["Depth"] >= coocc_depth) & (df_f["Allelic_ratio"] >= coocc_ar)].copy()
    if coocc_exclude_benign:
        df_coocc = df_coocc[~df_coocc["ACMG_class"].isin(["Benign", "Likely Benign"])]
    if coocc_exclude_nonfunc:
        df_coocc = df_coocc[~df_coocc["Variant_effect"].isin(NON_FUNCTIONAL_EFFECTS)]

    entity_col = "Gene_symbol" if coocc_level == "Par gène" else "Variant"

    st.markdown(f"**{len(df_coocc):,} variants** après filtrage, **{df_coocc[entity_col].nunique()} {coocc_level.lower().replace('par ', '')}s uniques**")

    if len(df_coocc) > 0 and df_coocc["Pseudo"].nunique() >= 3:
        with st.spinner("Construction de la matrice binaire..."):
            binary = compute_cooccurrence_matrix(df_coocc, entity_col, min_patients=coocc_min_pat)

        if binary.shape[1] < 2:
            st.warning("Pas assez d'entités fréquentes. Réduisez le seuil de fréquence min.")
        else:
            st.markdown(f"**{binary.shape[1]} entités** analysées (présentes chez ≥{coocc_min_pat} patients)")

            # ── HEATMAP DE CO-OCCURRENCE ──
            st.markdown("### Matrice de co-occurrence")
            # Matrice de Jaccard pour la heatmap
            n_entities = min(binary.shape[1], 40)  # Limiter pour lisibilité
            top_entities = binary.sum().sort_values(ascending=False).head(n_entities).index
            binary_top = binary[top_entities]

            coocc_matrix = binary_top.T.dot(binary_top).astype(float)
            # Normaliser par Jaccard : |A∩B| / |A∪B|
            for i in range(len(coocc_matrix)):
                for j in range(len(coocc_matrix)):
                    union = int((binary_top.iloc[:, i] | binary_top.iloc[:, j]).sum())
                    coocc_matrix.iloc[i, j] = coocc_matrix.iloc[i, j] / max(union, 1)

            fig = px.imshow(coocc_matrix, color_continuous_scale="YlOrRd",
                labels=dict(color="Index Jaccard"), aspect="auto")
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=max(500, n_entities * 18),
                margin=dict(l=150, b=150))
            st.plotly_chart(fig, use_container_width=True)

            # ── TEST DE FISHER ──
            st.markdown("### Tests de Fisher (paires significatives)")
            st.markdown(
                "Test exact de Fisher sur chaque paire : identifie les associations "
                "statistiquement significatives (p < 0.05 après correction Bonferroni)."
            )

            max_pairs_fisher = min(binary.shape[1], 60)
            with st.spinner(f"Tests de Fisher sur {max_pairs_fisher} entités..."):
                fisher_df = compute_pairwise_fisher(binary[binary.sum().sort_values(ascending=False).head(max_pairs_fisher).index],
                                                     max_pairs=max_pairs_fisher)

            if len(fisher_df) > 0:
                # Significatifs
                sig_fisher = fisher_df[fisher_df["P_adjusted"] < 0.05].copy()

                if len(sig_fisher) > 0:
                    co_occ = sig_fisher[sig_fisher["Type"] == "Co-occurrence"].head(20)
                    co_excl = sig_fisher[sig_fisher["Type"] == "Co-exclusion"].head(20)

                    col_o, col_e = st.columns(2)

                    with col_o:
                        st.markdown("#### 🟢 Co-occurrences significatives")
                        if len(co_occ) > 0:
                            display_cols = ["Entity_1", "Entity_2", "Both", "Freq_1", "Freq_2",
                                           "Odds_Ratio", "P_adjusted"]
                            st.dataframe(co_occ[display_cols].reset_index(drop=True),
                                use_container_width=True,
                                column_config={
                                    "Odds_Ratio": st.column_config.NumberColumn("OR", format="%.2f"),
                                    "P_adjusted": st.column_config.NumberColumn("P adj.", format="%.4f"),
                                })
                        else:
                            st.info("Aucune co-occurrence significative.")

                    with col_e:
                        st.markdown("#### 🔴 Co-exclusions significatives")
                        if len(co_excl) > 0:
                            display_cols = ["Entity_1", "Entity_2", "Both", "Freq_1", "Freq_2",
                                           "Odds_Ratio", "P_adjusted"]
                            st.dataframe(co_excl[display_cols].reset_index(drop=True),
                                use_container_width=True,
                                column_config={
                                    "Odds_Ratio": st.column_config.NumberColumn("OR", format="%.2f"),
                                    "P_adjusted": st.column_config.NumberColumn("P adj.", format="%.4f"),
                                })
                        else:
                            st.info("Aucune co-exclusion significative.")

                    # Volcano-like plot
                    st.markdown("### Volcano plot")
                    fisher_plot = fisher_df.copy()
                    fisher_plot["log10_p"] = -np.log10(fisher_plot["P_adjusted"].clip(lower=1e-10))
                    fisher_plot["log2_OR"] = np.log2(fisher_plot["Odds_Ratio"].clip(lower=0.01, upper=100))
                    fisher_plot["label"] = fisher_plot["Entity_1"] + " / " + fisher_plot["Entity_2"]
                    fisher_plot["Significant"] = fisher_plot["P_adjusted"] < 0.05

                    fig = px.scatter(fisher_plot, x="log2_OR", y="log10_p",
                        color="Significant", color_discrete_map={True: "#ff6b6b", False: "#555"},
                        hover_data=["Entity_1", "Entity_2", "Both", "Odds_Ratio", "P_adjusted"],
                        opacity=0.7)
                    fig.add_vline(x=0, line_dash="dash", line_color="#888", opacity=0.5)
                    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#ffd93d", opacity=0.5,
                                  annotation_text="p=0.05")
                    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)", height=500,
                        xaxis_title="log2(Odds Ratio) ← Co-exclusion | Co-occurrence →",
                        yaxis_title="-log10(P ajusté)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucune paire significative après correction Bonferroni. "
                            "Essayez de réduire le seuil de fréquence minimum.")

            # ── ONCOPRINT SIMPLIFIÉ ──
            st.markdown("### Oncoprint")
            st.markdown("Matrice binaire patient × entité (top entités les plus fréquentes).")
            n_onco = st.slider("Nombre d'entités à afficher", 10, 50, 20, key="onco_n")
            top_onco = binary.sum().sort_values(ascending=False).head(n_onco).index
            onco_data = binary[top_onco].T

            fig = px.imshow(onco_data, color_continuous_scale=["#0a192f", "#64ffda"],
                labels=dict(x="Patient", y=coocc_level.replace("Par ", ""), color="Muté"),
                aspect="auto")
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=max(400, n_onco * 20),
                margin=dict(l=180, b=100), xaxis_tickangle=-90)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pas assez de données après filtrage.")

# ═══════════════════════════════════════════════════════
# ANALYSE DES COMPLICATIONS (supervisée)
# ═══════════════════════════════════════════════════════
with tab_compl:
    st.markdown("## 🎯 Analyse des complications")
    st.markdown(
        "Analyse **supervisée** : existe-t-il une signature génomique particulière associée à "
        "l'apparition d'une complication (BO, PNP, MG, ou FDSCS) ? "
        "L'analyse se fait à deux niveaux : par **variant** (peu de signal attendu) et par **gène muté**."
    )

    # ── DÉFINITION DU GROUPE "COMPLIQUÉ" ──
    COMPLICATION_COLS = ["BO", "PNP", "MG", "FDSCS"]
    CLINICAL_ALL_COLS = ["Complication", "Chirurgie", "Recidive", "BO", "PNP", "MG",
                         "FDSCS", "Histo UCD", "Auto Ac"]

    avail_compl = [c for c in COMPLICATION_COLS if c in df_f.columns]
    avail_clin_all = [c for c in CLINICAL_ALL_COLS if c in df_f.columns]

    if len(avail_compl) == 0:
        st.error("Aucune colonne de complication trouvée (BO, PNP, MG, FDSCS).")
        st.stop()

    # Construction de la variable "Complication_any" par patient
    patient_clinical = {}
    for pseudo in df_f["Pseudo"].unique():
        dp = df_f[df_f["Pseudo"] == pseudo]
        # Dict des valeurs cliniques pour ce patient (prend la 1ère non-NaN)
        pat_data = {}
        has_any_clinical = False
        for col in avail_clin_all:
            vals = dp[col].dropna().unique()
            if len(vals) > 0:
                pat_data[col] = vals[0]
                has_any_clinical = True
            else:
                pat_data[col] = None

        # Complication_any : au moins une des 4 complications pures à 1
        has_compl = False
        for col in avail_compl:
            try:
                if pat_data.get(col) is not None and float(pat_data[col]) == 1:
                    has_compl = True
                    break
            except (ValueError, TypeError):
                pass
        pat_data["Complication_any"] = 1 if has_compl else 0
        pat_data["Has_clinical_info"] = has_any_clinical

        patient_clinical[pseudo] = pat_data

    df_pat_clin = pd.DataFrame.from_dict(patient_clinical, orient="index")

    # ── FILTRES & PARAMÈTRES ──
    st.markdown("### ⚙️ Paramètres de l'analyse")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        exclude_no_clinical = st.checkbox(
            "Exclure patients sans données cliniques", value=True,
            help="Exclut les patients pour lesquels toutes les colonnes cliniques sont vides."
        )
    with fc2:
        min_carriers = st.number_input(
            "Nb minimum de patients porteurs", value=2, min_value=2, step=1,
            help="Variant/gène testé seulement s'il est présent chez au moins N patients."
        )
    with fc3:
        correction_method = st.selectbox(
            "Correction multi-tests",
            ["Bonferroni", "FDR (Benjamini-Hochberg)", "Aucune (p brute)"],
            help="Bonferroni : strict. FDR : plus permissif, adapté à l'exploration.",
        )

    # ── FILTRES QUALITÉ DES VARIANTS (comme dans Clustering) ──
    st.markdown("### 🧹 Filtres qualité des variants")
    st.markdown(
        "> Ces filtres s'appliquent **uniquement à cette analyse** et sont indépendants des filtres "
        "globaux de la sidebar. Paramétrez la qualité minimale requise pour inclure un variant dans "
        "les tests d'association."
    )

    fqc1, fqc2, fqc3, fqc4 = st.columns(4)
    with fqc1:
        compl_depth = st.number_input("Profondeur min", value=100, step=50, key="compl_depth",
            help="Exclut les variants à faible couverture (artéfacts FFPE).")
    with fqc2:
        compl_ar = st.number_input("AR min", value=0.05, step=0.01, format="%.2f", key="compl_ar",
            help="Exclut le bruit à très basse fréquence allélique.")
    with fqc3:
        compl_af = st.number_input("gnomAD NFE max", value=0.01, step=0.005, format="%.3f",
            key="compl_af", help="Exclut les polymorphismes fréquents en population européenne.")
    with fqc4:
        compl_excl_ben = st.checkbox("Exclure Benign/LB", value=True, key="compl_ben",
            help="Exclut les variants classés Benign ou Likely Benign.")

    excluded_effects_compl = st.multiselect("Types de variants à exclure",
        sorted(NON_FUNCTIONAL_EFFECTS),
        default=sorted(NON_FUNCTIONAL_EFFECTS),
        key="compl_eff",
        help="Les variants non fonctionnels (synonymes, introniques, UTR) ajoutent du bruit.")

    # ── CONSTRUCTION DE LA COHORTE ──
    if exclude_no_clinical:
        eligible_patients = df_pat_clin[df_pat_clin["Has_clinical_info"]].index.tolist()
    else:
        eligible_patients = df_pat_clin.index.tolist()

    # Point de départ : df_f (filtres globaux appliqués) OU df pour avoir le total brut
    n_variants_base = len(df_f[df_f["Pseudo"].isin(eligible_patients)])

    df_c = df_f[df_f["Pseudo"].isin(eligible_patients)].copy()
    df_c = df_c[
        (df_c["Depth"] >= compl_depth) &
        (df_c["Allelic_ratio"] >= compl_ar) &
        (df_c["gnomad_exomes_NFE_AF"].fillna(0) <= compl_af) &
        (~df_c["Variant_effect"].isin(excluded_effects_compl))
    ]
    if compl_excl_ben:
        df_c = df_c[~df_c["ACMG_class"].isin(["Benign", "Likely Benign"])]

    n_variants_filtered = len(df_c)
    n_patients_filtered = df_c["Pseudo"].nunique()
    pct_kept = n_variants_filtered / max(n_variants_base, 1) * 100

    # Métriques du filtrage
    fm1, fm2, fm3, fm4 = st.columns(4)
    fm1.metric("Avant filtre", f"{n_variants_base:,}")
    fm2.metric("Après filtre", f"{n_variants_filtered:,}")
    fm3.metric("% conservés", f"{pct_kept:.1f}%")
    fm4.metric("Patients conservés", n_patients_filtered)

    if n_variants_filtered < 10:
        st.error("Trop peu de variants après filtrage. Assouplissez les filtres qualité.")
        st.stop()

    # ── MÉTRIQUES COHORTE ──
    st.markdown("---")
    st.markdown("### 👥 Cohorte analysée")

    df_pat_elig = df_pat_clin.loc[eligible_patients]
    n_elig = len(df_pat_elig)
    n_compl = int(df_pat_elig["Complication_any"].sum())
    n_no_compl = n_elig - n_compl
    n_excluded = len(df_pat_clin) - n_elig

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Patients éligibles", n_elig)
    mc2.metric("Avec complication", n_compl)
    mc3.metric("Sans complication", n_no_compl)
    mc4.metric("Exclus (pas de clinique)", n_excluded)

    if exclude_no_clinical and n_excluded > 0:
        excluded_list = df_pat_clin[~df_pat_clin["Has_clinical_info"]].index.tolist()
        with st.expander(f"⚠️ {n_excluded} patients exclus (aucune info clinique)"):
            st.markdown(", ".join(sorted(excluded_list)))

    # ── DÉTAIL DES COMPLICATIONS ──
    st.markdown("### Répartition des complications")
    compl_counts = {}
    for col in avail_compl:
        try:
            compl_counts[col] = int(df_pat_elig[col].fillna(0).astype(float).eq(1).sum())
        except:
            compl_counts[col] = 0

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        fig = go.Figure(go.Bar(
            x=list(compl_counts.keys()), y=list(compl_counts.values()),
            marker_color=["#ff6b6b", "#ffa500", "#ffd93d", "#9b59b6"][:len(compl_counts)],
            text=list(compl_counts.values()), textposition="outside",
        ))
        fig.update_layout(title="Nombre de patients par type de complication",
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=350,
            yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        fig = go.Figure(go.Pie(
            labels=["Compliqué", "Non compliqué"],
            values=[n_compl, n_no_compl],
            marker_colors=["#ff6b6b", "#4ecdc4"],
            hole=0.4, textinfo="label+percent+value",
        ))
        fig.update_layout(title="Compliqué vs Non compliqué",
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── VALIDATION PRÉ-ANALYSE ──
    if n_elig < 5:
        st.error("Moins de 5 patients éligibles. Analyse impossible.")
        st.stop()
    if n_compl == 0 or n_no_compl == 0:
        st.error("Tous les patients sont dans le même groupe — pas de comparaison possible.")
        st.stop()
    if n_compl < 3 or n_no_compl < 3:
        st.warning("⚠️ Très peu de patients dans un des groupes. La puissance statistique sera faible.")

    # ── FONCTION D'ANALYSE PAR ENTITÉ ──
    def run_association_test(df_variants, df_clinical, entity_col, min_carriers, filter_acmg=None):
        """
        Test d'association entre la présence d'une entité (variant ou gène muté)
        et la variable Complication_any via Fisher exact.
        """
        df_test = df_variants.copy()
        if filter_acmg is not None:
            df_test = df_test[df_test["ACMG_class"].isin(filter_acmg)]

        # Matrice binaire patient × entité
        entity_by_patient = df_test.groupby("Pseudo")[entity_col].apply(set)
        all_entities = set()
        for s in entity_by_patient: all_entities.update(s)

        # Comptage porteurs
        entity_carrier_count = df_test.groupby(entity_col)["Pseudo"].nunique()
        frequent_entities = entity_carrier_count[entity_carrier_count >= min_carriers].index.tolist()

        results = []
        patients = df_clinical.index.tolist()
        compl_set = set(df_clinical[df_clinical["Complication_any"] == 1].index)

        for ent in frequent_entities:
            # Liste des porteurs présents dans la cohorte clinique
            carriers = set(df_test[df_test[entity_col] == ent]["Pseudo"].unique()) & set(patients)
            non_carriers = set(patients) - carriers

            a = len(carriers & compl_set)          # porteur & compliqué
            b = len(carriers - compl_set)          # porteur & non compliqué
            c = len(non_carriers & compl_set)      # non porteur & compliqué
            d = len(non_carriers - compl_set)      # non porteur & non compliqué

            if a + b == 0 or c + d == 0:
                continue  # entité inutile

            table = [[a, b], [c, d]]
            try:
                or_val, p_val = fisher_exact(table, alternative="two-sided")
            except:
                or_val, p_val = 1.0, 1.0

            results.append({
                "Entity": ent,
                "N_carriers": a + b,
                "Carriers_with_compl": a,
                "Carriers_without_compl": b,
                "Non_carriers_with_compl": c,
                "Non_carriers_without_compl": d,
                "Freq_compl_carriers": a / max(a + b, 1),
                "Freq_compl_non_carriers": c / max(c + d, 1),
                "Odds_Ratio": or_val,
                "P_value": p_val,
                "Direction": "Enrichi chez compliqués" if or_val > 1 else "Déplété chez compliqués",
            })

        if len(results) == 0:
            return pd.DataFrame()

        df_res = pd.DataFrame(results).sort_values("P_value")

        # Correction multi-tests
        n_tests = len(df_res)
        if correction_method == "Bonferroni":
            df_res["P_adjusted"] = (df_res["P_value"] * n_tests).clip(upper=1.0)
        elif correction_method == "FDR (Benjamini-Hochberg)":
            # BH : rank * p / (n_tests * (rank / n_tests)) = p * n_tests / rank
            df_res_sorted = df_res.sort_values("P_value").reset_index(drop=True)
            df_res_sorted["rank"] = df_res_sorted.index + 1
            df_res_sorted["P_adjusted"] = (df_res_sorted["P_value"] * n_tests / df_res_sorted["rank"]).clip(upper=1.0)
            # Garantir monotonie (BH step-up) — copie writable
            p_adj = np.array(df_res_sorted["P_adjusted"].values, copy=True)
            for i in range(len(p_adj) - 2, -1, -1):
                p_adj[i] = min(p_adj[i], p_adj[i + 1])
            df_res_sorted["P_adjusted"] = p_adj
            df_res_sorted = df_res_sorted.drop(columns="rank")
            df_res = df_res_sorted.sort_values("P_value").reset_index(drop=True)
        else:
            df_res["P_adjusted"] = df_res["P_value"]

        return df_res


    # ═════════════════════════════════════════════════════
    # LANCEMENT DE L'ANALYSE
    # ═════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🧪 Résultats de l'analyse d'association")

    # Deux sous-onglets pour les 2 niveaux
    lvl_var, lvl_gene, lvl_pw, lvl_sig = st.tabs(
        ["📍 Niveau variant", "🧬 Niveau gène (Patho/LP/VUS)",
         "🔬 Niveau pathway", "🔥 Signatures par complication"]
    )

    # ─── NIVEAU VARIANT ───
    with lvl_var:
        st.markdown("### Association par variant")
        st.markdown(
            "Pour chaque variant présent chez au moins 2 patients, on teste si sa présence "
            "est associée au statut complication (test de Fisher exact)."
        )

        with st.spinner("Analyse variant par variant..."):
            df_res_var = run_association_test(df_c, df_pat_elig, "Variant", min_carriers)

        # Enrichissement avec le nom du gène + info hgvs pour chaque variant
        if len(df_res_var) > 0:
            variant_gene_map = df_c.drop_duplicates("Variant").set_index("Variant")[
                ["Gene_symbol", "hgvs.c", "hgvs.p", "Variant_effect", "ACMG_class"]
            ]
            df_res_var = df_res_var.merge(
                variant_gene_map, left_on="Entity", right_index=True, how="left"
            )
            df_res_var = df_res_var.rename(columns={"Gene_symbol": "Gène"})

        st.markdown(f"**{len(df_res_var)} variants testés** (présents chez ≥{min_carriers} patients)")

        if len(df_res_var) == 0:
            st.warning("Aucun variant n'est présent chez suffisamment de patients pour être testé. "
                      "Abaissez le seuil de porteurs minimum ou élargissez la cohorte.")
        else:
            # Significativité
            sig_var = df_res_var[df_res_var["P_adjusted"] < 0.05]
            nominal_sig_var = df_res_var[df_res_var["P_value"] < 0.05]

            # MESSAGE INTERPRÉTATIF
            st.markdown("### 📋 Conclusion statistique")
            if len(sig_var) > 0:
                top_with_gene = sig_var.head(5).apply(
                    lambda r: f"{r['Gène']} ({r['Entity']})" if pd.notna(r['Gène']) else r['Entity'],
                    axis=1
                ).tolist()
                st.success(
                    f"✅ **{len(sig_var)} variant(s) significativement associé(s) à la complication** "
                    f"après correction {correction_method} (p ajustée < 0.05). "
                    f"Top : {', '.join(top_with_gene)}. "
                    f"Ces variants pourraient constituer des biomarqueurs candidats, "
                    f"mais une validation sur une cohorte indépendante est nécessaire."
                )
            elif len(nominal_sig_var) > 0:
                st.warning(
                    f"⚠️ **Aucun variant significatif après correction {correction_method}** "
                    f"(p ajustée < 0.05). Toutefois, {len(nominal_sig_var)} variant(s) présentent une "
                    f"p-value brute < 0.05, suggérant des signaux à explorer mais non concluants. "
                    f"Cela est attendu avec une petite cohorte et beaucoup de tests multiples. "
                    f"Envisagez d'augmenter l'échantillon ou d'utiliser une correction FDR moins stricte."
                )
            else:
                st.info(
                    "ℹ️ **Aucun variant n'est associé à la complication**, même avant correction. "
                    "Résultat attendu : il est rare que plusieurs patients partagent exactement le même "
                    "variant. L'analyse par gène est plus appropriée pour ce type de cohorte."
                )

            # VOLCANO PLOT
            st.markdown("### 📊 Volcano plot")
            df_res_var["log2_OR"] = np.log2(df_res_var["Odds_Ratio"].clip(lower=0.01, upper=100))
            df_res_var["log10_p"] = -np.log10(df_res_var["P_adjusted"].clip(lower=1e-10))
            df_res_var["Significant"] = df_res_var["P_adjusted"] < 0.05
            # Label combiné gène + variant pour le hover
            df_res_var["Label"] = df_res_var.apply(
                lambda r: f"{r['Gène']} | {r['Entity']}" if pd.notna(r.get('Gène')) else r['Entity'],
                axis=1
            )

            fig = px.scatter(
                df_res_var, x="log2_OR", y="log10_p",
                color="Significant",
                color_discrete_map={True: "#ff6b6b", False: "#555555"},
                hover_data=["Label", "Gène", "Entity", "hgvs.p", "Variant_effect",
                            "N_carriers", "Carriers_with_compl",
                            "Odds_Ratio", "P_value", "P_adjusted"],
                opacity=0.7,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#888", opacity=0.5)
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#ffd93d", opacity=0.5,
                          annotation_text="p ajustée = 0.05")
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=500,
                xaxis_title="log2(Odds Ratio) ← Déplété | Enrichi →",
                yaxis_title="-log10(P ajustée)")
            st.plotly_chart(fig, use_container_width=True)

            # TOP HITS TABLE
            st.markdown("### 🔝 Top variants associés")
            display_cols = ["Gène", "Entity", "hgvs.p", "Variant_effect", "ACMG_class",
                           "N_carriers", "Carriers_with_compl", "Carriers_without_compl",
                           "Freq_compl_carriers", "Freq_compl_non_carriers",
                           "Odds_Ratio", "P_value", "P_adjusted", "Direction"]
            display_cols = [c for c in display_cols if c in df_res_var.columns]
            st.dataframe(
                df_res_var[display_cols].head(30).reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "Entity": "Variant",
                    "hgvs.p": "HGVS protéique",
                    "Variant_effect": "Type",
                    "ACMG_class": "ACMG",
                    "Freq_compl_carriers": st.column_config.NumberColumn("Fréq compl. chez porteurs", format="%.2f"),
                    "Freq_compl_non_carriers": st.column_config.NumberColumn("Fréq compl. chez non-porteurs", format="%.2f"),
                    "Odds_Ratio": st.column_config.NumberColumn("OR", format="%.2f"),
                    "P_value": st.column_config.NumberColumn("P brute", format="%.4f"),
                    "P_adjusted": st.column_config.NumberColumn("P ajustée", format="%.4f"),
                })

            # Export
            csv_var = df_res_var.to_csv(index=False, sep=";")
            st.download_button("📥 Résultats variants (CSV)", csv_var,
                              "association_variants.csv", "text/csv", key="dl_var")

    # ─── NIVEAU GÈNE ───
    with lvl_gene:
        st.markdown("### Association par gène")
        st.markdown(
            "Pour chaque gène muté chez au moins 2 patients, on teste si la présence d'une mutation "
            "(classée **Pathogenic, Likely Pathogenic ou VUS**, Benign/LB exclus) est associée au "
            "statut complication. Un variant bénin ne compte pas comme 'gène muté'."
        )

        # Filtre ACMG automatique : Patho + LP + VUS seulement
        acmg_filter = ["Pathogenic", "Likely Pathogenic", "VUS"]
        st.markdown(f"**Classes ACMG incluses** : {', '.join(acmg_filter)}")

        with st.spinner("Analyse gène par gène..."):
            df_res_gene = run_association_test(
                df_c, df_pat_elig, "Gene_symbol", min_carriers,
                filter_acmg=acmg_filter
            )

        st.markdown(f"**{len(df_res_gene)} gènes testés** (mutés chez ≥{min_carriers} patients)")

        if len(df_res_gene) == 0:
            st.warning("Aucun gène n'est muté chez suffisamment de patients pour être testé.")
        else:
            sig_gene = df_res_gene[df_res_gene["P_adjusted"] < 0.05]
            nominal_sig_gene = df_res_gene[df_res_gene["P_value"] < 0.05]

            # MESSAGE INTERPRÉTATIF
            st.markdown("### 📋 Conclusion statistique")
            if len(sig_gene) > 0:
                top_genes = sig_gene.head(5)["Entity"].tolist()
                st.success(
                    f"✅ **{len(sig_gene)} gène(s) significativement associé(s) à la complication** "
                    f"après correction {correction_method} (p ajustée < 0.05). "
                    f"Top gènes : {', '.join(top_genes)}. "
                    f"Ces gènes constituent des pistes biologiques intéressantes et méritent une "
                    f"validation fonctionnelle ou une recherche de cibles thérapeutiques."
                )
            elif len(nominal_sig_gene) > 0:
                top_nominal = nominal_sig_gene.head(5)["Entity"].tolist()
                st.warning(
                    f"⚠️ **Aucun gène significatif après correction {correction_method}** "
                    f"(p ajustée < 0.05). En revanche, {len(nominal_sig_gene)} gène(s) ont une p-value "
                    f"brute < 0.05 : {', '.join(top_nominal)}. "
                    f"Ces signaux suggèrent des pistes exploratoires mais ne résistent pas au "
                    f"contrôle des faux positifs dû aux tests multiples. "
                    f"Avec {n_elig} patients analysés, la puissance statistique est limitée. "
                    f"Essayez la correction FDR (moins stricte) pour identifier les signaux robustes."
                )
            else:
                st.info(
                    "ℹ️ **Aucun gène n'est associé à la complication**, même avant correction. "
                    "Soit le signal n'existe pas à l'échelle individuelle du gène, soit la cohorte "
                    "est trop petite pour le détecter. L'analyse par pathway (à venir) pourrait "
                    "révéler des associations à un niveau fonctionnel plus large."
                )

            # VOLCANO PLOT
            st.markdown("### 📊 Volcano plot")
            df_res_gene["log2_OR"] = np.log2(df_res_gene["Odds_Ratio"].clip(lower=0.01, upper=100))
            df_res_gene["log10_p"] = -np.log10(df_res_gene["P_adjusted"].clip(lower=1e-10))
            df_res_gene["Significant"] = df_res_gene["P_adjusted"] < 0.05

            fig = px.scatter(
                df_res_gene, x="log2_OR", y="log10_p",
                color="Significant",
                color_discrete_map={True: "#ff6b6b", False: "#555555"},
                hover_data=["Entity", "N_carriers", "Carriers_with_compl",
                            "Odds_Ratio", "P_value", "P_adjusted"],
                text=df_res_gene.apply(
                    lambda r: r["Entity"] if r["P_adjusted"] < 0.1 or r.name < 10 else "", axis=1),
                opacity=0.7,
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.add_vline(x=0, line_dash="dash", line_color="#888", opacity=0.5)
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#ffd93d", opacity=0.5,
                          annotation_text="p ajustée = 0.05")
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=550,
                xaxis_title="log2(Odds Ratio) ← Déplété chez compliqués | Enrichi →",
                yaxis_title="-log10(P ajustée)")
            st.plotly_chart(fig, use_container_width=True)

            # TOP HITS TABLE
            st.markdown("### 🔝 Top gènes associés")
            display_cols = ["Entity", "N_carriers", "Carriers_with_compl", "Carriers_without_compl",
                           "Freq_compl_carriers", "Freq_compl_non_carriers",
                           "Odds_Ratio", "P_value", "P_adjusted", "Direction"]
            st.dataframe(
                df_res_gene[display_cols].head(30).reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "Entity": "Gène",
                    "Freq_compl_carriers": st.column_config.NumberColumn("Fréq compl. chez mutés", format="%.2f"),
                    "Freq_compl_non_carriers": st.column_config.NumberColumn("Fréq compl. chez non-mutés", format="%.2f"),
                    "Odds_Ratio": st.column_config.NumberColumn("OR", format="%.2f"),
                    "P_value": st.column_config.NumberColumn("P brute", format="%.4f"),
                    "P_adjusted": st.column_config.NumberColumn("P ajustée", format="%.4f"),
                })

            # ONCOPRINT pour les top gènes
            st.markdown("### 🧬 Oncoprint des top gènes annoté par statut complication")
            n_onco_genes = min(20, len(df_res_gene))
            top_onco_genes = df_res_gene.head(n_onco_genes)["Entity"].tolist()

            # Matrice binaire patient × gène (avec filtre ACMG)
            df_c_filtered = df_c[df_c["ACMG_class"].isin(acmg_filter)]
            oncomat = pd.DataFrame(0, index=eligible_patients, columns=top_onco_genes)
            for gene in top_onco_genes:
                carriers = df_c_filtered[df_c_filtered["Gene_symbol"] == gene]["Pseudo"].unique()
                for p in carriers:
                    if p in oncomat.index:
                        oncomat.loc[p, gene] = 1

            # Trier patients : compliqués d'abord
            pat_order = sorted(eligible_patients,
                key=lambda p: (-int(df_pat_clin.loc[p, "Complication_any"]),
                               -oncomat.loc[p].sum()))
            oncomat = oncomat.loc[pat_order]

            # Créer matrice colorée : 0=non muté, 1=muté sans compl, 2=muté avec compl
            oncomat_colored = oncomat.copy().astype(int)
            for p in pat_order:
                if df_pat_clin.loc[p, "Complication_any"] == 1:
                    oncomat_colored.loc[p] = oncomat.loc[p] * 2  # muté+compl = 2

            # Annotation complication en ligne supplémentaire
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=oncomat_colored.T.values,
                x=oncomat.index,
                y=oncomat.columns,
                colorscale=[[0, "#0a192f"], [0.5, "#4ecdc4"], [1, "#ff6b6b"]],
                zmin=0, zmax=2,
                hovertemplate="Patient: %{x}<br>Gène: %{y}<br>Statut: %{z}<extra></extra>",
                showscale=False,
            ))
            # Bande de statut complication en haut
            compl_strip = [df_pat_clin.loc[p, "Complication_any"] for p in oncomat.index]

            fig.update_layout(
                title="Oncoprint : gènes (Patho/LP/VUS) × patients (triés par statut complication)",
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=max(500, n_onco_genes * 22),
                xaxis_tickangle=-90, margin=dict(l=150, b=120),
                xaxis_title="Patient", yaxis_title="Gène",
            )
            # Légende personnalisée
            fig.add_annotation(
                x=1.02, y=1, xref="paper", yref="paper",
                text="🟥 Muté + Compl. | 🟩 Muté sans Compl. | ⬛ Non muté",
                showarrow=False, font=dict(size=10), xanchor="left",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Afficher le statut complication en-dessous
            status_df = pd.DataFrame({
                "Patient": oncomat.index,
                "Complication": ["🔴 Oui" if df_pat_clin.loc[p, "Complication_any"] == 1 else "⚪ Non"
                                 for p in oncomat.index],
                "N gènes mutés (Patho/LP/VUS)": oncomat.sum(axis=1).values,
            })
            with st.expander("📋 Statut détaillé des patients"):
                st.dataframe(status_df, use_container_width=True, hide_index=True)

            # HEATMAP des gènes significatifs
            if len(nominal_sig_gene) > 0:
                st.markdown("### 🔥 Heatmap des gènes avec p brute < 0.05")
                sig_genes_names = nominal_sig_gene.head(15)["Entity"].tolist()
                hm_data = oncomat[sig_genes_names].T

                fig = go.Figure(go.Heatmap(
                    z=hm_data.values, x=hm_data.columns, y=hm_data.index,
                    colorscale=[[0, "#0a192f"], [1, "#ff6b6b"]],
                    showscale=False,
                    hovertemplate="Patient: %{x}<br>Gène: %{y}<br>Muté: %{z}<extra></extra>",
                ))
                fig.update_layout(
                    title=f"Gènes avec signal nominal × patients (colonnes triées par complication)",
                    template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=max(400, len(sig_genes_names) * 25),
                    xaxis_tickangle=-90, margin=dict(l=150, b=120),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Export
            csv_gene = df_res_gene.to_csv(index=False, sep=";")
            st.download_button("📥 Résultats gènes (CSV)", csv_gene,
                              "association_genes.csv", "text/csv", key="dl_gene")

    # ─── NIVEAU PATHWAY ───
    with lvl_pw:
        st.markdown("### Association par pathway")
        st.markdown(
            "Pour chaque pathway biologique (fichier GMT), on teste si le fait d'avoir **au moins un "
            "gène du pathway muté** (Patho/LP/VUS) est associé au statut complication. "
            "Cette approche agrège le signal au niveau fonctionnel : plusieurs gènes d'une même voie "
            "biologique peuvent converger vers un phénotype commun, même si aucun gène individuel "
            "n'atteint la significativité."
        )

        if pathways_dict is None:
            st.warning(
                "⚠️ **Aucun fichier GMT n'a été chargé.** "
                "Pour activer l'analyse par pathway, chargez un fichier `.gmt` (ex: MSigDB) "
                "dans la barre latérale. Vous pouvez télécharger des fichiers GMT depuis "
                "[MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/)."
            )
        else:
            # Filtrage des pathways pertinents (au moins N gènes du panel)
            st.markdown("### Paramètres spécifiques aux pathways")
            pwc1, pwc2 = st.columns(2)
            with pwc1:
                min_genes_in_panel = st.number_input(
                    "Gènes min du pathway dans le panel", value=3, min_value=2, step=1,
                    help="Un pathway doit contenir au moins N gènes couverts par le panel "
                         "de séquençage pour être testé. Évite les pathways non pertinents."
                )
            with pwc2:
                max_pathways = st.number_input(
                    "Nombre max de pathways à tester", value=500, min_value=50, step=50,
                    help="Les pathways sont triés par nombre de gènes du panel (les plus couverts "
                         "en premier). Limiter le nombre de tests réduit la pénalité de correction "
                         "multi-tests."
                )

            panel_genes = set(df_c["Gene_symbol"].unique())
            relevant_pw = {k: v & panel_genes for k, v in pathways_dict.items()
                          if len(v & panel_genes) >= min_genes_in_panel}
            # Trier par couverture décroissante
            relevant_pw = dict(sorted(relevant_pw.items(),
                key=lambda x: len(x[1]), reverse=True)[:max_pathways])

            st.markdown(f"**{len(relevant_pw)} pathways** retenus (≥{min_genes_in_panel} gènes du panel)")

            if len(relevant_pw) == 0:
                st.warning("Aucun pathway ne contient assez de gènes du panel.")
            else:
                # ── CONSTRUCTION DE LA MATRICE PATIENT × PATHWAY ──
                acmg_filter_pw = ["Pathogenic", "Likely Pathogenic", "VUS"]
                df_c_pw = df_c[df_c["ACMG_class"].isin(acmg_filter_pw)]

                # Diagnostic préalable
                n_pw_variants = len(df_c_pw)
                n_pw_patients = df_c_pw["Pseudo"].nunique()
                n_pw_genes = df_c_pw["Gene_symbol"].nunique()

                with st.expander(f"🔍 Diagnostic : {n_pw_variants} variants Patho/LP/VUS "
                                 f"chez {n_pw_patients} patients sur {n_pw_genes} gènes"):
                    st.markdown(
                        f"- **Variants retenus** (après filtres qualité + ACMG Patho/LP/VUS) : {n_pw_variants}\n"
                        f"- **Patients porteurs** d'au moins 1 variant Patho/LP/VUS : {n_pw_patients} / {len(eligible_patients)}\n"
                        f"- **Gènes touchés** : {n_pw_genes}\n"
                        f"- **Seuil porteurs minimum** : {min_carriers} (paramètre global)\n"
                        f"- **Correction** : {correction_method}"
                    )
                    if n_pw_patients < len(eligible_patients) * 0.5:
                        st.warning(
                            f"⚠️ Seulement {n_pw_patients}/{len(eligible_patients)} patients ont au moins "
                            f"un variant Patho/LP/VUS. Beaucoup de patients n'ont que des variants Benign. "
                            f"L'analyse par pathway va manquer de puissance."
                        )

                with st.spinner(f"Test de Fisher sur {len(relevant_pw)} pathways..."):
                    results_pw = []
                    skipped_low_carriers = 0
                    compl_set = set(df_pat_elig[df_pat_elig["Complication_any"] == 1].index)

                    # Gènes mutés (avec filtre ACMG) par patient
                    muted_genes_by_patient = df_c_pw.groupby("Pseudo")["Gene_symbol"].apply(set)

                    for pw_name, pw_genes in relevant_pw.items():
                        # Patients "porteurs" : au moins un gène du pathway muté
                        carriers = set()
                        for pseudo in eligible_patients:
                            if pseudo in muted_genes_by_patient.index:
                                pat_muted = muted_genes_by_patient.loc[pseudo]
                                if len(pat_muted & pw_genes) > 0:
                                    carriers.add(pseudo)

                        non_carriers = set(eligible_patients) - carriers

                        a = len(carriers & compl_set)
                        b = len(carriers - compl_set)
                        c = len(non_carriers & compl_set)
                        d = len(non_carriers - compl_set)

                        # Filtre : pathway doit avoir au moins N porteurs
                        if a + b < min_carriers:
                            skipped_low_carriers += 1
                            continue
                        if c + d == 0:
                            continue

                        table = [[a, b], [c, d]]
                        try:
                            or_val, p_val = fisher_exact(table, alternative="two-sided")
                        except:
                            or_val, p_val = 1.0, 1.0

                        # Liste des gènes mutés du pathway (pour détail)
                        muted_in_pw = set()
                        for pseudo in carriers:
                            muted_in_pw.update(muted_genes_by_patient.loc[pseudo] & pw_genes)

                        results_pw.append({
                            "Pathway": pw_name,
                            "N_genes_pw": len(pw_genes),
                            "Muted_genes_in_cohort": ", ".join(sorted(muted_in_pw)[:8]) +
                                ("..." if len(muted_in_pw) > 8 else ""),
                            "N_carriers": a + b,
                            "Carriers_with_compl": a,
                            "Carriers_without_compl": b,
                            "Non_carriers_with_compl": c,
                            "Non_carriers_without_compl": d,
                            "Freq_compl_carriers": a / max(a + b, 1),
                            "Freq_compl_non_carriers": c / max(c + d, 1),
                            "Odds_Ratio": or_val,
                            "P_value": p_val,
                            "Direction": "Enrichi chez compliqués" if or_val > 1 else "Déplété chez compliqués",
                        })

                # Résumé du filtrage
                st.markdown(
                    f"**{len(results_pw)} pathways testés** | "
                    f"{skipped_low_carriers} pathways ignorés (< {min_carriers} porteurs)"
                )

                if len(results_pw) == 0:
                    st.warning(
                        f"Aucun pathway ne compte assez de patients porteurs (seuil = {min_carriers}). "
                        f"Essayez de baisser le seuil 'Nb minimum de patients porteurs' en haut de l'onglet."
                    )
                else:
                    df_res_pw = pd.DataFrame(results_pw).sort_values("P_value").reset_index(drop=True)

                    # Correction multi-tests
                    n_tests_pw = len(df_res_pw)
                    if correction_method == "Bonferroni":
                        df_res_pw["P_adjusted"] = (df_res_pw["P_value"] * n_tests_pw).clip(upper=1.0)
                    elif correction_method == "FDR (Benjamini-Hochberg)":
                        df_res_pw["rank"] = df_res_pw.index + 1
                        df_res_pw["P_adjusted"] = (df_res_pw["P_value"] * n_tests_pw / df_res_pw["rank"]).clip(upper=1.0)
                        p_adj = np.array(df_res_pw["P_adjusted"].values, copy=True)
                        for i in range(len(p_adj) - 2, -1, -1):
                            p_adj[i] = min(p_adj[i], p_adj[i + 1])
                        df_res_pw["P_adjusted"] = p_adj
                        df_res_pw = df_res_pw.drop(columns="rank")
                    else:
                        df_res_pw["P_adjusted"] = df_res_pw["P_value"]

                    st.markdown(f"**{len(df_res_pw)} pathways testés** (présents chez ≥{min_carriers} patients)")

                    st.markdown(f"🔧 *Correction appliquée : {correction_method} | "
                                f"P brute min : {df_res_pw['P_value'].min():.6f} | "
                                f"P ajustée min : {df_res_pw['P_adjusted'].min():.6f}*")

                    sig_pw = df_res_pw[df_res_pw["P_adjusted"] < 0.05]
                    nominal_sig_pw = df_res_pw[df_res_pw["P_value"] < 0.05]

                    # MESSAGE INTERPRÉTATIF
                    st.markdown("### 📋 Conclusion statistique")
                    if len(sig_pw) > 0:
                        top_pw_list = sig_pw.head(5)["Pathway"].tolist()
                        st.success(
                            f"✅ **{len(sig_pw)} pathway(s) significativement associé(s) à la complication** "
                            f"après correction {correction_method} (p ajustée < 0.05). "
                            f"Top : {', '.join(top_pw_list[:3])}. "
                            f"Ces voies biologiques convergent dans le profil des patients compliqués "
                            f"et constituent des pistes mécanistiques intéressantes."
                        )
                    elif len(nominal_sig_pw) > 0:
                        top_nominal_pw = nominal_sig_pw.head(5)["Pathway"].tolist()
                        st.warning(
                            f"⚠️ **Aucun pathway significatif après correction {correction_method}** "
                            f"(p ajustée < 0.05). Toutefois, {len(nominal_sig_pw)} pathway(s) ont une "
                            f"p-value brute < 0.05. Top : {', '.join(top_nominal_pw[:3])}. "
                            f"Ces signaux exploratoires peuvent guider des analyses ciblées. "
                            f"L'agrégation au niveau pathway compense partiellement la faible puissance "
                            f"au niveau gène, mais {n_tests_pw} tests augmentent la pénalité de correction."
                        )
                    else:
                        st.info(
                            "ℹ️ **Aucun pathway n'est associé à la complication**, même avant correction. "
                            "La signature biologique des complications n'est pas détectable au niveau "
                            "fonctionnel avec cette cohorte. Cela peut être dû à une taille d'échantillon "
                            "limitée, ou à l'absence réelle d'une signature convergente."
                        )

                    # VOLCANO PLOT
                    st.markdown("### 📊 Volcano plot")
                    df_res_pw["log2_OR"] = np.log2(df_res_pw["Odds_Ratio"].clip(lower=0.01, upper=100))
                    df_res_pw["log10_p"] = -np.log10(df_res_pw["P_adjusted"].clip(lower=1e-10))
                    df_res_pw["Significant"] = df_res_pw["P_adjusted"] < 0.05
                    # Nom court pour l'affichage
                    df_res_pw["Short_name"] = df_res_pw["Pathway"].str[:40]

                    fig = px.scatter(
                        df_res_pw, x="log2_OR", y="log10_p",
                        color="Significant",
                        color_discrete_map={True: "#ff6b6b", False: "#555555"},
                        hover_data=["Pathway", "N_genes_pw", "N_carriers",
                                    "Carriers_with_compl", "Odds_Ratio",
                                    "P_value", "P_adjusted"],
                        text=df_res_pw.apply(
                            lambda r: r["Short_name"] if r["P_value"] < 0.05 else "", axis=1
                        ),
                        opacity=0.7,
                    )
                    fig.update_traces(textposition="top center", textfont_size=8)
                    fig.add_vline(x=0, line_dash="dash", line_color="#888", opacity=0.5)
                    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#ffd93d", opacity=0.5,
                                  annotation_text="p ajustée = 0.05")
                    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)", height=550,
                        xaxis_title="log2(Odds Ratio) ← Déplété | Enrichi chez compliqués →",
                        yaxis_title="-log10(P ajustée)")
                    st.plotly_chart(fig, use_container_width=True)

                    # TOP HITS TABLE
                    st.markdown("### 🔝 Top pathways associés")
                    display_cols_pw = ["Pathway", "N_genes_pw", "Muted_genes_in_cohort",
                                       "N_carriers", "Carriers_with_compl", "Carriers_without_compl",
                                       "Freq_compl_carriers", "Freq_compl_non_carriers",
                                       "Odds_Ratio", "P_value", "P_adjusted", "Direction"]
                    display_cols_pw = [c for c in display_cols_pw if c in df_res_pw.columns]
                    st.dataframe(
                        df_res_pw[display_cols_pw].head(30).reset_index(drop=True),
                        use_container_width=True,
                        column_config={
                            "N_genes_pw": st.column_config.NumberColumn("Gènes pw"),
                            "Muted_genes_in_cohort": "Gènes mutés",
                            "N_carriers": "N porteurs",
                            "Carriers_with_compl": "Compl+",
                            "Carriers_without_compl": "Compl-",
                            "Freq_compl_carriers": st.column_config.NumberColumn("Fréq compl porteurs", format="%.2f"),
                            "Freq_compl_non_carriers": st.column_config.NumberColumn("Fréq compl non-port.", format="%.2f"),
                            "Odds_Ratio": st.column_config.NumberColumn("OR", format="%.2f"),
                            "P_value": st.column_config.NumberColumn("P brute", format="%.4f"),
                            "P_adjusted": st.column_config.NumberColumn("P ajustée", format="%.4f"),
                        })

                    # DÉTAIL D'UN PATHWAY
                    st.markdown("### 🔍 Explorer un pathway")
                    pw_choice = st.selectbox(
                        "Sélectionner un pathway",
                        df_res_pw.head(30)["Pathway"].tolist(),
                        help="Visualise les gènes mutés du pathway chez les patients."
                    )
                    if pw_choice:
                        row = df_res_pw[df_res_pw["Pathway"] == pw_choice].iloc[0]
                        pw_all_genes = relevant_pw[pw_choice]

                        # Métriques du pathway
                        pwm1, pwm2, pwm3, pwm4 = st.columns(4)
                        pwm1.metric("Gènes dans pathway (panel)", len(pw_all_genes))
                        pwm2.metric("Patients porteurs", row["N_carriers"])
                        pwm3.metric("Odds Ratio", f"{row['Odds_Ratio']:.2f}")
                        pwm4.metric("P ajustée", f"{row['P_adjusted']:.4f}")

                        # Heatmap gènes × patients pour ce pathway
                        df_c_pw_choice = df_c_pw[df_c_pw["Gene_symbol"].isin(pw_all_genes)]
                        pw_genes_list = sorted(df_c_pw_choice["Gene_symbol"].unique())

                        if len(pw_genes_list) > 0:
                            heatmap_data = pd.DataFrame(0, index=pw_genes_list, columns=eligible_patients)
                            for _, r in df_c_pw_choice.iterrows():
                                if r["Pseudo"] in heatmap_data.columns:
                                    heatmap_data.loc[r["Gene_symbol"], r["Pseudo"]] = 1

                            # Trier patients : compliqués d'abord
                            pat_order = sorted(
                                eligible_patients,
                                key=lambda p: (-int(df_pat_clin.loc[p, "Complication_any"]),
                                               -heatmap_data[p].sum())
                            )
                            heatmap_data = heatmap_data[pat_order]

                            # Colorier : 2 si muté + compliqué, 1 si muté sans compl, 0 sinon
                            colored = heatmap_data.copy().astype(int)
                            for p in pat_order:
                                if df_pat_clin.loc[p, "Complication_any"] == 1:
                                    colored[p] = heatmap_data[p] * 2

                            fig = go.Figure(go.Heatmap(
                                z=colored.values, x=colored.columns, y=colored.index,
                                colorscale=[[0, "#0a192f"], [0.5, "#4ecdc4"], [1, "#ff6b6b"]],
                                zmin=0, zmax=2, showscale=False,
                                hovertemplate="Patient: %{x}<br>Gène: %{y}<br>Statut: %{z}<extra></extra>",
                            ))
                            fig.update_layout(
                                title=f"Gènes mutés (Patho/LP/VUS) du pathway : {pw_choice[:60]}",
                                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                height=max(400, len(pw_genes_list) * 25),
                                xaxis_tickangle=-90, margin=dict(l=150, b=120),
                            )
                            fig.add_annotation(
                                x=1.02, y=1, xref="paper", yref="paper",
                                text="🟥 Muté + Compl | 🟩 Muté sans Compl | ⬛ Non muté",
                                showarrow=False, font=dict(size=10), xanchor="left",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Export
                    csv_pw = df_res_pw.to_csv(index=False, sep=";")
                    st.download_button("📥 Résultats pathways (CSV)", csv_pw,
                                      "association_pathways.csv", "text/csv", key="dl_pw")

    # ─── SIGNATURES PAR TYPE DE COMPLICATION ───
    with lvl_sig:
        st.markdown("### 🔥 Signatures par type de complication")
        st.markdown(
            "Visualisation des mutations partagées entre patients **selon le type de complication** "
            "(BO, PNP, MG, FDSCS). Permet d'identifier si un sous-type de complication est associé "
            "à un profil mutationnel particulier."
        )

        # Radio pour choisir le niveau
        sig_level = st.radio("Niveau d'analyse", ["Par gène", "Par variant"],
            horizontal=True, key="sig_level",
            help="Gène : patients partageant une mutation sur le même gène. "
                 "Variant : patients partageant exactement le même variant.")

        acmg_filter_sig = ["Pathogenic", "Likely Pathogenic", "VUS"]
        df_c_sig = df_c[df_c["ACMG_class"].isin(acmg_filter_sig)].copy()

        entity_col_sig = "Gene_symbol" if sig_level == "Par gène" else "Variant"
        entity_label = "gène" if sig_level == "Par gène" else "variant"

        # Patients compliqués uniquement, avec détail par type
        compl_patients = {}  # patient -> list of complications
        for pseudo in eligible_patients:
            pat_compls = []
            for compl_type in avail_compl:
                try:
                    val = df_pat_clin.loc[pseudo, compl_type]
                    if pd.notna(val) and float(val) == 1:
                        pat_compls.append(compl_type)
                except:
                    pass
            if pat_compls:
                compl_patients[pseudo] = pat_compls

        if len(compl_patients) < 2:
            st.warning("Pas assez de patients compliqués pour cette analyse.")
        else:
            st.markdown(f"**{len(compl_patients)} patients compliqués** avec données de complication détaillées")

            # ── VUE 0 : CO-SURVENUE DES COMPLICATIONS ──
            st.markdown("### 🔄 Co-survenue des complications")
            st.markdown("Quelles complications tendent à coexister chez les mêmes patients ?")

            cooccur_compl = pd.DataFrame(0, index=avail_compl, columns=avail_compl, dtype=int)
            for pseudo, compls in compl_patients.items():
                for c1 in compls:
                    for c2 in compls:
                        cooccur_compl.loc[c1, c2] += 1

            fig = go.Figure(go.Heatmap(
                z=cooccur_compl.values, x=cooccur_compl.columns, y=cooccur_compl.index,
                colorscale=[[0, "#0a192f"], [1, "#ff6b6b"]],
                text=cooccur_compl.values, texttemplate="%{text}",
                showscale=True, colorbar_title="Patients",
            ))
            fig.update_layout(
                title="Co-survenue des complications (diagonale = effectif par type)",
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=350, width=500,
            )
            st.plotly_chart(fig, use_container_width=False)

            # ── VUE 1 : HEATMAP ENTITÉ × PATIENTS COMPLIQUÉS ──
            st.markdown(f"### 🗺️ Heatmap : {entity_label}s mutés × patients compliqués")
            st.markdown(
                f"Chaque ligne est un {entity_label} muté (Patho/LP/VUS), chaque colonne un patient compliqué, "
                f"coloré par son type de complication principal."
            )

            # Seuil de fréquence pour les entités
            min_freq_sig = st.slider(
                f"Nb min de patients compliqués porteurs du {entity_label}",
                min_value=2, max_value=max(5, len(compl_patients) // 2),
                value=2, key="sig_min_freq"
            )

            # Matrice binaire entité × patients compliqués
            compl_pseudos = list(compl_patients.keys())
            df_c_compl = df_c_sig[df_c_sig["Pseudo"].isin(compl_pseudos)]

            # Entités fréquentes chez les compliqués
            entity_counts = df_c_compl.groupby(entity_col_sig)["Pseudo"].nunique()
            freq_entities = entity_counts[entity_counts >= min_freq_sig].sort_values(ascending=False)

            if len(freq_entities) == 0:
                st.warning(f"Aucun {entity_label} n'est présent chez ≥{min_freq_sig} patients compliqués.")
            else:
                n_show = min(40, len(freq_entities))
                top_entities_sig = freq_entities.head(n_show).index.tolist()

                # Matrice binaire
                hm_binary = pd.DataFrame(0, index=top_entities_sig, columns=compl_pseudos)
                for ent in top_entities_sig:
                    carriers = df_c_compl[df_c_compl[entity_col_sig] == ent]["Pseudo"].unique()
                    for p in carriers:
                        if p in hm_binary.columns:
                            hm_binary.loc[ent, p] = 1

                # Trier les colonnes (patients) par type de complication
                def compl_sort_key(pseudo):
                    compls = compl_patients.get(pseudo, [])
                    return (compls[0] if compls else "ZZZ", -hm_binary[pseudo].sum())
                pat_sorted = sorted(compl_pseudos, key=compl_sort_key)
                hm_binary = hm_binary[pat_sorted]

                # Annotation couleur par complication
                compl_color_map = {"BO": "#ff6b6b", "PNP": "#ffa500", "MG": "#ffd93d", "FDSCS": "#9b59b6"}
                pat_colors = []
                pat_annotations = []
                for p in pat_sorted:
                    compls = compl_patients.get(p, [])
                    pat_colors.append(compl_color_map.get(compls[0], "#888") if compls else "#888")
                    pat_annotations.append(" + ".join(compls))

                # Figure principale
                fig = go.Figure()

                # Heatmap mutations
                fig.add_trace(go.Heatmap(
                    z=hm_binary.values,
                    x=hm_binary.columns, y=hm_binary.index,
                    colorscale=[[0, "#0a192f"], [1, "#64ffda"]],
                    showscale=False, zmin=0, zmax=1,
                    hovertemplate="Patient: %{x}<br>" + entity_label.capitalize() +
                        ": %{y}<br>Muté: %{z}<extra></extra>",
                ))

                fig.update_layout(
                    title=f"Top {n_show} {entity_label}s mutés chez les patients compliqués",
                    template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=max(500, n_show * 22 + 100),
                    xaxis_tickangle=-90,
                    margin=dict(l=180 if sig_level == "Par variant" else 120, b=120, t=80),
                    xaxis_title="Patient", yaxis_title=entity_label.capitalize(),
                )

                # Ajouter annotation des complications sous le graphe
                for i, (p, annot) in enumerate(zip(pat_sorted, pat_annotations)):
                    fig.add_annotation(
                        x=p, y=top_entities_sig[-1],
                        yshift=-20,
                        text=annot,
                        showarrow=False, font=dict(size=7, color=pat_colors[i]),
                        textangle=-90, xanchor="center",
                    )

                st.plotly_chart(fig, use_container_width=True)

                # Légende complications
                legend_html = " &nbsp;|&nbsp; ".join(
                    [f'<span style="color:{v}">⬤ {k}</span>' for k, v in compl_color_map.items()]
                )
                st.markdown(f"**Légende complications** : {legend_html}", unsafe_allow_html=True)

                # ── VUE 2 : MATRICE SIGNATURE — ENTITÉ × TYPE DE COMPLICATION ──
                st.markdown(f"### 📊 Fréquence de mutation par type de complication")
                st.markdown(
                    f"Pour chaque {entity_label} et chaque type de complication : "
                    f"**pourcentage de patients** de ce type portant une mutation. "
                    f"Permet d'identifier les {entity_label}s spécifiques à un sous-type."
                )

                # Calcul de la matrice % par complication
                freq_matrix = pd.DataFrame(0.0, index=top_entities_sig, columns=avail_compl)
                for compl_type in avail_compl:
                    # Patients ayant cette complication
                    pats_with = [p for p, cs in compl_patients.items() if compl_type in cs]
                    n_with = len(pats_with)
                    if n_with == 0:
                        continue
                    for ent in top_entities_sig:
                        carriers = set(df_c_compl[df_c_compl[entity_col_sig] == ent]["Pseudo"].unique())
                        n_carriers_with = len(carriers & set(pats_with))
                        freq_matrix.loc[ent, compl_type] = round(n_carriers_with / n_with * 100, 1)

                # Ajouter colonne "Non compliqué" pour référence
                non_compl_pats = [p for p in eligible_patients if p not in compl_patients]
                if len(non_compl_pats) > 0:
                    df_c_noncompl = df_c_sig[df_c_sig["Pseudo"].isin(non_compl_pats)]
                    n_nc = len(non_compl_pats)
                    freq_non_compl = []
                    for ent in top_entities_sig:
                        carriers = set(df_c_noncompl[df_c_noncompl[entity_col_sig] == ent]["Pseudo"].unique())
                        freq_non_compl.append(round(len(carriers) / n_nc * 100, 1))
                    freq_matrix["Non compliqué"] = freq_non_compl

                fig = px.imshow(
                    freq_matrix,
                    color_continuous_scale=["#0a192f", "#ffa500", "#ff6b6b"],
                    labels=dict(x="Complication", y=entity_label.capitalize(), color="% porteurs"),
                    aspect="auto",
                    text_auto=".0f",
                )
                fig.update_layout(
                    title=f"% de patients porteurs par type de complication — top {n_show} {entity_label}s",
                    template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=max(500, n_show * 22),
                    margin=dict(l=180 if sig_level == "Par variant" else 120, b=60),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── VUE 3 : TABLEAU RÉCAPITULATIF ──
                st.markdown(f"### 📋 Tableau récapitulatif")

                # Pour chaque entité, enrichir avec l'info du gène si variant
                recap_data = []
                for ent in top_entities_sig:
                    row = {"Entity": ent}
                    if sig_level == "Par variant":
                        gene_info = df_c_compl[df_c_compl["Variant"] == ent]["Gene_symbol"].iloc[0] \
                            if len(df_c_compl[df_c_compl["Variant"] == ent]) > 0 else ""
                        hgvsp = df_c_compl[df_c_compl["Variant"] == ent]["hgvs.p"].iloc[0] \
                            if len(df_c_compl[df_c_compl["Variant"] == ent]) > 0 else ""
                        row["Gène"] = gene_info
                        row["HGVS.p"] = hgvsp if pd.notna(hgvsp) else ""
                    row["N patients compliqués"] = int(hm_binary.loc[ent].sum())
                    for compl_type in avail_compl:
                        row[f"% {compl_type}"] = freq_matrix.loc[ent, compl_type]
                    if "Non compliqué" in freq_matrix.columns:
                        row["% Non compl."] = freq_matrix.loc[ent, "Non compliqué"]
                    recap_data.append(row)

                df_recap = pd.DataFrame(recap_data)
                st.dataframe(df_recap.reset_index(drop=True), use_container_width=True,
                    height=min(600, len(df_recap) * 35 + 40))

                # Export
                csv_sig = df_recap.to_csv(index=False, sep=";")
                st.download_button(f"📥 Signatures par complication (CSV)", csv_sig,
                    f"signatures_complication_{entity_label}.csv", "text/csv", key="dl_sig")

    # ── SYNTHÈSE GLOBALE ──
    st.markdown("---")
    st.markdown("### 📝 Synthèse de l'analyse")
    st.markdown(
        f"""
**Cohorte** : {n_elig} patients ({n_compl} compliqués, {n_no_compl} non compliqués)
**Complications pures analysées** : {', '.join(avail_compl)}
**Correction multi-tests** : {correction_method}
**Filtres appliqués à l'analyse** : Profondeur ≥ {compl_depth}, AR ≥ {compl_ar:.2f}, gnomAD NFE ≤ {compl_af:.3f}, Benign/LB {"exclus" if compl_excl_ben else "inclus"}
**Variants** : {n_variants_base:,} avant filtre → {n_variants_filtered:,} après filtre ({pct_kept:.1f}% conservés)
**Patients exclus (sans info clinique)** : {n_excluded if exclude_no_clinical else 0}

**Note méthodologique** : avec {n_elig} patients répartis en {n_compl}/{n_no_compl},
la puissance statistique pour détecter des associations après correction multi-tests est limitée.
Les résultats avec p brute < 0.05 mais p ajustée ≥ 0.05 doivent être considérés comme
**exploratoires** et nécessitent une validation sur une cohorte indépendante.
        """
    )

    # ── EXPORT PDF ──
    st.markdown("---")
    st.markdown("### 📄 Export du rapport complet")
    st.markdown(
        "Génère un rapport PDF avec toutes les figures, tableaux et interprétations "
        "de l'analyse en cours (cohorte, filtres, niveaux variant et gène, méthodologie)."
    )

    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        if st.button("📊 Générer rapport PDF", type="primary", use_container_width=True,
                     key="gen_compl_pdf"):
            # Renommer la colonne "Gène" en "Gene" pour le rapport (évite les problèmes d'encodage)
            df_res_var_report = df_res_var.copy() if 'df_res_var' in dir() and df_res_var is not None else None
            if df_res_var_report is not None and "Gène" in df_res_var_report.columns:
                df_res_var_report = df_res_var_report.rename(columns={"Gène": "Gene"})

            df_res_gene_report = df_res_gene.copy() if 'df_res_gene' in dir() and df_res_gene is not None else None

            excluded_list_report = []
            if exclude_no_clinical and n_excluded > 0:
                excluded_list_report = df_pat_clin[~df_pat_clin["Has_clinical_info"]].index.tolist()

            with st.spinner("Génération du rapport PDF (peut prendre 30-60 secondes)..."):
                try:
                    pdf_bytes = generate_complications_report(
                        df_c=df_c,
                        df_pat_elig=df_pat_elig,
                        df_pat_clin=df_pat_clin,
                        df_res_var=df_res_var_report,
                        df_res_gene=df_res_gene_report,
                        n_elig=n_elig, n_compl=n_compl, n_no_compl=n_no_compl,
                        n_excluded=n_excluded,
                        compl_counts=compl_counts, avail_compl=avail_compl,
                        compl_depth=compl_depth, compl_ar=compl_ar, compl_af=compl_af,
                        compl_excl_ben=compl_excl_ben,
                        excluded_effects_compl=excluded_effects_compl,
                        n_variants_base=n_variants_base,
                        n_variants_filtered=n_variants_filtered,
                        pct_kept=pct_kept,
                        min_carriers=min_carriers,
                        correction_method=correction_method,
                        exclude_no_clinical=exclude_no_clinical,
                        excluded_list=excluded_list_report,
                    )
                    st.session_state["compl_pdf"] = pdf_bytes
                    st.success("✅ Rapport généré !")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with exp_c2:
        if "compl_pdf" in st.session_state:
            st.download_button("📥 Télécharger le rapport PDF",
                st.session_state["compl_pdf"],
                "rapport_complications.pdf", "application/pdf",
                use_container_width=True, key="dl_compl_pdf")

# ═══════════════════════════════════════════════════════
# HOMOGÉNÉITÉ DES COHORTES
# ═══════════════════════════════════════════════════════
with tab_qc:
    st.markdown("## ⚖️ Homogénéité des cohortes")
    st.markdown(
        "**Contrôle qualité essentiel** : avant d'interpréter toute association génomique → complication, "
        "il faut vérifier que les deux groupes (compliqués vs non compliqués) sont comparables "
        "sur les paramètres techniques et démographiques. Si un groupe a une meilleure couverture "
        "ou plus de variants, les différences observées pourraient être des artéfacts."
    )

    # ── Construction des groupes ──
    COMPL_COLS_QC = ["BO", "PNP", "MG", "FDSCS"]
    CLINICAL_ALL_QC = ["Complication", "Chirurgie", "Recidive", "BO", "PNP", "MG",
                       "FDSCS", "Histo UCD", "Auto Ac"]

    avail_compl_qc = [c for c in COMPL_COLS_QC if c in df_f.columns]
    avail_clin_qc = [c for c in CLINICAL_ALL_QC if c in df_f.columns]

    # Statut complication par patient
    qc_patient_data = {}
    for pseudo in df_f["Pseudo"].unique():
        dp = df_f[df_f["Pseudo"] == pseudo]
        has_clinical = False
        has_compl = False
        for col in avail_clin_qc:
            vals = dp[col].dropna().unique()
            if len(vals) > 0:
                has_clinical = True
                break
        for col in avail_compl_qc:
            vals = dp[col].dropna().unique()
            if len(vals) > 0:
                try:
                    if float(vals[0]) == 1:
                        has_compl = True
                except:
                    pass
        qc_patient_data[pseudo] = {"has_clinical": has_clinical, "complication": has_compl}

    df_qc_status = pd.DataFrame.from_dict(qc_patient_data, orient="index")

    # Option : exclure patients sans clinique
    qc_excl = st.checkbox("Exclure patients sans données cliniques", value=True, key="qc_excl")
    if qc_excl:
        qc_patients = df_qc_status[df_qc_status["has_clinical"]].index.tolist()
    else:
        qc_patients = df_qc_status.index.tolist()

    df_qc = df_f[df_f["Pseudo"].isin(qc_patients)].copy()
    group_compl = [p for p in qc_patients if df_qc_status.loc[p, "complication"]]
    group_no_compl = [p for p in qc_patients if not df_qc_status.loc[p, "complication"]]

    qm1, qm2, qm3 = st.columns(3)
    qm1.metric("Patients analysés", len(qc_patients))
    qm2.metric("Compliqués", len(group_compl))
    qm3.metric("Non compliqués", len(group_no_compl))

    if len(group_compl) < 2 or len(group_no_compl) < 2:
        st.error("Pas assez de patients dans chaque groupe pour comparer.")
        st.stop()

    # ── CALCUL DES MÉTRIQUES PAR PATIENT ──
    from scipy.stats import mannwhitneyu

    patient_metrics = []
    for pseudo in qc_patients:
        dp = df_qc[df_qc["Pseudo"] == pseudo]
        is_compl = df_qc_status.loc[pseudo, "complication"]

        # Métriques techniques
        n_variants_total = len(dp)
        n_genes = dp["Gene_symbol"].nunique()
        mean_depth = dp["Depth"].mean()
        median_depth = dp["Depth"].median()
        mean_ar = dp["Allelic_ratio"].mean()
        median_ar = dp["Allelic_ratio"].median()
        mean_cadd = dp["CADD_phred"].mean() if dp["CADD_phred"].notna().any() else 0
        median_cadd = dp["CADD_phred"].median() if dp["CADD_phred"].notna().any() else 0

        # Filtrage fonctionnel
        dp_func = dp[~dp["Variant_effect"].isin(NON_FUNCTIONAL_EFFECTS)]
        n_func = len(dp_func)
        dp_plpv = dp[dp["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic", "VUS"])]
        n_plpv = len(dp_plpv)
        dp_patho = dp[dp["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
        n_patho = len(dp_patho)

        # Proportions ACMG
        pct_patho = n_patho / max(n_variants_total, 1) * 100
        pct_vus = len(dp[dp["ACMG_class"] == "VUS"]) / max(n_variants_total, 1) * 100
        pct_benign = len(dp[dp["ACMG_class"].isin(["Benign", "Likely Benign"])]) / max(n_variants_total, 1) * 100

        # VAF
        pct_clonal = (dp["Allelic_ratio"] >= 0.25).sum() / max(n_variants_total, 1) * 100

        # Impact
        mean_impact = dp["impact_score"].mean() if "impact_score" in dp.columns and dp["impact_score"].notna().any() else 0

        # gnomAD
        mean_gnomad = dp["gnomad_exomes_NFE_AF"].mean() if dp["gnomad_exomes_NFE_AF"].notna().any() else 0

        patient_metrics.append({
            "Patient": pseudo,
            "Groupe": "Compliqué" if is_compl else "Non compliqué",
            "N variants (total)": n_variants_total,
            "N variants fonctionnels": n_func,
            "N Patho/LP/VUS": n_plpv,
            "N Patho/LP": n_patho,
            "N gènes mutés": n_genes,
            "Profondeur moyenne": round(mean_depth, 0),
            "Profondeur médiane": round(median_depth, 0),
            "VAF moyenne": round(mean_ar, 3),
            "VAF médiane": round(median_ar, 3),
            "% clonal (VAF≥0.25)": round(pct_clonal, 1),
            "CADD moyen": round(mean_cadd, 1),
            "CADD médian": round(median_cadd, 1),
            "gnomAD NFE moyen": round(mean_gnomad, 5),
            "Impact score moyen": round(mean_impact, 2),
            "% Pathogène": round(pct_patho, 1),
            "% VUS": round(pct_vus, 1),
            "% Benign/LB": round(pct_benign, 1),
        })

    df_metrics = pd.DataFrame(patient_metrics)

    # ── TESTS STATISTIQUES (Mann-Whitney U) ──
    st.markdown("---")
    st.markdown("### 📊 Comparaison statistique des groupes")
    st.markdown(
        "Test de **Mann-Whitney U** (non paramétrique) pour chaque paramètre. "
        "Interprétation : p < 0.05 = différence significative entre les groupes."
    )

    test_cols = [c for c in df_metrics.columns if c not in ["Patient", "Groupe"]]
    test_results = []
    for col in test_cols:
        vals_compl = df_metrics[df_metrics["Groupe"] == "Compliqué"][col].dropna()
        vals_no = df_metrics[df_metrics["Groupe"] == "Non compliqué"][col].dropna()
        if len(vals_compl) >= 2 and len(vals_no) >= 2:
            try:
                stat, pval = mannwhitneyu(vals_compl, vals_no, alternative="two-sided")
            except:
                stat, pval = 0, 1.0
            test_results.append({
                "Paramètre": col,
                "Compliqués (médiane)": round(vals_compl.median(), 2),
                "Compliqués (moy ± sd)": f"{vals_compl.mean():.1f} ± {vals_compl.std():.1f}",
                "Non compliqués (médiane)": round(vals_no.median(), 2),
                "Non compliqués (moy ± sd)": f"{vals_no.mean():.1f} ± {vals_no.std():.1f}",
                "P-value (MW)": round(pval, 4),
                "Significatif": "✅ Oui" if pval < 0.05 else "❌ Non",
            })

    df_tests = pd.DataFrame(test_results)

    # Colorer selon significativité
    st.dataframe(df_tests.reset_index(drop=True), use_container_width=True, height=550,
        column_config={
            "P-value (MW)": st.column_config.NumberColumn("P-value", format="%.4f"),
        })

    # ── INTERPRÉTATION ──
    sig_params = df_tests[df_tests["Significatif"] == "✅ Oui"]
    if len(sig_params) > 0:
        st.warning(
            f"⚠️ **{len(sig_params)} paramètre(s) diffèrent significativement** entre les groupes : "
            f"{', '.join(sig_params['Paramètre'].tolist())}. "
            f"Ces différences doivent être prises en compte dans l'interprétation des résultats "
            f"de l'onglet Complications — elles pourraient expliquer des associations artéfactuelles."
        )
    else:
        st.success(
            "✅ **Aucune différence significative** entre les groupes sur les paramètres techniques. "
            "Les cohortes sont homogènes — les associations trouvées dans l'onglet Complications "
            "ne sont probablement pas des artéfacts techniques."
        )

    # ── VISUALISATIONS ──
    st.markdown("---")
    st.markdown("### 📈 Visualisations détaillées")

    # Sélection du paramètre à visualiser
    viz_param = st.selectbox("Paramètre à visualiser", test_cols, key="qc_param")

    vl, vr = st.columns(2)

    with vl:
        # Box plot
        fig = px.box(df_metrics, x="Groupe", y=viz_param, color="Groupe",
            color_discrete_map={"Compliqué": "#ff6b6b", "Non compliqué": "#4ecdc4"},
            points="all", notched=True if len(group_compl) >= 5 and len(group_no_compl) >= 5 else False)
        # Ajouter p-value dans le titre
        pval_display = df_tests[df_tests["Paramètre"] == viz_param]["P-value (MW)"].values
        pval_str = f"p = {pval_display[0]:.4f}" if len(pval_display) > 0 else ""
        fig.update_layout(
            title=f"{viz_param} — {pval_str}",
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=450,
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with vr:
        # Histogramme superposé
        fig = go.Figure()
        for grp, color in [("Compliqué", "#ff6b6b"), ("Non compliqué", "#4ecdc4")]:
            vals = df_metrics[df_metrics["Groupe"] == grp][viz_param]
            fig.add_trace(go.Histogram(x=vals, name=grp, marker_color=color,
                opacity=0.6, nbinsx=15))
        fig.update_layout(
            title=f"Distribution : {viz_param}",
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=450,
            barmode="overlay", xaxis_title=viz_param, yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

    # ── MATRICE COMPLÈTE ──
    st.markdown("### 📋 Vue panoramique (tous paramètres)")

    # Heatmap normalisée : z-score par paramètre, trié par groupe
    st.markdown(
        "Heatmap des z-scores (centrés-réduits) : chaque paramètre est normalisé pour que "
        "la moyenne = 0 et l'écart-type = 1. Les couleurs extrêmes indiquent les patients atypiques."
    )

    hm_cols = test_cols
    hm_data = df_metrics.set_index("Patient")[hm_cols].copy()
    # Z-score
    for col in hm_cols:
        m = hm_data[col].mean()
        s = hm_data[col].std()
        if s > 0:
            hm_data[col] = (hm_data[col] - m) / s
        else:
            hm_data[col] = 0

    # Trier : compliqués d'abord
    pat_order_qc = sorted(hm_data.index,
        key=lambda p: (-int(df_qc_status.loc[p, "complication"]), p))
    hm_data = hm_data.loc[pat_order_qc]

    fig = px.imshow(hm_data.T,
        color_continuous_scale=["#4ecdc4", "#0a192f", "#ff6b6b"],
        labels=dict(x="Patient", y="Paramètre", color="Z-score"),
        aspect="auto", color_continuous_midpoint=0)
    fig.update_layout(
        title="Z-scores par patient (triés : compliqués à gauche)",
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(500, len(hm_cols) * 25),
        xaxis_tickangle=-90, margin=dict(l=180, b=120))

    # Ajouter annotation du groupe
    for i, p in enumerate(pat_order_qc):
        color = "#ff6b6b" if df_qc_status.loc[p, "complication"] else "#4ecdc4"
        fig.add_annotation(
            x=p, y=hm_cols[-1], yshift=15,
            text="●", showarrow=False,
            font=dict(size=10, color=color))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("🔴 = Compliqué &nbsp; 🟢 = Non compliqué", unsafe_allow_html=True)

    # ── RADAR / PROFIL MOYEN PAR GROUPE ──
    st.markdown("### 🎯 Profil moyen par groupe (radar)")

    # Normaliser 0-1 pour le radar
    radar_cols = ["N variants (total)", "N gènes mutés", "Profondeur médiane",
                  "VAF médiane", "% clonal (VAF≥0.25)", "CADD médian",
                  "Impact score moyen", "% Pathogène", "% VUS", "N Patho/LP/VUS"]
    radar_cols = [c for c in radar_cols if c in df_metrics.columns]

    radar_data = df_metrics.groupby("Groupe")[radar_cols].mean()
    # Normaliser 0-1 pour comparaison visuelle
    radar_norm = radar_data.copy()
    for col in radar_cols:
        col_min = df_metrics[col].min()
        col_max = df_metrics[col].max()
        if col_max > col_min:
            radar_norm[col] = (radar_data[col] - col_min) / (col_max - col_min)
        else:
            radar_norm[col] = 0.5

    fig = go.Figure()
    color_rgba = {"Compliqué": "rgba(255,107,107,0.15)", "Non compliqué": "rgba(78,205,196,0.15)"}
    color_line = {"Compliqué": "#ff6b6b", "Non compliqué": "#4ecdc4"}
    for grp in ["Compliqué", "Non compliqué"]:
        if grp in radar_norm.index:
            vals = radar_norm.loc[grp].values.tolist()
            vals.append(vals[0])
            labels = radar_cols + [radar_cols[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels, fill='toself', name=grp,
                fillcolor=color_rgba[grp],
                line_color=color_line[grp], opacity=0.8,
            ))

    fig.update_layout(
        title="Profil moyen normalisé par groupe",
        template="plotly_dark",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── EXPORT ──
    st.markdown("---")
    csv_qc = df_metrics.to_csv(index=False, sep=";")
    st.download_button("📥 Métriques par patient (CSV)", csv_qc,
                      "homogeneite_cohortes.csv", "text/csv", key="dl_qc")

    csv_tests_qc = df_tests.to_csv(index=False, sep=";")
    st.download_button("📥 Tests statistiques (CSV)", csv_tests_qc,
                      "tests_homogeneite.csv", "text/csv", key="dl_qc_tests")

# ═══════════════════════════════════════════════════════
# CLUSTERING + PATHWAYS + INTERPRÉTATION
# ═══════════════════════════════════════════════════════
with tab_clust:
    st.markdown("## 🔬 Clustering des patients")
    st.markdown(
        "Signatures **génomiques**, **cliniques** et **pathways** combinées. "
        "UMAP + clustering, interprétation statistique et IA."
    )

    if df_f["Pseudo"].nunique() < 5:
        st.warning("Au moins 5 patients nécessaires."); st.stop()

    # ── FILTRES QUALITÉ ──
    st.markdown("### 🧹 Filtres qualité (pré-clustering)")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        cl_depth = st.number_input("Profondeur min", value=100, step=50, key="cld")
    with qc2:
        cl_ar = st.number_input("AR min", value=0.05, step=0.01, format="%.2f", key="cla")
    with qc3:
        cl_af = st.number_input("gnomAD NFE max", value=0.01, step=0.005, format="%.3f", key="clf")
    with qc4:
        cl_excl_ben = st.checkbox("Exclure Benign/LB", value=True, key="clb")

    excluded_effects = st.multiselect("Types à exclure", sorted(NON_FUNCTIONAL_EFFECTS),
        default=sorted(NON_FUNCTIONAL_EFFECTS), key="cl_eff")

    df_clust = df_f[
        (df_f["Depth"] >= cl_depth) & (df_f["Allelic_ratio"] >= cl_ar) &
        (df_f["gnomad_exomes_NFE_AF"].fillna(0) <= cl_af) &
        (~df_f["Variant_effect"].isin(excluded_effects))
    ].copy()
    if cl_excl_ben:
        df_clust = df_clust[~df_clust["ACMG_class"].isin(["Benign", "Likely Benign"])]

    n_before, n_after = len(df_f), len(df_clust)
    n_pat_after = df_clust["Pseudo"].nunique()

    qm1, qm2, qm3, qm4 = st.columns(4)
    qm1.metric("Avant filtre", f"{n_before:,}")
    qm2.metric("Après filtre", f"{n_after:,}")
    qm3.metric("% conservés", f"{n_after/max(n_before,1)*100:.1f}%")
    qm4.metric("Patients", n_pat_after)

    if n_pat_after < 5:
        st.error("< 5 patients après filtrage."); st.stop()

    # ── PARAMÈTRES CLUSTERING ──
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")

    # ── OPTION CLUSTERING 2 ÉTAPES ──
    two_step = st.checkbox("🔬 Clustering en 2 étapes (recommandé)", value=True,
        help="Étape 1 : exclut automatiquement les patients suspects FFPE (VAF médiane très basse) "
             "pour éviter que le bruit de dégradation ne pollue les groupements biologiques. "
             "Étape 2 : clustering des patients fiables uniquement.")

    if two_step:
        st.markdown(
            "> **Mode 2 étapes** : les patients dont la VAF médiane (sur variants filtrés) "
            "est inférieure au seuil sont isolés dans un groupe **« Suspect FFPE »** avant le clustering."
        )
        ts1, ts2 = st.columns(2)
        with ts1:
            vaf_threshold = st.number_input("Seuil VAF médiane (suspect FFPE)",
                value=0.03, step=0.005, format="%.3f", key="vaf_thr",
                help="Patients avec VAF médiane < seuil → exclus du clustering biologique")
        with ts2:
            min_variants_threshold = st.number_input("Nb variants min par patient",
                value=5, step=1, key="min_var_thr",
                help="Patients avec trop peu de variants → exclus (échecs techniques)")

        # Identifier les suspects
        patient_vaf_stats = df_clust.groupby("Pseudo").agg(
            vaf_med=("Allelic_ratio", "median"),
            n_var=("Variant", "count"),
        )
        suspect_ffpe = patient_vaf_stats[patient_vaf_stats["vaf_med"] < vaf_threshold].index.tolist()
        too_few = patient_vaf_stats[patient_vaf_stats["n_var"] < min_variants_threshold].index.tolist()
        excluded_patients = list(set(suspect_ffpe + too_few))
        reliable_patients = [p for p in df_clust["Pseudo"].unique() if p not in excluded_patients]

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Patients fiables", len(reliable_patients))
        ec2.metric("Suspects FFPE", len(suspect_ffpe))
        ec3.metric("Trop peu de variants", len(too_few))

        if suspect_ffpe:
            with st.expander(f"⚠️ {len(excluded_patients)} patients exclus du clustering biologique"):
                for p in sorted(excluded_patients):
                    stats = patient_vaf_stats.loc[p]
                    reasons = []
                    if p in suspect_ffpe: reasons.append(f"VAF médiane={stats['vaf_med']:.3f}")
                    if p in too_few: reasons.append(f"N variants={int(stats['n_var'])}")
                    st.markdown(f"- **{p}** : {', '.join(reasons)}")

        df_clust_bio = df_clust[df_clust["Pseudo"].isin(reliable_patients)].copy()

        if len(reliable_patients) < 5:
            st.error("< 5 patients fiables. Abaissez le seuil VAF."); st.stop()
    else:
        df_clust_bio = df_clust.copy()
        excluded_patients = []

    cp1, cp2, cp3, cp4 = st.columns(4)
    with cp1: use_gen = st.checkbox("Génomique", True)
    with cp2: use_clin = st.checkbox("Clinique", True)
    with cp3: use_pw = st.checkbox("Pathways", value=pathways_dict is not None,
        disabled=pathways_dict is None,
        help="Nécessite un fichier GMT chargé dans la sidebar.")
    with cp4: top_n = st.slider("Top N gènes", 10, 50, 30, 5)

    cc1, cc2 = st.columns(2)
    with cc1: method = st.selectbox("Méthode", ["Hiérarchique (Ward)", "K-Means"])
    with cc2: n_clust = st.slider("Nb clusters", 2, 8, 5)

    cu1, cu2 = st.columns(2)
    with cu1: n_neigh = st.slider("UMAP n_neighbors", 3, 30, 10)
    with cu2: m_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.3, 0.05)

    if not use_gen and not use_clin and not use_pw:
        st.warning("Sélectionnez au moins un type de features."); st.stop()

    # ── BUILD & CLUSTER ──
    with st.spinner("Construction matrice..."):
        df_feat = build_patient_features(df_clust_bio, use_gen, use_clin,
            use_pathways=use_pw and pathways_dict is not None,
            top_n_genes=top_n, pathways_dict=pathways_dict)

    st.markdown(f"**{df_feat.shape[0]} patients × {df_feat.shape[1]} features**")

    # Afficher le détail des features
    with st.expander("📋 Détail des features utilisées"):
        feat_types = {"Génomique (proportions)": [c for c in df_feat.columns if c.startswith("pct_")],
                      "Génomique (scores)": [c for c in df_feat.columns if c.startswith("impact_score") or c.startswith("median_") or c.startswith("mean_") or c == "n_unique_genes"],
                      "VAF / Clonalité": [c for c in df_feat.columns if c.startswith("vaf_") or c.startswith("pct_clonal") or c.startswith("pct_subclonal") or c.startswith("pct_minor") or c in ["clonal_ratio", "tmb_score"]],
                      "Gènes (binaire)": [c for c in df_feat.columns if c.startswith("gene_")],
                      "Gènes (score impact)": [c for c in df_feat.columns if c.startswith("genescore_")],
                      "Pathways (%)": [c for c in df_feat.columns if c.startswith("pw_pct_")],
                      "Pathways (score)": [c for c in df_feat.columns if c.startswith("pw_score_")],
                      "Clinique": [c for c in df_feat.columns if c in ["Histo_HV", "Histo_mixed"] + CLINICAL_COLS]}
        for cat, cols in feat_types.items():
            if cols: st.markdown(f"- **{cat}** : {len(cols)} features")

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
    df_u = pd.DataFrame({"UMAP_1": emb[:, 0], "UMAP_2": emb[:, 1],
        "Cluster": cluster_labels, "Patient": df_feat.index})
    for col in df_feat.columns:
        if (col.startswith("pct_") or col.startswith("median_") or col.startswith("mean_")
            or col.startswith("impact_score") or col.startswith("pw_pct_")
            or col.startswith("vaf_") or col.startswith("tmb_")
            or col in ["n_unique_genes", "clonal_ratio", "Histo_HV", "Histo_mixed",
                       "Complication", "Chirurgie", "Recidive", "BO", "PNP", "MG", "FDSCS"]):
            df_u[col] = df_feat[col].values

    # Add excluded patients as "Suspect FFPE" cluster if two-step mode
    if two_step and excluded_patients:
        for pat in excluded_patients:
            df_u = pd.concat([df_u, pd.DataFrame([{
                "UMAP_1": np.nan, "UMAP_2": np.nan,
                "Cluster": "⚠️ Suspect FFPE", "Patient": pat,
            }])], ignore_index=True)

    ccols = px.colors.qualitative.Bold[:n_clust]
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clusters", n_clust); m2.metric("Silhouette", f"{sil:.3f}"); m3.metric("Features", df_feat.shape[1])

    # ── UMAP ──
    st.markdown("### Projection UMAP")
    hover_extra = [c for c in ["pct_Pathogenic", "pct_Likely Pathogenic", "pct_VUS",
                               "vaf_median", "pct_clonal", "tmb_score",
                               "impact_score_mean", "n_unique_genes"] if c in df_u.columns]
    fig = px.scatter(df_u, x="UMAP_1", y="UMAP_2", color="Cluster", text="Patient",
        hover_data=["Patient", "Cluster"] + hover_extra, color_discrete_sequence=ccols)
    fig.update_traces(textposition="top center", textfont_size=10, marker_size=12)
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # ── DENDROGRAMME ──
    if method == "Hiérarchique (Ward)":
        st.markdown("### Dendrogramme")
        Z = linkage(X, method="ward")
        fig_d = ff.create_dendrogram(X, labels=df_feat.index.tolist(), linkagefun=lambda x: Z,
            color_threshold=Z[-(n_clust-1), 2] if n_clust > 1 else 0)
        fig_d.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=400, xaxis_tickangle=-90, margin=dict(b=120))
        st.plotly_chart(fig_d, use_container_width=True)

    # ── PROFIL CLUSTERS ──
    st.markdown("### Profil des clusters")
    df_fc = df_feat.copy()
    df_fc["Cluster"] = cluster_labels

    key_gen = [c for c in df_fc.columns if c in [
        "pct_Pathogenic", "pct_Likely Pathogenic", "pct_VUS", "pct_Likely Benign", "pct_Benign",
        "pct_impact_high", "pct_impact_moderate", "pct_impact_low",
        "pct_missensevariant", "pct_frameshiftvariant", "pct_stopgained",
        "impact_score_mean", "impact_score_max", "pct_high_impact_score",
        "median_CADD", "mean_gnomAD_AF", "n_unique_genes",
        "vaf_median", "vaf_mean", "vaf_iqr", "pct_clonal", "pct_subclonal",
        "pct_minor", "clonal_ratio", "tmb_score"]]
    key_clin = [c for c in df_fc.columns if c in [
        "Histo_HV", "Histo_mixed", "Complication", "Chirurgie",
        "Recidive", "BO", "PNP", "MG", "FDSCS"]]
    key_pw = [c for c in df_fc.columns if c.startswith("pw_pct_")][:15]  # Top 15 pathways
    key_feat = (key_gen if use_gen else []) + (key_clin if use_clin else []) + (key_pw if use_pw else [])

    if key_feat:
        cp = df_fc.groupby("Cluster")[key_feat].mean().round(2)
        fig = px.imshow(cp.T, color_continuous_scale="YlOrRd", aspect="auto",
            labels=dict(x="Cluster", y="Feature", color="Moyenne"), text_auto=".1f")
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=max(500, len(key_feat)*22), margin=dict(l=250))
        st.plotly_chart(fig, use_container_width=True)

    # ── BARPLOTS ──
    if use_clin and key_clin:
        st.markdown("### Profil clinique")
        cm = df_fc.groupby("Cluster")[key_clin].mean()
        fig = px.bar(cm.reset_index().melt(id_vars="Cluster"), x="variable", y="value",
            color="Cluster", barmode="group", color_discrete_sequence=ccols)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=450, title="Profil clinique moyen")
        st.plotly_chart(fig, use_container_width=True)

    if use_gen:
        acmg_c = [f"pct_{c}" for c in ACMG_ORDER if f"pct_{c}" in df_fc.columns]
        if acmg_c:
            st.markdown("### Profil ACMG")
            am = df_fc.groupby("Cluster")[acmg_c].mean()
            am.columns = [c.replace("pct_", "") for c in am.columns]
            fig = go.Figure()
            for cls in am.columns:
                fig.add_trace(go.Bar(name=cls, x=am.index, y=am[cls],
                    marker_color=ACMG_COLORS.get(cls, "#888")))
            fig.update_layout(barmode="stack", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450,
                title="ACMG par cluster (%)", yaxis_title="% moyen")
            st.plotly_chart(fig, use_container_width=True)

    # ── TOP PATHWAYS PAR CLUSTER ──
    if use_pw and pathways_dict and key_pw:
        st.markdown("### Top pathways par cluster")
        pw_means = df_fc.groupby("Cluster")[key_pw].mean()
        pw_means.columns = [c.replace("pw_pct_", "") for c in pw_means.columns]
        fig = px.imshow(pw_means.T, color_continuous_scale=["#0a192f", "#ff6b6b"],
            labels=dict(x="Cluster", y="Pathway", color="% muté"), aspect="auto", text_auto=".1f")
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=max(400, len(key_pw)*25), margin=dict(l=280))
        st.plotly_chart(fig, use_container_width=True)

    # ── INTERPRÉTATION STATISTIQUE ──
    st.markdown("---")
    st.markdown("## 🧠 Interprétation des clusters")

    if key_feat:
        interpretations = compute_cluster_interpretation(df_feat, cluster_labels, key_feat)
        labels_map = {cid: interp["patients"] for cid, interp in interpretations.items()}
        gene_sigs = get_gene_signature_per_cluster(df_clust_bio, labels_map)

        st.markdown("### 📊 Interprétation statistique")
        for cid in sorted(interpretations.keys()):
            interp = interpretations[cid]
            with st.expander(f"🔹 {cid} — {interp['n_patients']} patients ({', '.join(interp['patients'])})", expanded=True):
                sig = interp["significant"]
                if len(sig) > 0:
                    enriched = sig[sig > 0]
                    depleted = sig[sig < 0]
                    ce, cd = st.columns(2)
                    with ce:
                        st.markdown("**🔺 Enrichi**")
                        for f, z in enriched.items():
                            label = FEATURE_LABELS.get(f, f.replace("pw_pct_", "🔬 ").replace("pct_", "").replace("_", " "))
                            st.markdown(f'<span class="feat-up">▲ {label}</span> : '
                                f'{interp["cluster_mean"][f]:.2f} vs {interp["global_mean"][f]:.2f} (z={z:+.2f})',
                                unsafe_allow_html=True)
                    with cd:
                        st.markdown("**🔻 Réduit**")
                        for f, z in depleted.items():
                            label = FEATURE_LABELS.get(f, f.replace("pw_pct_", "🔬 ").replace("pct_", "").replace("_", " "))
                            st.markdown(f'<span class="feat-down">▼ {label}</span> : '
                                f'{interp["cluster_mean"][f]:.2f} vs {interp["global_mean"][f]:.2f} (z={z:+.2f})',
                                unsafe_allow_html=True)
                else:
                    st.info("Aucune feature significativement discriminante.")

                if cid in gene_sigs:
                    gs = gene_sigs[cid]
                    cg, ce2 = st.columns(2)
                    with cg:
                        if len(gs["pathogenic_genes"]) > 0:
                            st.markdown("**🧬 Gènes pathogènes :**")
                            for g, c in gs["pathogenic_genes"].head(5).items():
                                st.markdown(f"- **{g}** : {c}")
                    with ce2:
                        if len(gs["enriched_genes"]) > 0:
                            st.markdown("**📈 Gènes enrichis :**")
                            for g, r in gs["enriched_genes"].head(5).items():
                                st.markdown(f"- **{g}** : ×{r:.1f}")

        # ── INTERPRÉTATION IA ──
        st.markdown("---")
        st.markdown("### 🤖 Interprétation IA")
        if not ANTHROPIC_AVAILABLE:
            st.warning("Package `anthropic` non installé. Ajoutez-le à requirements.txt.")
        elif not api_key:
            st.info("💡 Entrez votre clé API Anthropic dans la sidebar pour activer l'interprétation IA.")
        else:
            if st.button("🧠 Lancer l'interprétation IA", type="primary", use_container_width=True):
                with st.spinner("Claude analyse vos clusters..."):
                    try:
                        prompt = build_ai_prompt(interpretations, gene_sigs, n_clust)
                        ai_resp = call_anthropic_api(prompt, api_key)
                        st.markdown(f'<div class="ai-interpretation"><h4>🤖 Analyse</h4>{ai_resp}</div>',
                            unsafe_allow_html=True)
                        st.session_state["ai_interpretation"] = ai_resp
                    except Exception as e:
                        st.error(f"Erreur API : {e}")
            elif "ai_interpretation" in st.session_state:
                st.markdown(f'<div class="ai-interpretation"><h4>🤖 Analyse (précédente)</h4>'
                    f'{st.session_state["ai_interpretation"]}</div>', unsafe_allow_html=True)

    st.markdown("### Export")
    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        exp = df_u[["Patient", "Cluster", "UMAP_1", "UMAP_2"]].to_csv(index=False, sep=";")
        st.download_button("📥 Clusters (CSV)", exp, "clusters.csv", "text/csv",
                           use_container_width=True)

    with exp_c2:
        if st.button("📊 Générer rapport PDF complet", type="primary", use_container_width=True):
            with st.spinner("Génération du rapport PDF (peut prendre 30s)..."):
                try:
                    pdf_bytes = generate_cluster_report(
                        df_clust=df_clust_bio if two_step else df_clust,
                        df_feat=df_feat,
                        cluster_labels=cluster_labels,
                        interpretations=interpretations,
                        gene_sigs=gene_sigs,
                        sil_score=sil,
                        method_name=method,
                        key_feat=key_feat,
                        excluded_patients=excluded_patients if two_step else [],
                        pathways_dict=pathways_dict,
                    )
                    st.session_state["pdf_report"] = pdf_bytes
                    st.success("✅ Rapport généré !")
                except Exception as e:
                    st.error(f"Erreur lors de la génération du rapport : {e}")
                    import traceback
                    st.code(traceback.format_exc())

    if "pdf_report" in st.session_state:
        st.download_button("📥 Télécharger le rapport PDF",
            st.session_state["pdf_report"],
            "rapport_clustering.pdf", "application/pdf",
            use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;color:#4a5568;font-size:0.85rem;">'
    '🧬 Variant Explorer v4.0 — Données locales uniquement</p>', unsafe_allow_html=True)
