"""Chargement des données, classification ACMG, scoring d'impact, chargement GMT."""
import pandas as pd
import streamlit as st

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
