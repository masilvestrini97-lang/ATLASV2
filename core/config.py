"""Constantes partagées du Variant Explorer."""

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
