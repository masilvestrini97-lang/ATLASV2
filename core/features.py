"""Construction de la matrice patient × features pour clustering."""
import pandas as pd
from core.config import ACMG_ORDER, IMPACT_ORDER, CLINICAL_COLS
from core.data_loader import get_relevant_pathways

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

