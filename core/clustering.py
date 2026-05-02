"""Interprétation des clusters et signatures de gènes."""

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

