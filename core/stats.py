"""Statistiques : co-occurrence/co-exclusion et test exact de Fisher."""
import pandas as pd
from scipy.stats import fisher_exact

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


