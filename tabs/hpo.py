"""
Onglet : Phenotypes HPO et maladies associees (Monarch Initiative).

Pour chaque gene mute du panel, recupere :
- Les termes HPO (Human Phenotype Ontology) associes
- Les maladies (OMIM, MONDO, Orphanet) liees au gene

Source : api.monarchinitiative.org (gratuit, sans cle).
"""
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from core.api_clients import (
    query_monarch_genes_batch,
    clear_api_cache,
)


def render(df_f, df, pathways_dict, api_key):
    st.markdown("## 🧬 Phenotypes HPO & maladies associees")
    st.markdown(
        "Pour chaque gene mute, recupere les termes **HPO (Human Phenotype Ontology)** "
        "et les **maladies** associees via [Monarch Initiative](https://monarchinitiative.org). "
        "Permet d'identifier rapidement les patterns phenotypiques recurrents dans la cohorte."
    )

    # ── SCOPE ──
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        scope = st.radio(
            "**Genes a interroger**",
            ["Tous les genes mutes", "Genes Patho/LP uniquement", "Selection manuelle"],
            horizontal=True,
        )
    with sc2:
        if st.button("🗑️ Vider cache HPO", width='stretch'):
            clear_api_cache("monarch_hpo")
            st.success("Cache Monarch vide.")

    if scope == "Tous les genes mutes":
        target_genes = sorted(df_f["Gene_symbol"].unique())
    elif scope == "Genes Patho/LP uniquement":
        sub = df_f[df_f["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
        target_genes = sorted(sub["Gene_symbol"].unique())
    else:
        target_genes = st.multiselect(
            "Genes",
            sorted(df_f["Gene_symbol"].unique()),
            help="Selectionnez les genes a interroger sur Monarch."
        )

    n = len(target_genes)
    st.markdown(f"**{n} genes** a interroger")

    if n == 0:
        st.info("Aucun gene a interroger.")
        return

    if n > 100:
        st.warning(
            f"⚠️ {n} genes a interroger. Premier appel = ~{n*0.5:.0f}s "
            f"(API rate-limited a ~2 req/s). Les appels suivants utilisent le cache."
        )

    if not st.button("🚀 Interroger Monarch", type="primary", width='stretch'):
        return

    # ── REQUÊTE ──
    progress = st.progress(0.0, text="Interrogation Monarch...")
    results = query_monarch_genes_batch(
        target_genes,
        progress_callback=lambda p: progress.progress(p, text=f"Monarch — {int(p*100)}%"),
    )
    progress.empty()

    n_with_hpo = sum(1 for r in results.values() if r["phenotypes"])
    n_with_disease = sum(1 for r in results.values() if r["diseases"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Genes interroges", n)
    m2.metric("Avec phenotype HPO", n_with_hpo)
    m3.metric("Avec maladie associee", n_with_disease)

    # ── HPO LE PLUS FRÉQUENTS ──
    st.markdown("### 🩺 Phenotypes HPO les plus frequents (cohorte)")
    st.caption(
        "Compte combien de genes (parmi ceux interroges) sont associes a chaque terme HPO. "
        "Permet d'identifier des phenotypes potentiellement recurrents."
    )

    hpo_counter = Counter()
    hpo_genes = {}  # term -> set of genes
    for gene, info in results.items():
        for ph in info["phenotypes"]:
            label = ph["label"]
            hpo_counter[label] += 1
            hpo_genes.setdefault(label, set()).add(gene)

    if hpo_counter:
        df_hpo = pd.DataFrame([
            {"HPO": term, "N genes": count,
             "Genes": ", ".join(sorted(hpo_genes[term])[:10])
                      + ("..." if len(hpo_genes[term]) > 10 else "")}
            for term, count in hpo_counter.most_common(50)
        ])
        st.dataframe(df_hpo, hide_index=True, width='stretch', height=400)

        # Bar plot top 20
        top20 = df_hpo.head(20).sort_values("N genes")
        fig = px.bar(
            top20, x="N genes", y="HPO", orientation="h",
            color="N genes", color_continuous_scale=["#0a192f", "#64ffda"],
            title="Top 20 phenotypes HPO (par nombre de genes associes)",
        )
        fig.update_layout(template="plotly_dark", height=600,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Aucun phenotype HPO retourne.")

    # ── MALADIES ──
    st.markdown("### 🏥 Maladies associees (cohorte)")
    disease_counter = Counter()
    disease_genes = {}
    for gene, info in results.items():
        for d in info["diseases"]:
            label = d["label"]
            disease_counter[label] += 1
            disease_genes.setdefault(label, set()).add(gene)

    if disease_counter:
        df_dis = pd.DataFrame([
            {"Maladie": d, "N genes": c,
             "Genes": ", ".join(sorted(disease_genes[d])[:10])
                      + ("..." if len(disease_genes[d]) > 10 else "")}
            for d, c in disease_counter.most_common(50)
        ])
        st.dataframe(df_dis, hide_index=True, width='stretch', height=400)
    else:
        st.info("Aucune maladie associee retournee.")

    # ── VUE PAR GÈNE ──
    st.markdown("### 🔎 Detail par gene")
    gene_pick = st.selectbox(
        "Gene",
        [g for g in target_genes if g in results],
        index=0 if target_genes else None,
    )
    if gene_pick:
        info = results[gene_pick]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Phenotypes HPO ({len(info['phenotypes'])})**")
            if info["phenotypes"]:
                st.dataframe(
                    pd.DataFrame(info["phenotypes"]).rename(columns={
                        "hpo_id": "HPO ID", "label": "Phenotype"
                    }),
                    hide_index=True, width='stretch', height=400,
                )
            else:
                st.caption("Aucun.")
        with c2:
            st.markdown(f"**Maladies ({len(info['diseases'])})**")
            if info["diseases"]:
                st.dataframe(
                    pd.DataFrame(info["diseases"]).rename(columns={
                        "id": "ID", "label": "Maladie"
                    }),
                    hide_index=True, width='stretch', height=400,
                )
            else:
                st.caption("Aucune.")

        if info.get("hgnc_id"):
            st.caption(f"ID Monarch : `{info['hgnc_id']}`")

    # ── EXPORT ──
    st.markdown("### 📥 Export complet")
    rows_export = []
    for gene, info in results.items():
        for ph in info["phenotypes"]:
            rows_export.append({
                "Gene": gene, "Type": "HPO",
                "ID": ph["hpo_id"], "Label": ph["label"],
            })
        for d in info["diseases"]:
            rows_export.append({
                "Gene": gene, "Type": "Disease",
                "ID": d["id"], "Label": d["label"],
            })

    if rows_export:
        df_export = pd.DataFrame(rows_export)
        st.download_button(
            "📥 Exporter HPO + maladies (CSV)",
            df_export.to_csv(index=False, sep=";"),
            "hpo_diseases.csv", "text/csv",
            width='stretch',
        )
