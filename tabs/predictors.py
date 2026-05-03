"""
Onglet : Predicteurs in silico

Annotation a la demande des variants via MyVariant.info (dbNSFP, hg19) :
- REVEL (consensus, missense)
- AlphaMissense (DeepMind, missense)
- SpliceAI (impact splice)
- MetaSVM, MetaLR, PolyPhen2, SIFT, CADD-PHRED, ClinVar

Comparaison avec la classification ACMG existante pour identifier les
variants reclassifies potentiels (notamment les VUS qui devraient etre
LP/P selon les predicteurs modernes).
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.api_clients import (
    parse_variant_to_hgvs_g,
    query_myvariant_batch,
    extract_myvariant_field,
    clear_api_cache,
)


# Seuils communs pour interpreter les scores
THRESHOLDS = {
    "REVEL": (0.5, "≥ 0.5 = pathogene probable"),
    "AlphaMissense": (0.564, "≥ 0.564 = pathogene probable"),
    "MetaSVM": (0, "≥ 0 = deletere"),
    "MetaLR": (0.5, "≥ 0.5 = deletere"),
    "PolyPhen2": (0.85, "≥ 0.85 = probably damaging"),
    "SIFT": (0.05, "≤ 0.05 = deletere (sens inverse)"),
    "CADD_API": (20, "≥ 20 = top 1% pathogene"),
    "SpliceAI_max": (0.5, "≥ 0.5 = impact splice probable"),
}


def _build_annotation_dataframe(df_subset, results):
    """Construit un DataFrame avec les colonnes de prediction extraites."""
    rows = []
    for _, var_row in df_subset.iterrows():
        hgvs = var_row["_hgvs_g"]
        if not hgvs:
            continue
        payload = results.get(hgvs, {})
        if "_error" in payload:
            continue

        spliceai_scores = [
            extract_myvariant_field(payload, "dbnsfp", "spliceai", k)
            for k in ("ds_ag", "ds_al", "ds_dg", "ds_dl")
        ]
        spliceai_max = None
        nums = [s for s in spliceai_scores if isinstance(s, (int, float))]
        if nums:
            spliceai_max = max(nums)

        clinvar_sig = extract_myvariant_field(payload, "clinvar", "rcv", "clinical_significance")

        rows.append({
            "Pseudo": var_row.get("Pseudo"),
            "Gene": var_row.get("Gene_symbol"),
            "Variant": var_row.get("Variant"),
            "hgvs.c": var_row.get("hgvs.c"),
            "hgvs.p": var_row.get("hgvs.p"),
            "Effect": var_row.get("Variant_effect"),
            "ACMG_existing": var_row.get("ACMG_class"),
            "CADD_local": var_row.get("CADD_phred"),
            "REVEL": extract_myvariant_field(payload, "dbnsfp", "revel", "score"),
            "AlphaMissense": extract_myvariant_field(payload, "dbnsfp", "alphamissense", "score"),
            "AlphaMissense_pred": extract_myvariant_field(payload, "dbnsfp", "alphamissense", "pred"),
            "MetaSVM": extract_myvariant_field(payload, "dbnsfp", "metasvm", "score"),
            "MetaSVM_pred": extract_myvariant_field(payload, "dbnsfp", "metasvm", "pred"),
            "MetaLR": extract_myvariant_field(payload, "dbnsfp", "metalr", "score"),
            "MetaLR_pred": extract_myvariant_field(payload, "dbnsfp", "metalr", "pred"),
            "PolyPhen2": extract_myvariant_field(payload, "dbnsfp", "polyphen2", "hdiv", "score"),
            "PolyPhen2_pred": extract_myvariant_field(payload, "dbnsfp", "polyphen2", "hdiv", "pred"),
            "SIFT": extract_myvariant_field(payload, "dbnsfp", "sift", "score"),
            "SIFT_pred": extract_myvariant_field(payload, "dbnsfp", "sift", "pred"),
            "CADD_API": extract_myvariant_field(payload, "dbnsfp", "cadd", "phred"),
            "SpliceAI_max": spliceai_max,
            "ClinVar": clinvar_sig,
            "_hgvs_g": hgvs,
        })
    return pd.DataFrame(rows)


def _interpret(score, predictor):
    """Renvoie un libelle 'pathogene'/'benin'/'?' selon les seuils."""
    if score is None or pd.isna(score):
        return None
    threshold, _ = THRESHOLDS.get(predictor, (None, ""))
    if threshold is None:
        return None
    if predictor == "SIFT":
        return "deletere" if score <= threshold else "tolere"
    return "pathogene" if score >= threshold else "benin"


def render(df_f, df, pathways_dict, api_key):
    st.markdown("## 🔮 Predicteurs in silico")
    st.markdown(
        "Enrichit les variants avec **REVEL**, **AlphaMissense**, **SpliceAI**, **MetaSVM**, "
        "**MetaLR**, **PolyPhen2**, **SIFT**, **CADD** et **ClinVar** via "
        "[MyVariant.info](https://myvariant.info) (assemblage **GRCh37/hg19**, "
        "source dbNSFP). Les resultats sont mis en cache localement."
    )
    st.caption(
        "ℹ️ Seuls les SNVs (substitutions simples) sont annotes pour le moment. "
        "Indels et variants complexes sont ignores."
    )

    # ── PRÉPARATION DES IDS ──
    df_work = df_f.copy()
    df_work["_hgvs_g"] = df_work["Variant"].apply(parse_variant_to_hgvs_g)
    df_snv = df_work[df_work["_hgvs_g"].notna()]

    n_total = len(df_work)
    n_snv = len(df_snv)
    n_unique = df_snv["_hgvs_g"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Variants total", n_total)
    c2.metric("SNVs annotables", n_snv, delta=f"{n_snv - n_total}" if n_snv < n_total else None)
    c3.metric("HGVS uniques", n_unique)

    if n_snv == 0:
        st.warning("Aucun variant annotable (format SNV `chr:pos:ref>alt` requis).")
        return

    # ── ACTIONS ──
    cols = st.columns([2, 1, 1])
    with cols[0]:
        run = st.button(
            f"🚀 Annoter {n_unique} variants uniques via MyVariant.info",
            type="primary", width='stretch'
        )
    with cols[1]:
        if st.button("🗑️ Vider le cache", width='stretch'):
            clear_api_cache("myvariant_hg19")
            st.success("Cache MyVariant vide.")
    with cols[2]:
        st.caption("Assembly : **hg19**")

    if run:
        progress = st.progress(0.0, text="Interrogation MyVariant.info...")
        unique_ids = df_snv["_hgvs_g"].unique().tolist()
        results = query_myvariant_batch(
            unique_ids, assembly="hg19",
            progress_callback=lambda p: progress.progress(p, text=f"MyVariant.info — {int(p*100)}%"),
        )
        progress.empty()

        n_found = sum(1 for v in results.values() if v and "_error" not in v)
        n_err = sum(1 for v in results.values() if "_error" in v)
        st.success(
            f"✅ {n_found} / {len(unique_ids)} variants annotes."
            + (f" ⚠️ {n_err} erreurs reseau." if n_err else "")
        )
        st.session_state["predictors_results"] = results

    # ── AFFICHAGE ──
    if "predictors_results" not in st.session_state:
        st.info("Cliquez sur **Annoter** pour lancer l'enrichissement.")
        return

    results = st.session_state["predictors_results"]
    df_ann = _build_annotation_dataframe(df_snv, results)

    if len(df_ann) == 0:
        st.warning("Aucune annotation exploitable retournee.")
        return

    # ── SYNTHÈSE PAR PRÉDICTEUR ──
    st.markdown("### 📊 Couverture par predicteur")
    cov_data = []
    for col in ["REVEL", "AlphaMissense", "MetaSVM", "MetaLR", "PolyPhen2",
                "SIFT", "CADD_API", "SpliceAI_max"]:
        n = df_ann[col].notna().sum()
        cov_data.append({"Predicteur": col, "N": n,
                         "%": f"{100*n/len(df_ann):.0f}%"})
    df_cov = pd.DataFrame(cov_data)
    st.dataframe(df_cov, hide_index=True, width='stretch')

    # ── DISTRIBUTIONS ──
    st.markdown("### 📈 Distributions des scores")
    selected = st.multiselect(
        "Predicteurs a afficher",
        ["REVEL", "AlphaMissense", "MetaSVM", "MetaLR",
         "PolyPhen2", "SIFT", "CADD_API", "SpliceAI_max"],
        default=["REVEL", "AlphaMissense", "SpliceAI_max"]
    )

    if selected:
        cols = st.columns(min(len(selected), 3))
        for i, pred in enumerate(selected):
            data = df_ann[pred].dropna()
            if len(data) == 0:
                continue
            with cols[i % 3]:
                fig = px.histogram(
                    data, nbins=30, title=f"{pred} (n={len(data)})",
                    color_discrete_sequence=["#64ffda"],
                )
                threshold, descr = THRESHOLDS.get(pred, (None, ""))
                if threshold is not None:
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#ff6b6b",
                                   annotation_text=f"seuil")
                fig.update_layout(template="plotly_dark", height=280,
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width='stretch')
                if descr:
                    st.caption(descr)

    # ── DISCORDANCES VUS ──
    st.markdown("### 🚨 VUS reclassifiables")
    st.markdown(
        "Variants actuellement classes **VUS** mais que les predicteurs modernes "
        "(REVEL, AlphaMissense) flaggent comme pathogenes."
    )

    df_vus = df_ann[df_ann["ACMG_existing"] == "VUS"].copy()
    df_vus["REVEL_path"] = df_vus["REVEL"].apply(
        lambda x: x is not None and not pd.isna(x) and x >= 0.5
    )
    df_vus["AM_path"] = df_vus["AlphaMissense"].apply(
        lambda x: x is not None and not pd.isna(x) and x >= 0.564
    )
    df_vus["concordant"] = df_vus["REVEL_path"] & df_vus["AM_path"]

    df_reclass = df_vus[df_vus["concordant"]].sort_values(
        ["REVEL", "AlphaMissense"], ascending=[False, False]
    )

    if len(df_reclass) > 0:
        st.warning(f"⚠️ **{len(df_reclass)} VUS** flagges pathogenes par REVEL ET AlphaMissense.")
        cols_show = ["Pseudo", "Gene", "hgvs.c", "hgvs.p", "Effect",
                     "REVEL", "AlphaMissense", "MetaSVM_pred", "ClinVar"]
        cols_show = [c for c in cols_show if c in df_reclass.columns]
        st.dataframe(
            df_reclass[cols_show].head(50).reset_index(drop=True),
            width='stretch', height=400
        )
    else:
        st.success("Aucun VUS flagge pathogene par concordance REVEL+AlphaMissense.")

    # ── SPLICE ──
    st.markdown("### 🧬 Variants avec impact splice probable")
    df_splice = df_ann[df_ann["SpliceAI_max"].notna() &
                       (df_ann["SpliceAI_max"] >= 0.5)].copy()
    df_splice = df_splice.sort_values("SpliceAI_max", ascending=False)

    if len(df_splice) > 0:
        st.warning(f"⚠️ **{len(df_splice)} variants** avec SpliceAI ≥ 0.5 (impact splice probable).")
        cols_show = ["Pseudo", "Gene", "hgvs.c", "Effect", "ACMG_existing",
                     "SpliceAI_max", "ClinVar"]
        cols_show = [c for c in cols_show if c in df_splice.columns]
        st.dataframe(
            df_splice[cols_show].head(50).reset_index(drop=True),
            width='stretch', height=300
        )
    else:
        st.success("Aucun variant avec impact splice probable (SpliceAI < 0.5 partout).")

    # ── TABLEAU COMPLET + EXPORT ──
    st.markdown("### 🗃️ Annotations completes")
    cols_full = [c for c in df_ann.columns if c != "_hgvs_g"]
    st.dataframe(df_ann[cols_full], width='stretch', height=500)

    st.download_button(
        "📥 Exporter annotations (CSV)",
        df_ann[cols_full].to_csv(index=False, sep=";"),
        "predictors_annotations.csv", "text/csv",
        width='stretch'
    )

    # ── INTÉGRATION ACMG vs PRÉDICTEURS ──
    st.markdown("### 🔍 Concordance ACMG existant vs predicteurs")
    df_with_revel = df_ann[df_ann["REVEL"].notna()].copy()
    if len(df_with_revel) > 0:
        df_with_revel["REVEL_class"] = df_with_revel["REVEL"].apply(
            lambda x: "pathogene" if x >= 0.5 else "benin"
        )
        ct = pd.crosstab(df_with_revel["ACMG_existing"], df_with_revel["REVEL_class"])
        fig = px.imshow(ct, color_continuous_scale=["#0a192f", "#64ffda"],
                         labels=dict(x="REVEL", y="ACMG existant", color="N"),
                         text_auto=True, aspect="auto")
        fig.update_layout(template="plotly_dark", height=350,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          title="Matrice de concordance ACMG existant × REVEL")
        st.plotly_chart(fig, width='stretch')
