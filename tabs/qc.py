"""Onglet : Homogénéité (QC)"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import NON_FUNCTIONAL_EFFECTS


def render(df_f):
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
