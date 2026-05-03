"""Onglet : Complications"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import fisher_exact

from core.reports import generate_complications_report
from core.config import NON_FUNCTIONAL_EFFECTS


# ═══════════════════════════════════════════════════════════════════
# FONCTIONS DE CALCUL (au niveau module + cache pour eviter le recalcul
# a chaque interaction Streamlit)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_patient_clinical(df_f_hash_key, df_f_pickle):
    """
    Construit la matrice patient × variables cliniques.
    Le hash_key force l'invalidation quand df_f change ; df_f_pickle est
    le DataFrame serialise (Streamlit hash automatiquement).
    """
    df_f = df_f_pickle
    COMPLICATION_COLS = ["BO", "PNP", "MG", "FDSCS"]
    CLINICAL_ALL_COLS = ["Complication", "Chirurgie", "Recidive", "BO", "PNP", "MG",
                         "FDSCS", "Histo UCD", "Auto Ac"]
    avail_compl = [c for c in COMPLICATION_COLS if c in df_f.columns]
    avail_clin_all = [c for c in CLINICAL_ALL_COLS if c in df_f.columns]

    patient_clinical = {}
    for pseudo in df_f["Pseudo"].unique():
        dp = df_f[df_f["Pseudo"] == pseudo]
        pat_data = {}
        has_any_clinical = False
        for col in avail_clin_all:
            vals = dp[col].dropna().unique()
            if len(vals) > 0:
                pat_data[col] = vals[0]
                has_any_clinical = True
            else:
                pat_data[col] = None

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

    return pd.DataFrame.from_dict(patient_clinical, orient="index"), avail_compl, avail_clin_all


def _run_association_core(df_variants, df_clinical, entity_col, min_carriers,
                          filter_acmg, correction_method):
    """Coeur du test d'association (sans cache - appele par la version cachee)."""
    df_test = df_variants.copy()
    if filter_acmg is not None:
        df_test = df_test[df_test["ACMG_class"].isin(filter_acmg)]

    entity_carrier_count = df_test.groupby(entity_col)["Pseudo"].nunique()
    frequent_entities = entity_carrier_count[entity_carrier_count >= min_carriers].index.tolist()

    results = []
    patients = df_clinical.index.tolist()
    compl_set = set(df_clinical[df_clinical["Complication_any"] == 1].index)

    for ent in frequent_entities:
        carriers = set(df_test[df_test[entity_col] == ent]["Pseudo"].unique()) & set(patients)
        non_carriers = set(patients) - carriers

        a = len(carriers & compl_set)
        b = len(carriers - compl_set)
        c = len(non_carriers & compl_set)
        d = len(non_carriers - compl_set)

        if a + b == 0 or c + d == 0:
            continue

        table = [[a, b], [c, d]]
        try:
            or_val, p_val = fisher_exact(table, alternative="two-sided")
        except Exception:
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
    n_tests = len(df_res)

    if correction_method == "Bonferroni":
        df_res["P_adjusted"] = (df_res["P_value"] * n_tests).clip(upper=1.0)
    elif correction_method == "FDR (Benjamini-Hochberg)":
        df_res_sorted = df_res.sort_values("P_value").reset_index(drop=True)
        df_res_sorted["rank"] = df_res_sorted.index + 1
        df_res_sorted["P_adjusted"] = (df_res_sorted["P_value"] * n_tests / df_res_sorted["rank"]).clip(upper=1.0)
        p_adj = np.array(df_res_sorted["P_adjusted"].values, copy=True)
        for i in range(len(p_adj) - 2, -1, -1):
            p_adj[i] = min(p_adj[i], p_adj[i + 1])
        df_res_sorted["P_adjusted"] = p_adj
        df_res_sorted = df_res_sorted.drop(columns="rank")
        df_res = df_res_sorted.sort_values("P_value").reset_index(drop=True)
    else:
        df_res["P_adjusted"] = df_res["P_value"]

    return df_res


@st.cache_data(show_spinner=False)
def _run_association_cached(df_variants, df_clinical, entity_col, min_carriers,
                             filter_acmg_tuple, correction_method):
    """Version cachee : evite de refaire les tests Fisher a chaque rerun."""
    filter_acmg = list(filter_acmg_tuple) if filter_acmg_tuple else None
    return _run_association_core(df_variants, df_clinical, entity_col,
                                  min_carriers, filter_acmg, correction_method)


def render(df_f, df, pathways_dict, api_key):
    st.markdown("## 🎯 Analyse des complications")
    st.markdown(
        "Analyse **supervisée** : existe-t-il une signature génomique particulière associée à "
        "l'apparition d'une complication (BO, PNP, MG, ou FDSCS) ? "
        "L'analyse se fait à deux niveaux : par **variant** (peu de signal attendu) et par **gène muté**."
    )

    # ── DÉFINITION DU GROUPE "COMPLIQUÉ" (cache) ──
    # On hash sur les colonnes/patients pour invalider quand df_f change
    cache_key = f"{len(df_f)}_{df_f['Pseudo'].nunique()}_{','.join(sorted(df_f.columns)[:20])}"
    df_pat_clin, avail_compl, avail_clin_all = _build_patient_clinical(cache_key, df_f)

    if len(avail_compl) == 0:
        st.error("Aucune colonne de complication trouvée (BO, PNP, MG, FDSCS).")
        st.stop()

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
        st.plotly_chart(fig, width='stretch')

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
        st.plotly_chart(fig, width='stretch')

    # ── VALIDATION PRÉ-ANALYSE ──
    if n_elig < 5:
        st.error("Moins de 5 patients éligibles. Analyse impossible.")
        st.stop()
    if n_compl == 0 or n_no_compl == 0:
        st.error("Tous les patients sont dans le même groupe — pas de comparaison possible.")
        st.stop()
    if n_compl < 3 or n_no_compl < 3:
        st.warning("⚠️ Très peu de patients dans un des groupes. La puissance statistique sera faible.")

    # ── WRAPPER VERS LA FONCTION CACHEE (definie au niveau module) ──
    def run_association_test(df_variants, df_clinical, entity_col, min_carriers, filter_acmg=None):
        """Wrapper : delegue a la fonction cachee au niveau module."""
        filter_tuple = tuple(sorted(filter_acmg)) if filter_acmg else None
        return _run_association_cached(
            df_variants, df_clinical, entity_col, min_carriers,
            filter_tuple, correction_method
        )


    # ═════════════════════════════════════════════════════
    # LANCEMENT DE L'ANALYSE
    # ═════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🧪 Résultats de l'analyse d'association")

    # Deux sous-onglets pour les 2 niveaux
    lvl_var, lvl_gene, lvl_pw, lvl_sig, lvl_vaf_loh = st.tabs(
        ["📍 Niveau variant", "🧬 Niveau gène (Patho/LP/VUS)",
         "🔬 Niveau pathway", "🔥 Signatures par complication", 
         "🧬 Profil VAF & LOH"]
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
            st.plotly_chart(fig, width='stretch')

            # TOP HITS TABLE
            st.markdown("### 🔝 Top variants associés")
            display_cols = ["Gène", "Entity", "hgvs.p", "Variant_effect", "ACMG_class",
                           "N_carriers", "Carriers_with_compl", "Carriers_without_compl",
                           "Freq_compl_carriers", "Freq_compl_non_carriers",
                           "Odds_Ratio", "P_value", "P_adjusted", "Direction"]
            display_cols = [c for c in display_cols if c in df_res_var.columns]
            st.dataframe(
                df_res_var[display_cols].head(30).reset_index(drop=True),
                width='stretch',
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
            st.plotly_chart(fig, width='stretch')

            # TOP HITS TABLE
            st.markdown("### 🔝 Top gènes associés")
            display_cols = ["Entity", "N_carriers", "Carriers_with_compl", "Carriers_without_compl",
                           "Freq_compl_carriers", "Freq_compl_non_carriers",
                           "Odds_Ratio", "P_value", "P_adjusted", "Direction"]
            st.dataframe(
                df_res_gene[display_cols].head(30).reset_index(drop=True),
                width='stretch',
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
            st.plotly_chart(fig, width='stretch')

            # Afficher le statut complication en-dessous
            status_df = pd.DataFrame({
                "Patient": oncomat.index,
                "Complication": ["🔴 Oui" if df_pat_clin.loc[p, "Complication_any"] == 1 else "⚪ Non"
                                 for p in oncomat.index],
                "N gènes mutés (Patho/LP/VUS)": oncomat.sum(axis=1).values,
            })
            with st.expander("📋 Statut détaillé des patients"):
                st.dataframe(status_df, width='stretch', hide_index=True)

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
                st.plotly_chart(fig, width='stretch')

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
                    st.plotly_chart(fig, width='stretch')

                    # TOP HITS TABLE
                    st.markdown("### 🔝 Top pathways associés")
                    display_cols_pw = ["Pathway", "N_genes_pw", "Muted_genes_in_cohort",
                                       "N_carriers", "Carriers_with_compl", "Carriers_without_compl",
                                       "Freq_compl_carriers", "Freq_compl_non_carriers",
                                       "Odds_Ratio", "P_value", "P_adjusted", "Direction"]
                    display_cols_pw = [c for c in display_cols_pw if c in df_res_pw.columns]
                    st.dataframe(
                        df_res_pw[display_cols_pw].head(30).reset_index(drop=True),
                        width='stretch',
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
                            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='content')

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

                st.plotly_chart(fig, width='stretch')

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
                st.plotly_chart(fig, width='stretch')

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
                st.dataframe(df_recap.reset_index(drop=True), width='stretch',
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


    # ─── PROFIL VAF & LOH ───
    with lvl_vaf_loh:
        st.markdown("### 🧬 Profil VAF, LOH & TMB")
        st.markdown(
            "Analyse détaillée des **fréquences alléliques (VAF)**, détection de **perte d'hétérozygotie (LOH)** "
            "et calcul du **Tumor Mutational Burden (TMB)** pour chaque patient avec complications."
        )
        
        # Info TMB
        with st.expander("ℹ️ Comprendre les métriques"):
            st.markdown("""
            **VAF (Variant Allele Frequency)** : Proportion de reads portant le variant.
            - VAF ≈ 0.5 : hétérozygote
            - VAF ≈ 1.0 : homozygote ou perte d'hétérozygotie (LOH)
            - VAF < 0.25 : sous-clonal
            
            **LOH (Loss of Heterozygosity)** : Perte d'un allèle, détectée par VAF élevée (≥0.9).
            
            **TMB (Tumor Mutational Burden)** : Score composite = nombre_variants × VAF_moyenne.
            Reflète la charge mutationnelle globale pondérée par la fréquence allélique.
            """)
        
        # ── PARAMÈTRES ──
        st.markdown("#### ⚙️ Paramètres de détection")
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            vaf_loh_threshold = st.slider(
                "Seuil VAF pour LOH suspectée",
                min_value=0.7, max_value=1.0, value=0.9, step=0.05,
                help="Variants avec VAF ≥ ce seuil sont considérés comme potentiellement homozygotes/LOH"
            )
        
        with col_p2:
            vaf_clonal_threshold = st.slider(
                "Seuil VAF clonal",
                min_value=0.15, max_value=0.5, value=0.25, step=0.05,
                help="Variants avec VAF ≥ ce seuil sont considérés comme clonaux"
            )
        
        with col_p3:
            min_depth_loh = st.number_input(
                "Profondeur minimale pour LOH",
                min_value=50, max_value=500, value=150, step=50,
                help="Profondeur de séquençage minimale pour avoir confiance en la détection de LOH"
            )
        
        # ── FILTRAGE DES DONNÉES ──
        # Utiliser df_c (déjà filtré) pour l'analyse
        df_vaf_analysis = df_c.copy()
        
        # Ajouter les statuts VAF
        df_vaf_analysis['VAF_status'] = pd.cut(
            df_vaf_analysis['Allelic_ratio'],
            bins=[0, 0.1, vaf_clonal_threshold, vaf_loh_threshold, 1.0],
            labels=['Mineur', 'Sous-clonal', 'Clonal', 'LOH suspectée'],
            include_lowest=True
        )
        
        # Identifier les variants LOH
        df_vaf_analysis['Is_LOH'] = (
            (df_vaf_analysis['Allelic_ratio'] >= vaf_loh_threshold) &
            (df_vaf_analysis['Depth'] >= min_depth_loh)
        )
        
        # ── CALCULS PAR PATIENT ──
        patient_vaf_stats = []
        
        for pseudo in eligible_patients:
            dp = df_vaf_analysis[df_vaf_analysis['Pseudo'] == pseudo]
            
            if len(dp) == 0:
                continue
            
            # Statut complication
            has_compl = df_pat_elig.loc[pseudo, 'Complication_any'] == 1 if pseudo in df_pat_elig.index else False
            compl_types = []
            if has_compl:
                for compl_type in avail_compl:
                    try:
                        if df_pat_clin.loc[pseudo, compl_type] == 1:
                            compl_types.append(compl_type)
                    except:
                        pass
            
            # Statistiques VAF
            vafs = dp['Allelic_ratio']
            
            # Comptage LOH
            n_loh = dp['Is_LOH'].sum()
            pct_loh = (n_loh / len(dp) * 100) if len(dp) > 0 else 0
            
            # TMB
            tmb = len(dp) * vafs.mean()
            
            # Statistiques par statut
            status_counts = dp['VAF_status'].value_counts()
            
            patient_vaf_stats.append({
                'Patient': pseudo,
                'Has_complication': has_compl,
                'Complication_types': ', '.join(compl_types) if compl_types else 'Aucune',
                'N_variants': len(dp),
                'VAF_median': vafs.median(),
                'VAF_mean': vafs.mean(),
                'VAF_max': vafs.max(),
                'VAF_IQR': vafs.quantile(0.75) - vafs.quantile(0.25),
                'N_LOH': int(n_loh),
                'Pct_LOH': pct_loh,
                'N_mineur': int(status_counts.get('Mineur', 0)),
                'N_subclonal': int(status_counts.get('Sous-clonal', 0)),
                'N_clonal': int(status_counts.get('Clonal', 0)),
                'Pct_clonal': (status_counts.get('Clonal', 0) / len(dp) * 100) if len(dp) > 0 else 0,
                'TMB_score': tmb,
            })
        
        df_patient_vaf = pd.DataFrame(patient_vaf_stats)
        
        if len(df_patient_vaf) == 0:
            st.warning("Aucun patient avec variants après filtrage.")
        else:
            # Séparer compliqués vs non-compliqués
            df_compl = df_patient_vaf[df_patient_vaf['Has_complication']]
            df_no_compl = df_patient_vaf[~df_patient_vaf['Has_complication']]
            
            # ══════════════════════════════════════════
            # 1. VUE D'ENSEMBLE : MÉTRIQUES
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📊 Vue d'ensemble")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Patients avec complications",
                    len(df_compl),
                    f"{len(df_compl)/len(df_patient_vaf)*100:.1f}%"
                )
            
            with col_m2:
                tmb_compl_mean = df_compl['TMB_score'].mean() if len(df_compl) > 0 else 0
                tmb_no_compl_mean = df_no_compl['TMB_score'].mean() if len(df_no_compl) > 0 else 0
                delta_tmb = tmb_compl_mean - tmb_no_compl_mean
                st.metric(
                    "TMB moyen (compliqués)",
                    f"{tmb_compl_mean:.1f}",
                    f"{delta_tmb:+.1f} vs non-compl."
                )
            
            with col_m3:
                loh_compl_mean = df_compl['N_LOH'].mean() if len(df_compl) > 0 else 0
                loh_no_compl_mean = df_no_compl['N_LOH'].mean() if len(df_no_compl) > 0 else 0
                delta_loh = loh_compl_mean - loh_no_compl_mean
                st.metric(
                    "LOH moyen (compliqués)",
                    f"{loh_compl_mean:.1f}",
                    f"{delta_loh:+.1f} vs non-compl."
                )
            
            with col_m4:
                vaf_compl_median = df_compl['VAF_median'].median() if len(df_compl) > 0 else 0
                vaf_no_compl_median = df_no_compl['VAF_median'].median() if len(df_no_compl) > 0 else 0
                delta_vaf = vaf_compl_median - vaf_no_compl_median
                st.metric(
                    "VAF médiane (compliqués)",
                    f"{vaf_compl_median:.3f}",
                    f"{delta_vaf:+.3f} vs non-compl."
                )
            
            # ══════════════════════════════════════════
            # 2. DISTRIBUTION DES VAF
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📈 1. Distribution des VAF")
            
            vaf_view_option = st.radio(
                "Niveau de visualisation",
                ["Par patient", "Par type de complication", "Comparaison globale"],
                horizontal=True,
                key="vaf_view"
            )
            
            if vaf_view_option == "Par patient":
                st.markdown("#### Distribution VAF par patient (compliqués uniquement)")
                
                if len(df_compl) == 0:
                    st.info("Aucun patient compliqué avec variants.")
                else:
                    # Sélection patient
                    selected_patient = st.selectbox(
                        "Sélectionner un patient",
                        df_compl['Patient'].tolist(),
                        key="vaf_patient_select"
                    )
                    
                    # Données du patient
                    dp_selected = df_vaf_analysis[df_vaf_analysis['Pseudo'] == selected_patient]
                    patient_info = df_compl[df_compl['Patient'] == selected_patient].iloc[0]
                    
                    # Infos patient
                    st.markdown(f"**Patient {selected_patient}** — Complications : {patient_info['Complication_types']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Variants", patient_info['N_variants'])
                    c2.metric("VAF médiane", f"{patient_info['VAF_median']:.3f}")
                    c3.metric("LOH détectées", int(patient_info['N_LOH']))
                    c4.metric("TMB", f"{patient_info['TMB_score']:.1f}")
                    
                    # Histogramme VAF
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=dp_selected['Allelic_ratio'],
                        nbinsx=50,
                        marker_color='#64ffda',
                        opacity=0.7,
                        name='Distribution VAF'
                    ))
                    
                    # Lignes de seuil
                    fig.add_vline(x=vaf_clonal_threshold, line_dash="dash", line_color="#ffa500",
                                  annotation_text=f"Clonal ({vaf_clonal_threshold})")
                    fig.add_vline(x=vaf_loh_threshold, line_dash="dash", line_color="#ff6b6b",
                                  annotation_text=f"LOH ({vaf_loh_threshold})")
                    
                    fig.update_layout(
                        title=f"Distribution VAF — {selected_patient}",
                        xaxis_title="Variant Allele Frequency (VAF)",
                        yaxis_title="Nombre de variants",
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Tableau des variants LOH pour ce patient
                    if patient_info['N_LOH'] > 0:
                        st.markdown("##### Variants suspects de LOH")
                        loh_variants = dp_selected[dp_selected['Is_LOH']][
                            ['Gene_symbol', 'Variant', 'hgvs.p', 'Allelic_ratio', 'Depth', 
                             'ACMG_class', 'Putative_impact']
                        ].sort_values('Allelic_ratio', ascending=False)
                        
                        st.dataframe(
                            loh_variants.reset_index(drop=True),
                            width='stretch',
                            height=min(400, len(loh_variants) * 35 + 38)
                        )
            
            elif vaf_view_option == "Par type de complication":
                st.markdown("#### Distribution VAF par type de complication")
                
                if len(df_compl) == 0:
                    st.info("Aucun patient compliqué avec variants.")
                else:
                    # Box plots par type de complication
                    fig = go.Figure()
                    
                    for compl_type in avail_compl:
                        # Patients avec cette complication
                        patients_with_compl = df_compl[
                            df_compl['Complication_types'].str.contains(compl_type, na=False)
                        ]['Patient'].tolist()
                        
                        if len(patients_with_compl) > 0:
                            # VAFs pour ces patients
                            vafs_compl = df_vaf_analysis[
                                df_vaf_analysis['Pseudo'].isin(patients_with_compl)
                            ]['Allelic_ratio']
                            
                            fig.add_trace(go.Box(
                                y=vafs_compl,
                                name=compl_type,
                                marker_color={'BO': '#ff6b6b', 'PNP': '#ffa500', 
                                            'MG': '#ffd93d', 'FDSCS': '#9b59b6'}.get(compl_type, '#888'),
                                boxmean='sd'
                            ))
                    
                    # Ajouter non-compliqués pour comparaison
                    if len(df_no_compl) > 0:
                        vafs_no_compl = df_vaf_analysis[
                            df_vaf_analysis['Pseudo'].isin(df_no_compl['Patient'])
                        ]['Allelic_ratio']
                        
                        fig.add_trace(go.Box(
                            y=vafs_no_compl,
                            name='Non compliqué',
                            marker_color='#4ecdc4',
                            boxmean='sd'
                        ))
                    
                    fig.update_layout(
                        title="Distribution VAF par type de complication",
                        yaxis_title="Variant Allele Frequency (VAF)",
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Statistiques par complication
                    st.markdown("##### Statistiques VAF par complication")
                    
                    stats_compl = []
                    for compl_type in avail_compl:
                        patients_with = df_compl[
                            df_compl['Complication_types'].str.contains(compl_type, na=False)
                        ]
                        
                        if len(patients_with) > 0:
                            vafs = df_vaf_analysis[
                                df_vaf_analysis['Pseudo'].isin(patients_with['Patient'])
                            ]['Allelic_ratio']
                            
                            stats_compl.append({
                                'Complication': compl_type,
                                'N_patients': len(patients_with),
                                'VAF_median': vafs.median(),
                                'VAF_mean': vafs.mean(),
                                'VAF_std': vafs.std(),
                                'Pct_clonal': (vafs >= vaf_clonal_threshold).sum() / len(vafs) * 100,
                                'Pct_LOH': (vafs >= vaf_loh_threshold).sum() / len(vafs) * 100,
                            })
                    
                    if stats_compl:
                        df_stats_compl = pd.DataFrame(stats_compl)
                        st.dataframe(
                            df_stats_compl.style.format({
                                'VAF_median': '{:.3f}',
                                'VAF_mean': '{:.3f}',
                                'VAF_std': '{:.3f}',
                                'Pct_clonal': '{:.1f}%',
                                'Pct_LOH': '{:.1f}%'
                            }),
                            width='stretch'
                        )
            
            else:  # Comparaison globale
                st.markdown("#### Comparaison VAF : Compliqués vs Non-compliqués")
                
                # Violin plots
                fig = go.Figure()
                
                if len(df_compl) > 0:
                    vafs_compl = df_vaf_analysis[
                        df_vaf_analysis['Pseudo'].isin(df_compl['Patient'])
                    ]['Allelic_ratio']
                    
                    fig.add_trace(go.Violin(
                        y=vafs_compl,
                        name='Avec complication',
                        marker_color='#ff6b6b',
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                if len(df_no_compl) > 0:
                    vafs_no_compl = df_vaf_analysis[
                        df_vaf_analysis['Pseudo'].isin(df_no_compl['Patient'])
                    ]['Allelic_ratio']
                    
                    fig.add_trace(go.Violin(
                        y=vafs_no_compl,
                        name='Sans complication',
                        marker_color='#4ecdc4',
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                fig.update_layout(
                    title="Distribution VAF : Compliqués vs Non-compliqués",
                    yaxis_title="Variant Allele Frequency (VAF)",
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # ══════════════════════════════════════════
            # 3. DÉTECTION LOH
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 🔬 2. Détection de perte d'hétérozygotie (LOH)")
            
            loh_view = st.radio(
                "Vue LOH",
                ["Tableau des LOH", "Heatmap patient × variant", "Distribution par patient"],
                horizontal=True,
                key="loh_view"
            )
            
            # Filtrer variants LOH
            df_loh = df_vaf_analysis[df_vaf_analysis['Is_LOH']].copy()
            
            if len(df_loh) == 0:
                st.info(f"Aucun variant avec LOH suspectée (VAF ≥ {vaf_loh_threshold}, Depth ≥ {min_depth_loh}).")
            else:
                st.markdown(f"**{len(df_loh)} variants** avec LOH suspectée chez **{df_loh['Pseudo'].nunique()} patients**")
                
                if loh_view == "Tableau des LOH":
                    st.markdown("#### Tous les variants avec LOH suspectée")
                    
                    # Ajouter info complication
                    df_loh_display = df_loh.copy()
                    df_loh_display['Has_compl'] = df_loh_display['Pseudo'].map(
                        lambda p: '✓' if p in df_compl['Patient'].values else '✗'
                    )
                    df_loh_display['Compl_types'] = df_loh_display['Pseudo'].map(
                        lambda p: df_compl[df_compl['Patient'] == p]['Complication_types'].values[0] 
                        if p in df_compl['Patient'].values else ''
                    )
                    
                    cols_display = ['Pseudo', 'Has_compl', 'Compl_types', 'Gene_symbol', 'Variant', 
                                   'hgvs.p', 'Allelic_ratio', 'Depth', 'ACMG_class', 'Putative_impact']
                    cols_display = [c for c in cols_display if c in df_loh_display.columns]
                    
                    st.dataframe(
                        df_loh_display[cols_display].sort_values(['Has_compl', 'Allelic_ratio'], 
                                                                  ascending=[False, False]).reset_index(drop=True),
                        width='stretch',
                        height=400,
                        column_config={
                            'Allelic_ratio': st.column_config.ProgressColumn(
                                'VAF',
                                min_value=0,
                                max_value=1,
                                format='%.3f'
                            )
                        }
                    )
                    
                    # Export
                    csv_loh = df_loh_display[cols_display].to_csv(index=False, sep=';')
                    st.download_button(
                        "📥 Télécharger tableau LOH (CSV)",
                        csv_loh,
                        "loh_variants.csv",
                        "text/csv",
                        key="dl_loh_table"
                    )
                
                elif loh_view == "Heatmap patient × variant":
                    st.markdown("#### Heatmap : Patients × Variants LOH (colorée par VAF)")
                    
                    # Limiter aux LOH fréquentes
                    loh_variant_counts = df_loh.groupby('Variant')['Pseudo'].nunique()
                    min_carriers_loh = st.slider(
                        "Nb minimum de patients porteurs",
                        min_value=1, max_value=max(3, df_loh['Pseudo'].nunique() // 2),
                        value=1,
                        key="loh_min_carriers"
                    )
                    
                    freq_loh_variants = loh_variant_counts[loh_variant_counts >= min_carriers_loh].index.tolist()
                    
                    if len(freq_loh_variants) == 0:
                        st.warning(f"Aucun variant LOH présent chez ≥{min_carriers_loh} patients.")
                    else:
                        # Limiter à top N
                        n_show_loh = min(30, len(freq_loh_variants))
                        top_loh_variants = loh_variant_counts.loc[freq_loh_variants].sort_values(ascending=False).head(n_show_loh).index.tolist()
                        
                        # Patients avec au moins une LOH
                        loh_patients = df_loh['Pseudo'].unique().tolist()
                        
                        # Matrice VAF
                        hm_vaf = pd.DataFrame(0.0, index=top_loh_variants, columns=loh_patients)
                        
                        for variant in top_loh_variants:
                            dv = df_loh[df_loh['Variant'] == variant]
                            for _, row in dv.iterrows():
                                if row['Pseudo'] in hm_vaf.columns:
                                    hm_vaf.loc[variant, row['Pseudo']] = row['Allelic_ratio']
                        
                        # Trier colonnes (patients) par complication
                        def compl_sort_loh(p):
                            has_c = 1 if p in df_compl['Patient'].values else 0
                            return (-has_c, p)
                        
                        loh_patients_sorted = sorted(loh_patients, key=compl_sort_loh)
                        hm_vaf = hm_vaf[loh_patients_sorted]
                        
                        # Annoter gènes
                        variant_genes = {}
                        for v in top_loh_variants:
                            genes = df_loh[df_loh['Variant'] == v]['Gene_symbol'].unique()
                            variant_genes[v] = genes[0] if len(genes) > 0 else ''
                        
                        # Créer labels
                        y_labels = [f"{variant_genes.get(v, '')} ({v.split(':')[-1][:20]}...)" 
                                   for v in top_loh_variants]
                        
                        # Figure
                        fig = go.Figure(go.Heatmap(
                            z=hm_vaf.values,
                            x=hm_vaf.columns,
                            y=y_labels,
                            colorscale='RdYlGn',
                            zmin=0, zmax=1,
                            colorbar_title="VAF",
                            hovertemplate="Patient: %{x}<br>Variant: %{y}<br>VAF: %{z:.3f}<extra></extra>"
                        ))
                        
                        fig.update_layout(
                            title=f"Top {n_show_loh} variants LOH par VAF",
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=max(500, n_show_loh * 25 + 100),
                            xaxis_tickangle=-90,
                            margin=dict(l=200, b=120)
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                
                else:  # Distribution par patient
                    st.markdown("#### Nombre de LOH par patient")
                    
                    # Bar chart
                    loh_per_patient = df_loh.groupby('Pseudo').size().reset_index(name='N_LOH')
                    loh_per_patient = loh_per_patient.sort_values('N_LOH', ascending=False)
                    
                    # Ajouter info complication
                    loh_per_patient['Has_compl'] = loh_per_patient['Pseudo'].map(
                        lambda p: 'Avec complication' if p in df_compl['Patient'].values else 'Sans complication'
                    )
                    
                    fig = px.bar(
                        loh_per_patient,
                        x='Pseudo',
                        y='N_LOH',
                        color='Has_compl',
                        color_discrete_map={'Avec complication': '#ff6b6b', 'Sans complication': '#4ecdc4'},
                        title="Nombre de variants LOH par patient",
                        labels={'N_LOH': 'Nombre de LOH', 'Pseudo': 'Patient'}
                    )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400,
                        xaxis_tickangle=-90
                    )
                    
                    st.plotly_chart(fig, width='stretch')
            
            # ══════════════════════════════════════════
            # 4. ANALYSE TMB
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📊 3. Tumor Mutational Burden (TMB)")
            
            tmb_view = st.radio(
                "Vue TMB",
                ["Comparaison Compl. vs Non-compl.", "TMB par type de complication", "Stratification par TMB"],
                horizontal=True,
                key="tmb_view"
            )
            
            if tmb_view == "Comparaison Compl. vs Non-compl.":
                st.markdown("#### TMB : Patients avec vs sans complication")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Box plot
                    fig = go.Figure()
                    
                    if len(df_compl) > 0:
                        fig.add_trace(go.Box(
                            y=df_compl['TMB_score'],
                            name='Avec complication',
                            marker_color='#ff6b6b',
                            boxmean='sd'
                        ))
                    
                    if len(df_no_compl) > 0:
                        fig.add_trace(go.Box(
                            y=df_no_compl['TMB_score'],
                            name='Sans complication',
                            marker_color='#4ecdc4',
                            boxmean='sd'
                        ))
                    
                    fig.update_layout(
                        title="Distribution TMB",
                        yaxis_title="TMB Score",
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    # Statistiques
                    st.markdown("##### Statistiques TMB")
                    
                    stats_tmb = []
                    
                    if len(df_compl) > 0:
                        stats_tmb.append({
                            'Groupe': 'Avec complication',
                            'N': len(df_compl),
                            'TMB_mean': df_compl['TMB_score'].mean(),
                            'TMB_median': df_compl['TMB_score'].median(),
                            'TMB_std': df_compl['TMB_score'].std(),
                            'TMB_min': df_compl['TMB_score'].min(),
                            'TMB_max': df_compl['TMB_score'].max(),
                        })
                    
                    if len(df_no_compl) > 0:
                        stats_tmb.append({
                            'Groupe': 'Sans complication',
                            'N': len(df_no_compl),
                            'TMB_mean': df_no_compl['TMB_score'].mean(),
                            'TMB_median': df_no_compl['TMB_score'].median(),
                            'TMB_std': df_no_compl['TMB_score'].std(),
                            'TMB_min': df_no_compl['TMB_score'].min(),
                            'TMB_max': df_no_compl['TMB_score'].max(),
                        })
                    
                    if stats_tmb:
                        df_stats_tmb = pd.DataFrame(stats_tmb)
                        st.dataframe(
                            df_stats_tmb.style.format({
                                'TMB_mean': '{:.2f}',
                                'TMB_median': '{:.2f}',
                                'TMB_std': '{:.2f}',
                                'TMB_min': '{:.2f}',
                                'TMB_max': '{:.2f}'
                            }),
                            width='stretch'
                        )
                        
                        # Test statistique
                        if len(df_compl) > 0 and len(df_no_compl) > 0:
                            from scipy.stats import mannwhitneyu
                            stat, p_value = mannwhitneyu(
                                df_compl['TMB_score'],
                                df_no_compl['TMB_score'],
                                alternative='two-sided'
                            )
                            
                            st.markdown(f"""
                            **Test de Mann-Whitney U**  
                            - Statistique U : {stat:.2f}  
                            - P-value : {p_value:.4f}  
                            - Significatif (p<0.05) : {'✓ Oui' if p_value < 0.05 else '✗ Non'}
                            """)
            
            elif tmb_view == "TMB par type de complication":
                st.markdown("#### TMB par type de complication")
                
                # Box plots
                fig = go.Figure()
                
                for compl_type in avail_compl:
                    patients_with = df_compl[
                        df_compl['Complication_types'].str.contains(compl_type, na=False)
                    ]
                    
                    if len(patients_with) > 0:
                        fig.add_trace(go.Box(
                            y=patients_with['TMB_score'],
                            name=compl_type,
                            marker_color={'BO': '#ff6b6b', 'PNP': '#ffa500', 
                                        'MG': '#ffd93d', 'FDSCS': '#9b59b6'}.get(compl_type, '#888'),
                            boxmean='sd'
                        ))
                
                # Ajouter non-compliqués
                if len(df_no_compl) > 0:
                    fig.add_trace(go.Box(
                        y=df_no_compl['TMB_score'],
                        name='Non compliqué',
                        marker_color='#4ecdc4',
                        boxmean='sd'
                    ))
                
                fig.update_layout(
                    title="TMB par type de complication",
                    yaxis_title="TMB Score",
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Tableau statistiques
                st.markdown("##### Statistiques par complication")
                
                stats_tmb_compl = []
                for compl_type in avail_compl:
                    patients_with = df_compl[
                        df_compl['Complication_types'].str.contains(compl_type, na=False)
                    ]
                    
                    if len(patients_with) > 0:
                        stats_tmb_compl.append({
                            'Complication': compl_type,
                            'N_patients': len(patients_with),
                            'TMB_mean': patients_with['TMB_score'].mean(),
                            'TMB_median': patients_with['TMB_score'].median(),
                            'TMB_std': patients_with['TMB_score'].std(),
                        })
                
                if stats_tmb_compl:
                    df_stats_tmb_compl = pd.DataFrame(stats_tmb_compl)
                    st.dataframe(
                        df_stats_tmb_compl.style.format({
                            'TMB_mean': '{:.2f}',
                            'TMB_median': '{:.2f}',
                            'TMB_std': '{:.2f}'
                        }),
                        width='stretch'
                    )
            
            else:  # Stratification par TMB
                st.markdown("#### Stratification des patients par niveau de TMB")
                
                # Définir seuils TMB
                tmb_threshold_low = st.slider(
                    "Seuil TMB faible/moyen",
                    min_value=0.0,
                    max_value=df_patient_vaf['TMB_score'].quantile(0.75),
                    value=df_patient_vaf['TMB_score'].quantile(0.33),
                    step=0.5,
                    key="tmb_thresh_low"
                )
                
                tmb_threshold_high = st.slider(
                    "Seuil TMB moyen/élevé",
                    min_value=tmb_threshold_low,
                    max_value=df_patient_vaf['TMB_score'].max(),
                    value=df_patient_vaf['TMB_score'].quantile(0.67),
                    step=0.5,
                    key="tmb_thresh_high"
                )
                
                # Catégoriser
                df_patient_vaf['TMB_category'] = pd.cut(
                    df_patient_vaf['TMB_score'],
                    bins=[0, tmb_threshold_low, tmb_threshold_high, np.inf],
                    labels=['TMB faible', 'TMB moyen', 'TMB élevé'],
                    include_lowest=True
                )
                
                # Tableau croisé TMB × Complication
                cross_tmb = pd.crosstab(
                    df_patient_vaf['TMB_category'],
                    df_patient_vaf['Has_complication'],
                    margins=True,
                    margins_name='Total'
                )
                cross_tmb.columns = ['Sans complication', 'Avec complication', 'Total']
                
                st.markdown("##### Répartition TMB × Complication")
                st.dataframe(cross_tmb, width='stretch')
                
                # Proportions
                st.markdown("##### Proportion de complications par niveau de TMB")
                
                prop_compl_by_tmb = []
                for cat in ['TMB faible', 'TMB moyen', 'TMB élevé']:
                    subset = df_patient_vaf[df_patient_vaf['TMB_category'] == cat]
                    if len(subset) > 0:
                        n_compl = subset['Has_complication'].sum()
                        prop = n_compl / len(subset) * 100
                        prop_compl_by_tmb.append({
                            'Catégorie TMB': cat,
                            'N_patients': len(subset),
                            'N_avec_compl': int(n_compl),
                            'Pct_compl': prop
                        })
                
                if prop_compl_by_tmb:
                    df_prop_tmb = pd.DataFrame(prop_compl_by_tmb)
                    
                    # Bar chart
                    fig = px.bar(
                        df_prop_tmb,
                        x='Catégorie TMB',
                        y='Pct_compl',
                        text='Pct_compl',
                        title="% de patients avec complication par niveau de TMB",
                        labels={'Pct_compl': '% avec complication'},
                        color='Catégorie TMB',
                        color_discrete_map={
                            'TMB faible': '#4ecdc4',
                            'TMB moyen': '#ffd93d',
                            'TMB élevé': '#ff6b6b'
                        }
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    st.dataframe(
                        df_prop_tmb.style.format({'Pct_compl': '{:.1f}%'}),
                        width='stretch'
                    )
            
            # ══════════════════════════════════════════
            # EXPORT
            # ══════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📥 Export des données")
            
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                csv_patient_vaf = df_patient_vaf.to_csv(index=False, sep=';')
                st.download_button(
                    "📥 Stats VAF par patient (CSV)",
                    csv_patient_vaf,
                    "patient_vaf_stats.csv",
                    "text/csv",
                    key="dl_vaf_stats",
                    width='stretch'
                )
            
            with col_e2:
                if len(df_loh) > 0:
                    csv_loh_export = df_loh.to_csv(index=False, sep=';')
                    st.download_button(
                        "📥 Variants LOH (CSV)",
                        csv_loh_export,
                        "loh_variants_all.csv",
                        "text/csv",
                        key="dl_loh_all",
                        width='stretch'
                    )
            
            with col_e3:
                csv_tmb = df_patient_vaf[['Patient', 'Has_complication', 'Complication_types', 
                                          'N_variants', 'TMB_score']].to_csv(index=False, sep=';')
                st.download_button(
                    "📥 TMB par patient (CSV)",
                    csv_tmb,
                    "tmb_scores.csv",
                    "text/csv",
                    key="dl_tmb",
                    width='stretch'
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
        if st.button("📊 Générer rapport PDF", type="primary", width='stretch',
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
                width='stretch', key="dl_compl_pdf")
