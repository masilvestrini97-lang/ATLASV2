"""
Onglet : Clustering patient.

Pipeline : build_patient_features -> standardisation -> KMeans/Agglo
-> UMAP 2D -> interpretation -> signatures genes -> rapport PDF
+ option d'interpretation IA via Anthropic.

Stocke les labels dans st.session_state["cluster_labels_map"]
pour reutilisation par d'autres onglets (Reseau STRING, mode cluster).
"""
import pandas as pd
import plotly.express as px
import streamlit as st

# Note : sklearn et umap sont importes a l'interieur de render() (lazy loading)
# pour eviter de ralentir le demarrage de l'app de ~10s (numba/llvmlite).

from core.config import FEATURE_LABELS, NON_FUNCTIONAL_EFFECTS
from core.features import build_patient_features
from core.clustering import compute_cluster_interpretation, get_gene_signature_per_cluster
from core.ai import build_ai_prompt, call_anthropic_api, ANTHROPIC_AVAILABLE
from core.reports import generate_cluster_report


def render(df_f, df, pathways_dict, api_key):
    # ── LAZY IMPORTS (ne se chargent qu'a l'ouverture de cet onglet) ──
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    import umap

    st.markdown("## 🧮 Clustering des patients")
    st.markdown(
        "Construction d'une matrice patient × features (proportions ACMG, scores d'impact, "
        "VAF/clonalite, genes binaires, pathways MSigDB optionnels) puis clustering "
        "non supervise. Les labels obtenus sont stockes pour reutilisation dans "
        "l'onglet **Reseau STRING** (mode cluster)."
    )

    # ── PARAMÈTRES ──
    with st.expander("⚙️ Parametres", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            method = st.selectbox(
                "Methode", ["KMeans", "Agglomerative (Ward)"],
                help="KMeans : centroides. Agglomerative : hierarchique avec lien Ward."
            )
            n_clusters = st.slider("Nombre de clusters (k)", 2, 8, 3)
        with c2:
            use_genomic = st.checkbox("Features genomiques", value=True,
                help="ACMG, impact, VAF, genes binaires.")
            use_pathways = st.checkbox("Features pathways", value=bool(pathways_dict),
                                        disabled=not pathways_dict)
            use_clinical = st.checkbox("Features cliniques", value=False)
        with c3:
            top_n_genes = st.slider("Top N genes", 10, 100, 30, 5)
            exclude_low_vaf = st.checkbox("Exclure VAF mediane basse", value=True,
                help="Retire prealablement les patients avec VAF mediane sous le seuil.")
            vaf_threshold = st.number_input("Seuil VAF mediane", 0.0, 0.5, 0.1, 0.01,
                disabled=not exclude_low_vaf)

    # ── PRÉPARATION ──
    df_clust = df_f[~df_f["Variant_effect"].isin(NON_FUNCTIONAL_EFFECTS)].copy()

    excluded_patients = []
    if exclude_low_vaf:
        vaf_med = df_clust.groupby("Pseudo")["Allelic_ratio"].median()
        excluded_patients = vaf_med[vaf_med < vaf_threshold].index.tolist()
        df_clust = df_clust[~df_clust["Pseudo"].isin(excluded_patients)]

    if df_clust["Pseudo"].nunique() < n_clusters:
        st.error(
            f"❌ Pas assez de patients ({df_clust['Pseudo'].nunique()}) "
            f"pour {n_clusters} clusters."
        )
        return

    if not st.button("🚀 Lancer le clustering", type="primary", width='stretch'):
        st.info("Configurez les parametres puis cliquez sur **Lancer le clustering**.")
        return

    # ── EXÉCUTION ──
    with st.spinner("Construction de la matrice patient × features..."):
        df_feat = build_patient_features(
            df_clust,
            use_genomic=use_genomic,
            use_clinical=use_clinical,
            use_pathways=use_pathways,
            top_n_genes=top_n_genes,
            pathways_dict=pathways_dict if use_pathways else None,
        )

    if len(df_feat) < n_clusters:
        st.error(f"❌ Apres filtrage : {len(df_feat)} patients restants, insuffisant pour {n_clusters} clusters.")
        return

    st.success(f"✅ Matrice : {len(df_feat)} patients × {len(df_feat.columns)} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat.values)

    with st.spinner(f"{method} en cours..."):
        if method == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(X_scaled)

    label_names = [f"Cluster_{i+1}" for i in labels]

    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = float("nan")

    # ── METRIQUES ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients", len(df_feat))
    m2.metric("Features", len(df_feat.columns))
    m3.metric("Clusters", n_clusters)
    m4.metric("Silhouette", f"{sil:.3f}")

    if sil > 0.5:
        st.success("🟢 Bonne separation des clusters.")
    elif sil > 0.3:
        st.warning("🟠 Separation moderee.")
    else:
        st.info("🔵 Faible separation, chevauchement entre clusters.")

    # ── UMAP ──
    st.markdown("### 📍 Projection UMAP 2D")
    with st.spinner("UMAP..."):
        try:
            n_neighbors = min(15, max(2, len(df_feat) - 1))
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                                random_state=42, n_components=2)
            embedding = reducer.fit_transform(X_scaled)
            df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=df_feat.index)
            df_umap["Cluster"] = label_names
            df_umap["Patient"] = df_umap.index

            fig = px.scatter(
                df_umap, x="UMAP1", y="UMAP2", color="Cluster", text="Patient",
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_dark", height=500,
            )
            fig.update_traces(textposition="top center",
                              marker=dict(size=14, line=dict(width=1, color="white")))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.warning(f"UMAP non disponible : {e}")

    # ── INTERPRETATION ──
    st.markdown("### 📋 Composition des clusters")

    key_features = [c for c in df_feat.columns
                    if not c.startswith("gene_") and not c.startswith("genescore_")]

    interpretations = compute_cluster_interpretation(
        df_feat, label_names, key_features, top_n=10
    )

    labels_map = {cid: info["patients"] for cid, info in interpretations.items()}
    gene_sigs = get_gene_signature_per_cluster(df_clust, labels_map)

    # ── STOCKAGE SESSION ──
    st.session_state["cluster_labels_map"] = labels_map
    st.session_state["cluster_method"] = method
    st.session_state["cluster_n"] = n_clusters
    st.session_state["cluster_silhouette"] = sil

    # ── AFFICHAGE PAR CLUSTER ──
    for cid in sorted(interpretations.keys()):
        info = interpretations[cid]
        with st.expander(
            f"**{cid}** — {info['n_patients']} patients : "
            f"{', '.join(sorted(info['patients']))}", expanded=True
        ):
            sig = info["significant"]
            cu, cd = st.columns(2)
            with cu:
                st.markdown("**🔼 Features enrichies (z > 0.5)**")
                up = sig[sig > 0]
                if len(up) == 0:
                    st.caption("Aucune.")
                else:
                    rows = []
                    for f, z in up.head(10).items():
                        rows.append({
                            "Feature": FEATURE_LABELS.get(f, f),
                            "Valeur": f"{info['cluster_mean'][f]:.2f}",
                            "Global": f"{info['global_mean'][f]:.2f}",
                            "z": f"{z:+.2f}",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            with cd:
                st.markdown("**🔽 Features reduites (z < -0.5)**")
                dn = sig[sig < 0]
                if len(dn) == 0:
                    st.caption("Aucune.")
                else:
                    rows = []
                    for f, z in dn.head(10).items():
                        rows.append({
                            "Feature": FEATURE_LABELS.get(f, f),
                            "Valeur": f"{info['cluster_mean'][f]:.2f}",
                            "Global": f"{info['global_mean'][f]:.2f}",
                            "z": f"{z:+.2f}",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

            if cid in gene_sigs:
                gs = gene_sigs[cid]
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.markdown("**🧬 Genes enrichis**")
                    if len(gs["enriched_genes"]) > 0:
                        df_g = gs["enriched_genes"].head(10).reset_index()
                        df_g.columns = ["Gene", "Enrichissement"]
                        df_g["Enrichissement"] = df_g["Enrichissement"].apply(lambda x: f"x{x:.1f}")
                        st.dataframe(df_g, hide_index=True, width='stretch')
                    else:
                        st.caption("Aucun.")
                with gc2:
                    st.markdown("**⚠️ Genes pathogenes**")
                    if len(gs["pathogenic_genes"]) > 0:
                        df_g = gs["pathogenic_genes"].head(10).reset_index()
                        df_g.columns = ["Gene", "N variants"]
                        st.dataframe(df_g, hide_index=True, width='stretch')
                    else:
                        st.caption("Aucun.")

    # ── EXPORTS ──
    st.markdown("### 📥 Exports")
    cexp1, cexp2, cexp3 = st.columns(3)

    with cexp1:
        df_export = df_feat.copy()
        df_export["Cluster"] = label_names
        df_export = df_export.reset_index().rename(columns={"index": "Pseudo"})
        st.download_button(
            "📊 Matrice features + clusters (CSV)",
            df_export.to_csv(index=False, sep=";"),
            "clustering_features.csv", "text/csv",
            width='stretch', key="dl_clust_feat"
        )

    with cexp2:
        df_map = pd.DataFrame([
            {"Pseudo": p, "Cluster": c}
            for c, pats in labels_map.items() for p in pats
        ])
        st.download_button(
            "🗺️ Mapping patient -> cluster (CSV)",
            df_map.to_csv(index=False, sep=";"),
            "clustering_mapping.csv", "text/csv",
            width='stretch', key="dl_clust_map"
        )

    with cexp3:
        if st.button("📄 Generer rapport PDF", width='stretch', key="gen_pdf_clust"):
            with st.spinner("Generation du rapport..."):
                try:
                    pdf_bytes = generate_cluster_report(
                        df_clust, df_feat, label_names, interpretations,
                        gene_sigs, sil, method, key_features,
                        excluded_patients=excluded_patients,
                        pathways_dict=pathways_dict if use_pathways else None,
                    )
                    st.session_state["clust_pdf"] = pdf_bytes
                    st.success("✅ Rapport pret.")
                except Exception as e:
                    st.error(f"Erreur generation PDF : {e}")

        if "clust_pdf" in st.session_state:
            st.download_button(
                "⬇️ Telecharger PDF",
                st.session_state["clust_pdf"],
                "rapport_clustering.pdf", "application/pdf",
                width='stretch', key="dl_clust_pdf"
            )

    # ── INTERPRETATION IA ──
    st.markdown("---")
    st.markdown("### 🤖 Interpretation IA (optionnel)")

    if not ANTHROPIC_AVAILABLE:
        st.info("Le module `anthropic` n'est pas installe.")
    elif not api_key:
        st.info("Renseignez une cle API Anthropic dans la sidebar pour activer.")
    else:
        if st.button("🧠 Demander a l'IA d'interpreter les clusters", key="ai_clust"):
            with st.spinner("Generation du prompt et appel API..."):
                try:
                    prompt = build_ai_prompt(interpretations, gene_sigs, n_clusters)
                    response = call_anthropic_api(prompt, api_key)
                    st.markdown('<div class="ai-interpretation">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur API : {e}")

    if excluded_patients:
        st.markdown("---")
        st.caption(
            f"ℹ️ {len(excluded_patients)} patient(s) exclu(s) (VAF mediane < {vaf_threshold}) : "
            f"{', '.join(sorted(excluded_patients))}"
        )
