"""
Onglet : Reseau STRING

Visualisation d'interactions proteiques pour les genes mutes selon 3 modes :
- Cluster : reutilise les labels stockes dans st.session_state par l'onglet Clustering
- Patient : choix d'un patient unique
- Manuel : selection libre dans la liste des genes du panel
+ enrichissement fonctionnel STRING (GO / KEGG / Reactome).
"""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.api_clients import (
    query_string_network,
    get_string_enrichment,
    clear_api_cache,
)


MAX_GENES = 50  # cap pour lisibilite


def _get_pathogenic_or_lp(df, genes_only=True):
    """Genes uniques ayant au moins un variant Patho/LP, optionnellement enrichis."""
    sub = df[df["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
    if genes_only:
        return sorted(sub["Gene_symbol"].unique())
    return sub


def _draw_network_plotly(network, title=""):
    """Genere un graphe plotly a partir du resultat query_string_network()."""
    nodes = network.get("nodes", [])
    edges = network.get("edges", [])

    if not nodes:
        return None

    # Layout circulaire si peu de noeuds, sinon spring approximatif
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node["id"]] = (math.cos(angle), math.sin(angle))

    # Spring relaxation simple si beaucoup d'aretes
    if edges and n >= 6:
        for _ in range(50):
            forces = {nid: [0.0, 0.0] for nid in positions}
            # Repulsion
            ids = list(positions.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    dx = positions[a][0] - positions[b][0]
                    dy = positions[a][1] - positions[b][1]
                    d2 = dx * dx + dy * dy + 0.01
                    f = 0.05 / d2
                    forces[a][0] += f * dx
                    forces[a][1] += f * dy
                    forces[b][0] -= f * dx
                    forces[b][1] -= f * dy
            # Attraction le long des aretes
            for e in edges:
                a, b = e["source"], e["target"]
                if a in positions and b in positions:
                    dx = positions[b][0] - positions[a][0]
                    dy = positions[b][1] - positions[a][1]
                    f = 0.05 * e.get("score", 0.5)
                    forces[a][0] += f * dx
                    forces[a][1] += f * dy
                    forces[b][0] -= f * dx
                    forces[b][1] -= f * dy
            for nid in positions:
                positions[nid] = (
                    positions[nid][0] + 0.1 * forces[nid][0],
                    positions[nid][1] + 0.1 * forces[nid][1],
                )

    # Aretes
    edge_x, edge_y, edge_widths = [], [], []
    for e in edges:
        a, b = e["source"], e["target"]
        if a in positions and b in positions:
            edge_x.extend([positions[a][0], positions[b][0], None])
            edge_y.extend([positions[a][1], positions[b][1], None])
            edge_widths.append(e.get("score", 0.5))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(100, 255, 218, 0.4)", width=1),
        hoverinfo="none", showlegend=False,
    ))

    node_x = [positions[n["id"]][0] for n in nodes]
    node_y = [positions[n["id"]][1] for n in nodes]
    node_text = [n["label"] for n in nodes]

    # Degre = importance visuelle
    degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    sizes = [15 + 4 * math.sqrt(degree.get(n["id"], 0)) for n in nodes]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=sizes, color="#64ffda",
                    line=dict(width=2, color="#0a192f")),
        text=node_text, textposition="top center",
        textfont=dict(color="white", size=11),
        hovertext=[f"{n['label']}<br>Connexions : {degree.get(n['id'], 0)}" for n in nodes],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render(df_f, df, pathways_dict, api_key):
    st.markdown("## 🕸️ Reseau d'interactions STRING")
    st.markdown(
        "Visualise les interactions proteiques connues parmi les genes mutes "
        "selon [STRING-DB v12](https://string-db.org). 3 modes de selection "
        "des genes : **cluster** (reutilise les labels de l'onglet Clustering), "
        "**patient** (un patient), ou **manuel** (selection libre)."
    )

    # ── PARAMÈTRES COMMUNS ──
    pc1, pc2, pc3 = st.columns([2, 1, 1])
    with pc1:
        score_label = st.select_slider(
            "Confiance STRING minimale",
            options=["low (0.150)", "medium (0.400)", "high (0.700)", "highest (0.900)"],
            value="medium (0.400)",
        )
        score_map = {"low (0.150)": 150, "medium (0.400)": 400,
                     "high (0.700)": 700, "highest (0.900)": 900}
        required_score = score_map[score_label]
    with pc2:
        only_path = st.checkbox(
            "Uniquement Patho/LP", value=True,
            help="Limite aux genes ayant au moins un variant Pathogene ou Likely Pathogenic."
        )
    with pc3:
        if st.button("🗑️ Vider cache STRING", width='stretch'):
            clear_api_cache("string")
            clear_api_cache("string_enrich")
            st.success("Cache STRING vide.")

    # ── MODE ──
    mode = st.radio(
        "**Mode de selection**",
        ["Cluster", "Patient", "Manuel"],
        horizontal=True, label_visibility="collapsed",
    )

    selected_genes = []
    context_label = ""

    if mode == "Cluster":
        labels_map = st.session_state.get("cluster_labels_map")
        if not labels_map:
            st.warning(
                "⚠️ Aucun clustering disponible. Lancez d'abord l'onglet **Clustering** "
                "pour generer les labels."
            )
            return
        cluster_choice = st.selectbox(
            f"Cluster ({len(labels_map)} disponibles)", sorted(labels_map.keys())
        )
        patients = labels_map[cluster_choice]
        st.caption(f"Patients : {', '.join(sorted(patients))}")

        df_sub = df_f[df_f["Pseudo"].isin(patients)]
        if only_path:
            df_sub = df_sub[df_sub["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
        selected_genes = sorted(df_sub["Gene_symbol"].unique())
        context_label = f"{cluster_choice} ({len(patients)} patients)"

    elif mode == "Patient":
        patient = st.selectbox("Patient", sorted(df_f["Pseudo"].unique()))
        df_sub = df_f[df_f["Pseudo"] == patient]
        if only_path:
            df_sub = df_sub[df_sub["ACMG_class"].isin(["Pathogenic", "Likely Pathogenic"])]
        selected_genes = sorted(df_sub["Gene_symbol"].unique())
        context_label = patient

    else:  # Manuel
        if only_path:
            available = _get_pathogenic_or_lp(df_f)
            help_text = f"{len(available)} genes avec variants Patho/LP dans la cohorte filtree."
        else:
            available = sorted(df_f["Gene_symbol"].unique())
            help_text = f"{len(available)} genes au total dans la cohorte filtree."

        selected_genes = st.multiselect(
            "Genes a interroger", available, help=help_text
        )
        context_label = f"selection manuelle ({len(selected_genes)} genes)"

    # ── VALIDATION ──
    n_selected = len(selected_genes)
    st.markdown(f"**{n_selected} genes** selectionnes — *{context_label}*")

    if n_selected == 0:
        st.info("Aucun gene a interroger.")
        return
    if n_selected < 2:
        st.warning("Au moins 2 genes sont necessaires pour visualiser un reseau.")
        return
    if n_selected > MAX_GENES:
        st.warning(
            f"⚠️ Trop de genes ({n_selected} > {MAX_GENES}). "
            f"Le reseau sera limite aux {MAX_GENES} premiers."
        )
        selected_genes = selected_genes[:MAX_GENES]

    if not st.button("🚀 Interroger STRING", type="primary", width='stretch'):
        return

    # ── REQUÊTE ──
    with st.spinner(f"Interrogation STRING ({len(selected_genes)} genes)..."):
        network = query_string_network(selected_genes, required_score=required_score)

    if network.get("_error"):
        st.error(f"❌ {network['_error']}")
        return

    n_nodes = len(network.get("nodes", []))
    n_edges = len(network.get("edges", []))
    n_isolated = n_nodes - len({e["source"] for e in network["edges"]} |
                                {e["target"] for e in network["edges"]})

    m1, m2, m3 = st.columns(3)
    m1.metric("Genes mappes", n_nodes)
    m2.metric("Interactions", n_edges)
    m3.metric("Genes isoles", n_isolated)

    if n_edges == 0:
        st.warning(
            "Aucune interaction trouvee au seuil de confiance choisi. "
            "Essayez de baisser la confiance (low/medium)."
        )
        return

    # ── VISUALISATION ──
    fig = _draw_network_plotly(network, title=f"Reseau STRING — {context_label}")
    if fig:
        st.plotly_chart(fig, width='stretch')

    # ── DEGRÉS ──
    with st.expander("🔍 Genes les plus connectes (hubs)", expanded=False):
        degree = {n["id"]: 0 for n in network["nodes"]}
        for e in network["edges"]:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1
        df_deg = pd.DataFrame([
            {"Gene": k, "Connexions": v} for k, v in degree.items() if v > 0
        ]).sort_values("Connexions", ascending=False)
        st.dataframe(df_deg, hide_index=True, width='stretch', height=300)

    # ── ENRICHISSEMENT ──
    st.markdown("### 🧪 Enrichissement fonctionnel STRING")
    with st.spinner("Enrichissement..."):
        enrichment = get_string_enrichment(selected_genes)

    if not enrichment:
        st.info("Aucun enrichissement significatif retourne par STRING.")
    else:
        # STRING categorise par 'category' (GO Process, KEGG, Reactome, etc.)
        df_enrich = pd.DataFrame(enrichment)

        # Garde les colonnes principales
        keep = [c for c in ["category", "term", "description", "number_of_genes",
                            "number_of_genes_in_background", "p_value", "fdr",
                            "preferredNames"] if c in df_enrich.columns]
        df_enrich = df_enrich[keep].copy()

        # Filtre FDR < 0.05 si dispo
        if "fdr" in df_enrich.columns:
            df_enrich = df_enrich[df_enrich["fdr"].astype(float) < 0.05]

        if len(df_enrich) == 0:
            st.info("Aucun terme avec FDR < 0.05.")
        else:
            categories = df_enrich["category"].unique() if "category" in df_enrich.columns else []
            for cat in sorted(categories):
                with st.expander(f"📂 {cat} ({(df_enrich['category'] == cat).sum()} termes)"):
                    sub = df_enrich[df_enrich["category"] == cat].sort_values("fdr").head(20)
                    st.dataframe(sub.reset_index(drop=True),
                                  width='stretch', height=300)

            st.download_button(
                "📥 Exporter enrichissement (CSV)",
                df_enrich.to_csv(index=False, sep=";"),
                f"string_enrichment_{context_label.replace(' ', '_')}.csv",
                "text/csv", width='stretch'
            )
