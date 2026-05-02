"""Sidebar : upload, GMT pathways, clé API, filtres globaux."""
import streamlit as st

from core.config import IMPACT_ORDER, ACMG_ORDER
from core.data_loader import load_data, load_gmt, load_gmt_from_path, load_gmt_from_url


def render_sidebar():
    """
    Rend la sidebar (header + upload + GMT + clé API + filtres) et retourne :
      df, df_f, pathways_dict, api_key
    Si aucun fichier n'est uploadé, appelle st.stop().
    """
    st.markdown('<p class="main-title">🧬 Variant Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Exploration interactive de variants génomiques — Séquençage ciblé</p>', unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("📁 Fichier variants (.csv)", type=["csv"])
    
    # ── CHARGEMENT DES PATHWAYS (multi-sources) ──
    st.sidebar.markdown("### 🧬 Pathways (optionnel)")
    
    # 1. Essayer de trouver un fichier GMT local dans le repo (détection flexible)
    import os
    import glob
    
    def find_local_gmt():
        """Cherche un fichier GMT dans le repo en étant flexible sur le nom/extension."""
        # Chemins prioritaires (noms canoniques)
        priority_paths = ["pathways.gmt", "data/pathways.gmt", "gmt/pathways.gmt"]
        for p in priority_paths:
            if os.path.exists(p):
                return p
    
        # Fallback : n'importe quel fichier .gmt ou contenant "gmt" dans le nom à la racine ou sous-dossier
        patterns = ["*.gmt", "*gmt*.txt", "data/*.gmt", "gmt/*.gmt", "data/*gmt*.txt"]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                # Retourner le plus petit (souvent le plus pertinent) ou le premier
                return matches[0]
        return None
    
    local_gmt_path = find_local_gmt()
    
    # 2. Interface
    pathways_dict = None
    gmt_source = "Aucun"
    
    if local_gmt_path:
        # Fichier trouvé automatiquement dans le repo
        use_local_gmt = st.sidebar.checkbox(
            f"✅ Utiliser `{local_gmt_path}` (repo)", value=True,
            help="Fichier GMT détecté automatiquement dans le repo GitHub."
        )
        if use_local_gmt:
            pathways_dict = load_gmt_from_path(local_gmt_path)
            gmt_source = f"Local: {local_gmt_path}"
    
    # Option alternative : uploader un fichier
    gmt_file = st.sidebar.file_uploader(
        "📁 Ou uploader un GMT (.gmt)", type=["gmt"],
        help="Alternative : uploader manuellement un fichier GMT (MSigDB)."
    )
    if gmt_file:
        pathways_dict = load_gmt(gmt_file)
        gmt_source = f"Upload: {gmt_file.name}"
    
    # Option alternative : URL
    with st.sidebar.expander("🌐 Ou depuis une URL"):
        gmt_url = st.text_input("URL du fichier GMT", value="",
            placeholder="https://...")
        if st.button("Télécharger", key="dl_gmt") and gmt_url:
            with st.spinner("Téléchargement..."):
                pw = load_gmt_from_url(gmt_url)
                if pw:
                    pathways_dict = pw
                    gmt_source = f"URL ({len(pw)} pathways)"
                    st.success(f"✅ {len(pw)} pathways chargés")
                else:
                    st.error("Échec du téléchargement. Vérifiez l'URL.")
    
    if pathways_dict:
        st.sidebar.markdown(f"📊 **{len(pathways_dict)} pathways** chargés")
        st.sidebar.caption(f"Source : {gmt_source}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 IA")
    api_key = st.sidebar.text_input("Clé API Anthropic", type="password",
        help="Optionnel. Pour l'interprétation IA des clusters.")
    
    if uploaded_file is None:
        st.info("👈 **Chargez votre fichier CSV** via la barre latérale."); st.stop()

    df = load_data(uploaded_file)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Filtres globaux")
    sel_patients = st.sidebar.multiselect("Patients", sorted(df["Pseudo"].unique()), placeholder="Tous")
    sel_genes = st.sidebar.multiselect("Gènes", sorted(df["Gene_symbol"].unique()), placeholder="Tous")
    sel_impacts = st.sidebar.multiselect("Impact", IMPACT_ORDER, placeholder="Tous")
    sel_acmg = st.sidebar.multiselect("ACMG", ACMG_ORDER, placeholder="Toutes")
    af_max = st.sidebar.slider("gnomAD NFE AF max", 0.0, 1.0, 1.0, 0.001, format="%.3f")
    cadd_min = st.sidebar.slider("CADD min", 0.0, 50.0, 0.0, 0.5)
    ar_range = st.sidebar.slider("Allelic ratio", 0.0, 1.0, (0.0, 1.0), 0.01)
    depth_min = st.sidebar.number_input("Profondeur min", min_value=0, value=0, step=10)
    
    df_f = df.copy()
    if sel_patients: df_f = df_f[df_f["Pseudo"].isin(sel_patients)]
    if sel_genes: df_f = df_f[df_f["Gene_symbol"].isin(sel_genes)]
    if sel_impacts: df_f = df_f[df_f["Putative_impact"].isin(sel_impacts)]
    if sel_acmg: df_f = df_f[df_f["ACMG_class"].isin(sel_acmg)]
    df_f = df_f[
        (df_f["gnomad_exomes_NFE_AF"].fillna(0) <= af_max) &
        (df_f["CADD_phred"].fillna(0) >= cadd_min) &
        (df_f["Allelic_ratio"].between(ar_range[0], ar_range[1])) &
        (df_f["Depth"] >= depth_min)
    ]

    return df, df_f, pathways_dict, api_key
