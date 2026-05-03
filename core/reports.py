"""Génération des rapports PDF (clustering + complications)."""
import io
import numpy as np
import pandas as pd

from core.config import FEATURE_LABELS

def generate_cluster_report(df_clust, df_feat, cluster_labels, interpretations,
                            gene_sigs, sil_score, method_name, key_feat,
                            excluded_patients=None, pathways_dict=None):
    """Genere un rapport PDF complet sur le clustering."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, PageBreak, Image, HRFlowable)
    import datetime

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MainTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e")))
    styles.add(ParagraphStyle(name='SubTitle2', parent=styles['Normal'],
        fontSize=12, textColor=colors.HexColor("#666666"), spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionTitle', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor("#0f3460"), spaceBefore=20, spaceAfter=10))
    styles.add(ParagraphStyle(name='SubSection', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name='BodyJ', parent=styles['Normal'],
        fontSize=9, leading=13, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name='ClusterName2', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor("#ff6b6b"), spaceBefore=16, spaceAfter=6))
    styles.add(ParagraphStyle(name='FeatureUp', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor("#cc0000")))
    styles.add(ParagraphStyle(name='FeatureDown', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor("#008080")))

    story = []

    def make_table(data, col_widths=None, header_color="#0f3460"):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        return t

    df_fc = df_feat.copy()
    df_fc["Cluster"] = cluster_labels
    n_clusters = len(set(cluster_labels))

    # ========== PAGE 1 : TITRE + RESUME ==========
    story.append(Spacer(1, 30))
    story.append(Paragraph("Variant Explorer - Rapport de Clustering", styles['MainTitle']))
    story.append(Paragraph(f"Date : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['SubTitle2']))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Resume executif", styles['SectionTitle']))

    summary = [
        ["Parametre", "Valeur"],
        ["Patients analyses", str(len(df_feat))],
        ["Patients exclus (FFPE/techniques)", str(len(excluded_patients)) if excluded_patients else "0"],
        ["Variants (apres filtrage)", f"{len(df_clust):,}"],
        ["Genes", str(df_clust["Gene_symbol"].nunique())],
        ["Nombre de clusters", str(n_clusters)],
        ["Methode", method_name],
        ["Score silhouette", f"{sil_score:.3f}"],
        ["Features utilisees", str(len(key_feat))],
    ]
    story.append(make_table(summary, col_widths=[160, 220]))
    story.append(Spacer(1, 10))

    if excluded_patients:
        story.append(Paragraph("Patients exclus du clustering biologique", styles['SubSection']))
        story.append(Paragraph(
            f"Retires avant clustering (VAF mediane trop basse / trop peu de variants) : "
            f"<b>{', '.join(sorted(excluded_patients))}</b>", styles['BodyJ']))
        story.append(Spacer(1, 6))

    sil_txt = ("Bonne separation." if sil_score > 0.5
               else "Separation moderee." if sil_score > 0.3
               else "Faible separation, chevauchement partiel.")
    story.append(Paragraph(f"<b>Silhouette ({sil_score:.3f})</b> : {sil_txt}", styles['BodyJ']))

    # ========== PAGE 2 : PROFILS GLOBAUX ==========
    story.append(PageBreak())
    story.append(Paragraph("Profil global des clusters", styles['SectionTitle']))

    # Tableau comparatif clusters
    comp_cols = [("vaf_median", "VAF med."), ("vaf_mean", "VAF moy."),
                 ("pct_clonal", "% clonal"), ("pct_minor", "% mineur"),
                 ("clonal_ratio", "Ratio cl/min"), ("tmb_score", "Score TMB"),
                 ("impact_score_mean", "Impact moy"), ("n_unique_genes", "N genes"),
                 ("pct_Pathogenic", "% Patho"), ("pct_VUS", "% VUS"),
                 ("pct_impact_high", "% HIGH"), ("median_CADD", "CADD med")]

    avail_comp = [(c, l) for c, l in comp_cols if c in df_fc.columns]
    if avail_comp:
        header = ["Cluster", "N pat."] + [l for _, l in avail_comp]
        comp_data = [header]
        for cl in sorted(df_fc["Cluster"].unique()):
            sub = df_fc[df_fc["Cluster"] == cl]
            row_data = [str(cl), str(len(sub))]
            for c, _ in avail_comp:
                v = sub[c].mean()
                row_data.append(f"{v:.3f}" if v < 1 else f"{v:.1f}")
            comp_data.append(row_data)
        # Ligne globale
        glob_row = ["GLOBAL", str(len(df_fc))]
        for c, _ in avail_comp:
            v = df_fc[c].mean()
            glob_row.append(f"{v:.3f}" if v < 1 else f"{v:.1f}")
        comp_data.append(glob_row)

        story.append(Paragraph("Comparaison inter-clusters (moyennes)", styles['SubSection']))
        cw = [65, 35] + [42] * len(avail_comp)
        t_comp = make_table(comp_data, col_widths=cw)
        story.append(t_comp)
    story.append(Spacer(1, 12))

    # Heatmap features cles
    if key_feat:
        story.append(Paragraph("Heatmap des features cles", styles['SubSection']))
        display_kf = [f for f in key_feat if not f.startswith("pw_")][:20]
        hm_data = [["Feature", "Global"] + sorted(df_fc["Cluster"].unique())]
        for f in display_kf:
            label = FEATURE_LABELS.get(f, f.replace("pct_", "").replace("_", " "))[:25]
            row_hm = [label, f"{df_fc[f].mean():.2f}"]
            for cl in sorted(df_fc["Cluster"].unique()):
                v = df_fc[df_fc["Cluster"] == cl][f].mean()
                row_hm.append(f"{v:.2f}")
            hm_data.append(row_hm)
        cw_hm = [100, 45] + [50] * n_clusters
        story.append(make_table(hm_data, col_widths=cw_hm, header_color="#16213e"))

    # ========== PAGES 3+ : DETAIL PAR CLUSTER ==========
    for cid in sorted(interpretations.keys()):
        interp = interpretations[cid]
        story.append(PageBreak())
        story.append(Paragraph(f"{cid}", styles['ClusterName2']))
        story.append(Paragraph(
            f"<b>{interp['n_patients']} patients</b> : {', '.join(sorted(interp['patients']))}",
            styles['BodyJ']))
        story.append(Spacer(1, 8))

        cluster_data = df_fc[df_fc["Cluster"] == cid]

        # Stats descriptives
        story.append(Paragraph("Statistiques descriptives", styles['SubSection']))
        stat_items = [("vaf_median", "VAF mediane"), ("pct_clonal", "% clonal"),
                      ("pct_minor", "% mineur"), ("tmb_score", "Score TMB"),
                      ("impact_score_mean", "Score impact moy"), ("n_unique_genes", "Genes uniques"),
                      ("pct_Pathogenic", "% Pathogenes"), ("pct_VUS", "% VUS"),
                      ("pct_impact_high", "% Impact HIGH"), ("median_CADD", "CADD median"),
                      ("clonal_ratio", "Ratio clonal/mineur")]
        stat_data = [["Metrique", "Moyenne", "Mediane", "Ecart-type", "Min", "Max"]]
        for col, label in stat_items:
            if col in cluster_data.columns:
                vals = cluster_data[col]
                stat_data.append([label, f"{vals.mean():.2f}", f"{vals.median():.2f}",
                    f"{vals.std():.2f}", f"{vals.min():.2f}", f"{vals.max():.2f}"])
        story.append(make_table(stat_data, col_widths=[90, 55, 55, 55, 55, 55]))
        story.append(Spacer(1, 8))

        # Features discriminantes
        sig = interp["significant"]
        if len(sig) > 0:
            story.append(Paragraph("Features discriminantes (|z| &gt; 0.5)", styles['SubSection']))
            enriched = sig[sig > 0]
            if len(enriched) > 0:
                story.append(Paragraph("<b>Enrichies :</b>", styles['BodyJ']))
                for feat, z in enriched.items():
                    label = FEATURE_LABELS.get(feat, feat.replace("pw_pct_", "PW: ").replace("pct_", "").replace("_", " "))
                    val = interp['cluster_mean'][feat]
                    glob = interp['global_mean'][feat]
                    story.append(Paragraph(
                        f"  &#9650; {label} : {val:.2f} vs {glob:.2f} (z = {z:+.2f})", styles['FeatureUp']))
            depleted = sig[sig < 0]
            if len(depleted) > 0:
                story.append(Spacer(1, 3))
                story.append(Paragraph("<b>Reduites :</b>", styles['BodyJ']))
                for feat, z in depleted.items():
                    label = FEATURE_LABELS.get(feat, feat.replace("pw_pct_", "PW: ").replace("pct_", "").replace("_", " "))
                    val = interp['cluster_mean'][feat]
                    glob = interp['global_mean'][feat]
                    story.append(Paragraph(
                        f"  &#9660; {label} : {val:.2f} vs {glob:.2f} (z = {z:+.2f})", styles['FeatureDown']))
        story.append(Spacer(1, 8))

        # Genes
        if cid in gene_sigs:
            gs = gene_sigs[cid]
            if len(gs["pathogenic_genes"]) > 0:
                story.append(Paragraph("Genes avec variants pathogenes", styles['SubSection']))
                gd = [["Gene", "N variants"]]
                for g, c in gs["pathogenic_genes"].head(10).items():
                    gd.append([str(g), str(c)])
                story.append(make_table(gd, col_widths=[120, 100], header_color="#cc0000"))
                story.append(Spacer(1, 4))
            if len(gs["enriched_genes"]) > 0:
                story.append(Paragraph("Genes enrichis (vs autres clusters)", styles['SubSection']))
                ed = [["Gene", "Enrichissement"]]
                for g, r in gs["enriched_genes"].head(10).items():
                    ed.append([str(g), f"x{r:.1f}"])
                story.append(make_table(ed, col_widths=[120, 100], header_color="#008080"))
        story.append(Spacer(1, 6))

        # Tableau patients
        story.append(Paragraph("Detail des patients", styles['SubSection']))
        pat_data = [["Patient", "VAF med.", "% clonal", "TMB", "Impact", "Genes", "% Patho", "% VUS"]]
        for pat in sorted(interp["patients"]):
            if pat in df_feat.index:
                r = df_feat.loc[pat]
                pat_data.append([pat,
                    f"{r.get('vaf_median', 0):.3f}", f"{r.get('pct_clonal', 0):.1f}%",
                    f"{r.get('tmb_score', 0):.1f}", f"{r.get('impact_score_mean', 0):.2f}",
                    f"{int(r.get('n_unique_genes', 0))}", f"{r.get('pct_Pathogenic', 0):.1f}%",
                    f"{r.get('pct_VUS', 0):.1f}%"])
        story.append(make_table(pat_data, col_widths=[55, 45, 45, 40, 40, 40, 45, 40]))

    # ========== ANNEXE ==========
    story.append(PageBreak())
    story.append(Paragraph("Annexe : Matrice complete des features cles", styles['SectionTitle']))
    display_feats = [c for c in key_feat if not c.startswith("pw_") and not c.startswith("gene")][:15]
    if display_feats:
        header = ["Patient", "Cluster"] + [FEATURE_LABELS.get(f, f)[:16] for f in display_feats]
        annex = [header]
        for pat in sorted(df_feat.index):
            cl = df_fc.loc[pat, "Cluster"]
            row = [pat, str(cl)]
            for f in display_feats:
                v = df_feat.loc[pat].get(f, 0)
                row.append(f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}")
            annex.append(row)
        cw_a = [50, 50] + [35] * len(display_feats)
        story.append(make_table(annex, col_widths=cw_a, header_color="#0f3460"))

    # ========== METHODOLOGIE ==========
    story.append(PageBreak())
    story.append(Paragraph("Methodologie", styles['SectionTitle']))
    story.append(Paragraph(
        "Filtrage qualite : profondeur minimale, ratio allelique minimum, frequence gnomAD NFE maximale. "
        "Exclusion des variants synonymes, introniques et UTR. Features genomiques en proportions (%) "
        "pour neutraliser le biais lie a la qualite FFPE. Features VAF (mediane, % clonal, ratio "
        "clonal/mineur, score TMB) integrees pour capturer la structure clonale. Features de genes "
        "binaires ponderees par score d'impact maximal. Pathways (GMT, MSigDB) representes par le "
        "pourcentage de genes touches et le score d'impact maximal.", styles['BodyJ']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Score d'impact composite (0-10) par variant : impact fonctionnel (HIGH=4, MODERATE=2, "
        "LOW=0.5), CADD normalise (0-2), rarete allelique gnomAD NFE (0-2), ClinVar (0-2).",
        styles['BodyJ']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Clustering : {method_name} apres standardisation z-score. "
        f"Score silhouette : {sil_score:.3f}. UMAP pour visualisation 2D. "
        f"Mode 2 etapes : exclusion prealable des suspects FFPE (VAF mediane basse).",
        styles['BodyJ']))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()




def generate_complications_report(df_c, df_pat_elig, df_pat_clin,
                                   df_res_var, df_res_gene,
                                   n_elig, n_compl, n_no_compl, n_excluded,
                                   compl_counts, avail_compl,
                                   compl_depth, compl_ar, compl_af, compl_excl_ben,
                                   excluded_effects_compl,
                                   n_variants_base, n_variants_filtered, pct_kept,
                                   min_carriers, correction_method,
                                   exclude_no_clinical, excluded_list):
    """Genere un rapport PDF complet de l'analyse des complications."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, PageBreak, Image, HRFlowable)
    import datetime

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    # Helper: add style only if it doesn't exist
    def add_style(name, **kwargs):
        if name not in styles:
            styles.add(ParagraphStyle(name=name, **kwargs))

    add_style('CompMainTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    add_style('CompSubTitle', parent=styles['Normal'],
        fontSize=12, textColor=colors.HexColor("#666666"), spaceAfter=20)
    add_style('CompSection', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor("#0f3460"), spaceBefore=16, spaceAfter=10)
    add_style('CompSub', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=12, spaceAfter=8)
    add_style('CompBody', parent=styles['Normal'],
        fontSize=9, leading=13, alignment=TA_JUSTIFY)
    add_style('CompSmall', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    add_style('CompSuccess', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#006600"),
        backColor=colors.HexColor("#e6f7e6"), borderPadding=6, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=6)
    add_style('CompWarning', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#b07700"),
        backColor=colors.HexColor("#fff5e0"), borderPadding=6, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=6)
    add_style('CompInfo', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=colors.HexColor("#0a4466"),
        backColor=colors.HexColor("#e0f0fa"), borderPadding=6, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=6)

    story = []

    def make_table(data, col_widths=None, header_color="#0f3460", fontsize=8):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), fontsize),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        return t

    def fig_to_image(matplotlib_fig, width_mm=170):
        """Convert a matplotlib Figure to a reportlab Image via in-memory PNG."""
        import matplotlib.pyplot as plt
        img_buf = io.BytesIO()
        matplotlib_fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight",
                               facecolor="white", edgecolor="none")
        plt.close(matplotlib_fig)
        img_buf.seek(0)
        # Calculate height from the figure dims
        w_in, h_in = matplotlib_fig.get_size_inches()
        height_mm = width_mm * (h_in / w_in)
        return Image(img_buf, width=width_mm*mm, height=height_mm*mm)

    # ========== PAGE 1 : TITRE + RESUME ==========
    story.append(Spacer(1, 30))
    story.append(Paragraph("Analyse des complications - Rapport", styles['CompMainTitle']))
    story.append(Paragraph(f"Date : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
                           styles['CompSubTitle']))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Question scientifique", styles['CompSection']))
    story.append(Paragraph(
        "Existe-t-il une signature genomique particuliere (au niveau du variant ou du gene) "
        "associee a l'apparition d'une complication (BO, PNP, MG, FDSCS) chez les patients "
        "de la cohorte ? Analyse supervisee par test exact de Fisher.",
        styles['CompBody']))
    story.append(Spacer(1, 12))

    # Tableau recap
    story.append(Paragraph("Resume de l'analyse", styles['CompSub']))
    recap = [
        ["Parametre", "Valeur"],
        ["Patients eligibles", str(n_elig)],
        ["Avec complication", str(n_compl)],
        ["Sans complication", str(n_no_compl)],
        ["Exclus (aucune info clinique)", str(n_excluded)],
        ["Variants avant filtrage", f"{n_variants_base:,}"],
        ["Variants apres filtrage", f"{n_variants_filtered:,}"],
        ["Pourcentage conserve", f"{pct_kept:.1f}%"],
        ["Seuil patients porteurs min", str(min_carriers)],
        ["Correction multi-tests", correction_method],
    ]
    story.append(make_table(recap, col_widths=[180, 180]))
    story.append(Spacer(1, 12))

    # Filtres detailles
    story.append(Paragraph("Filtres qualite appliques", styles['CompSub']))
    filters_tbl = [
        ["Filtre", "Valeur"],
        ["Profondeur minimale (Depth)", f"≥ {compl_depth}"],
        ["Ratio allelique minimum (AR)", f"≥ {compl_ar:.2f}"],
        ["gnomAD NFE frequence max", f"≤ {compl_af:.3f}"],
        ["Benign / Likely Benign", "Exclus" if compl_excl_ben else "Inclus"],
        ["Types exclus", ", ".join(sorted(excluded_effects_compl)) if excluded_effects_compl else "Aucun"],
        ["Patients sans info clinique", "Exclus" if exclude_no_clinical else "Inclus"],
    ]
    story.append(make_table(filters_tbl, col_widths=[200, 260]))

    if exclude_no_clinical and n_excluded > 0 and excluded_list:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"<b>Patients exclus ({len(excluded_list)})</b> : {', '.join(sorted(excluded_list))}",
            styles['CompSmall']))

    # ========== PAGE 2 : REPARTITION DES COMPLICATIONS ==========
    story.append(PageBreak())
    story.append(Paragraph("Repartition des complications", styles['CompSection']))

    # Tableau comptage
    compl_tbl = [["Type de complication", "Nb patients", "% de la cohorte"]]
    for col in avail_compl:
        n = compl_counts.get(col, 0)
        compl_tbl.append([col, str(n), f"{n/max(n_elig,1)*100:.1f}%"])
    story.append(make_table(compl_tbl, col_widths=[150, 100, 150]))
    story.append(Spacer(1, 10))

    # Figure : bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig_mpl, ax = plt.subplots(figsize=(8, 3.5))
        bar_colors = ["#ff6b6b", "#ffa500", "#ffd93d", "#9b59b6"][:len(compl_counts)]
        bars = ax.bar(list(compl_counts.keys()), list(compl_counts.values()),
                      color=bar_colors, edgecolor="#333", linewidth=0.5)
        for bar, val in zip(bars, compl_counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   str(val), ha="center", fontsize=10)
        ax.set_ylabel("Patients", fontsize=10)
        ax.set_title("Nombre de patients par type de complication", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
        fig_mpl.tight_layout()
        story.append(fig_to_image(fig_mpl, width_mm=170))
    except Exception as e:
        story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))
    story.append(Spacer(1, 8))

    # Figure : pie chart compl vs non-compl
    try:
        fig_mpl, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            [n_compl, n_no_compl],
            labels=["Complique", "Non complique"],
            colors=["#ff6b6b", "#4ecdc4"],
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100 * (n_compl+n_no_compl)))})",
            startangle=90, wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10))
        ax.set_title("Repartition compliques vs non compliques", fontsize=11)
        fig_mpl.tight_layout()
        story.append(fig_to_image(fig_mpl, width_mm=120))
    except Exception as e:
        story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))

    # ========== PAGE 3+ : NIVEAU VARIANT ==========
    story.append(PageBreak())
    story.append(Paragraph("Analyse par variant", styles['CompSection']))

    if df_res_var is None or len(df_res_var) == 0:
        story.append(Paragraph(
            "Aucun variant n'est present chez suffisamment de patients pour etre teste. "
            "Abaissez le seuil de porteurs minimum ou elargissez la cohorte.",
            styles['CompWarning']))
    else:
        sig_var = df_res_var[df_res_var["P_adjusted"] < 0.05]
        nominal_sig_var = df_res_var[df_res_var["P_value"] < 0.05]

        # Conclusion statistique
        story.append(Paragraph("Conclusion statistique", styles['CompSub']))
        if len(sig_var) > 0:
            top_with_gene = sig_var.head(5).apply(
                lambda r: f"{r.get('Gene', 'N/A')} ({r['Entity']})" if pd.notna(r.get('Gene')) else str(r['Entity']),
                axis=1
            ).tolist()
            story.append(Paragraph(
                f"<b>{len(sig_var)} variant(s) significativement associe(s)</b> a la complication "
                f"apres correction {correction_method} (p ajustee < 0.05). "
                f"Top : {', '.join(top_with_gene)}. Validation sur cohorte independante necessaire.",
                styles['CompSuccess']))
        elif len(nominal_sig_var) > 0:
            story.append(Paragraph(
                f"<b>Aucun variant significatif apres correction {correction_method}</b> "
                f"(p ajustee &lt; 0.05). {len(nominal_sig_var)} variant(s) ont une p brute &lt; 0.05, "
                f"suggerant des signaux exploratoires non concluants. "
                f"Attendu avec une petite cohorte et beaucoup de tests multiples.",
                styles['CompWarning']))
        else:
            story.append(Paragraph(
                "<b>Aucun variant n'est associe</b> a la complication. "
                "Resultat attendu : il est rare que plusieurs patients partagent exactement "
                "le meme variant. L'analyse par gene est plus appropriee.",
                styles['CompInfo']))

        # Volcano plot variants
        story.append(Spacer(1, 6))
        story.append(Paragraph("Volcano plot", styles['CompSub']))
        df_res_var_plot = df_res_var.copy()
        df_res_var_plot["log2_OR"] = np.log2(df_res_var_plot["Odds_Ratio"].clip(lower=0.01, upper=100))
        df_res_var_plot["log10_p"] = -np.log10(df_res_var_plot["P_adjusted"].clip(lower=1e-10))
        df_res_var_plot["Significant"] = df_res_var_plot["P_adjusted"] < 0.05

        try:
            fig_mpl, ax = plt.subplots(figsize=(9, 5))
            # Non-significatifs en gris
            ns = df_res_var_plot[~df_res_var_plot["Significant"]]
            s = df_res_var_plot[df_res_var_plot["Significant"]]
            ax.scatter(ns["log2_OR"], ns["log10_p"], c="#555555", s=25, alpha=0.6,
                       edgecolor="none", label="Non significatif")
            if len(s) > 0:
                ax.scatter(s["log2_OR"], s["log10_p"], c="#ff6b6b", s=40, alpha=0.8,
                           edgecolor="#cc0000", linewidth=0.5, label="Significatif")
            ax.axvline(0, linestyle="--", color="#888", linewidth=0.8)
            ax.axhline(-np.log10(0.05), linestyle="--", color="#ffa500", linewidth=0.8)
            ax.text(ax.get_xlim()[1] * 0.95, -np.log10(0.05) + 0.05, "p ajustee = 0.05",
                    fontsize=8, color="#ffa500", ha="right")
            ax.set_xlabel("log2(Odds Ratio)", fontsize=10)
            ax.set_ylabel("-log10(P ajustee)", fontsize=10)
            ax.set_title("Volcano plot : variants", fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=9, loc="upper right")
            ax.tick_params(labelsize=9)
            fig_mpl.tight_layout()
            story.append(fig_to_image(fig_mpl, width_mm=170))
        except Exception as e:
            story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))
        story.append(Spacer(1, 8))

        # Tableau top 20 variants
        story.append(PageBreak())
        story.append(Paragraph("Top 20 variants associes", styles['CompSub']))
        gene_col = "Gene" if "Gene" in df_res_var.columns else ("Gène" if "Gène" in df_res_var.columns else None)
        hgvs_col = "hgvs.p" if "hgvs.p" in df_res_var.columns else None
        eff_col = "Variant_effect" if "Variant_effect" in df_res_var.columns else None

        headers = ["Gene", "Variant", "HGVS p.", "Effet", "N port.", "Comp+", "Comp-", "OR", "P brute", "P adj."]
        var_tbl = [headers]
        for _, r in df_res_var.head(20).iterrows():
            var_tbl.append([
                str(r.get(gene_col, "-"))[:10] if gene_col else "-",
                str(r["Entity"])[:18],
                str(r.get(hgvs_col, "-") if hgvs_col else "-")[:14],
                str(r.get(eff_col, "-") if eff_col else "-")[:12],
                str(int(r["N_carriers"])),
                str(int(r["Carriers_with_compl"])),
                str(int(r["Carriers_without_compl"])),
                f"{r['Odds_Ratio']:.2f}",
                f"{r['P_value']:.4f}",
                f"{r['P_adjusted']:.4f}",
            ])
        col_widths_var = [50, 85, 65, 55, 32, 32, 32, 32, 40, 40]
        story.append(make_table(var_tbl, col_widths=col_widths_var, fontsize=7))

    # ========== NIVEAU GENE ==========
    story.append(PageBreak())
    story.append(Paragraph("Analyse par gene (Pathogenic / LP / VUS)", styles['CompSection']))
    story.append(Paragraph(
        "Test sur les genes ou au moins une mutation Pathogenic, Likely Pathogenic ou VUS "
        "est presente chez au moins N patients. Les variants Benign/LB sont exclus.",
        styles['CompBody']))
    story.append(Spacer(1, 6))

    if df_res_gene is None or len(df_res_gene) == 0:
        story.append(Paragraph(
            "Aucun gene n'est mute chez suffisamment de patients pour etre teste.",
            styles['CompWarning']))
    else:
        sig_gene = df_res_gene[df_res_gene["P_adjusted"] < 0.05]
        nominal_sig_gene = df_res_gene[df_res_gene["P_value"] < 0.05]

        story.append(Paragraph("Conclusion statistique", styles['CompSub']))
        if len(sig_gene) > 0:
            top_genes = sig_gene.head(5)["Entity"].tolist()
            story.append(Paragraph(
                f"<b>{len(sig_gene)} gene(s) significativement associe(s)</b> a la complication "
                f"apres correction {correction_method} (p ajustee &lt; 0.05). "
                f"Top genes : {', '.join(top_genes)}. "
                f"Ces genes constituent des pistes biologiques interessantes.",
                styles['CompSuccess']))
        elif len(nominal_sig_gene) > 0:
            top_nominal = nominal_sig_gene.head(5)["Entity"].tolist()
            story.append(Paragraph(
                f"<b>Aucun gene significatif apres correction {correction_method}</b> "
                f"(p ajustee &lt; 0.05). {len(nominal_sig_gene)} gene(s) ont une p brute &lt; 0.05 : "
                f"{', '.join(top_nominal)}. Signaux exploratoires a valider. "
                f"Essayez la correction FDR (moins stricte).",
                styles['CompWarning']))
        else:
            story.append(Paragraph(
                "<b>Aucun gene n'est associe</b> a la complication, meme avant correction. "
                "Soit le signal n'existe pas a l'echelle individuelle du gene, soit la cohorte "
                "est trop petite. L'analyse par pathway pourrait reveler des associations.",
                styles['CompInfo']))

        # Volcano plot genes
        story.append(Spacer(1, 6))
        story.append(Paragraph("Volcano plot", styles['CompSub']))
        df_res_gene_plot = df_res_gene.copy()
        df_res_gene_plot["log2_OR"] = np.log2(df_res_gene_plot["Odds_Ratio"].clip(lower=0.01, upper=100))
        df_res_gene_plot["log10_p"] = -np.log10(df_res_gene_plot["P_adjusted"].clip(lower=1e-10))
        df_res_gene_plot["Significant"] = df_res_gene_plot["P_adjusted"] < 0.05

        try:
            fig_mpl, ax = plt.subplots(figsize=(9, 5.5))
            ns = df_res_gene_plot[~df_res_gene_plot["Significant"]]
            s = df_res_gene_plot[df_res_gene_plot["Significant"]]
            ax.scatter(ns["log2_OR"], ns["log10_p"], c="#555555", s=25, alpha=0.6,
                       edgecolor="none", label="Non significatif")
            if len(s) > 0:
                ax.scatter(s["log2_OR"], s["log10_p"], c="#ff6b6b", s=45, alpha=0.85,
                           edgecolor="#cc0000", linewidth=0.5, label="Significatif")
            # Annoter les top hits (p brute < 0.1)
            to_label = df_res_gene_plot[df_res_gene_plot["P_value"] < 0.1].head(10)
            for _, r in to_label.iterrows():
                ax.annotate(str(r["Entity"]), xy=(r["log2_OR"], r["log10_p"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.axvline(0, linestyle="--", color="#888", linewidth=0.8)
            ax.axhline(-np.log10(0.05), linestyle="--", color="#ffa500", linewidth=0.8)
            ax.text(ax.get_xlim()[1] * 0.95, -np.log10(0.05) + 0.05, "p ajustee = 0.05",
                    fontsize=8, color="#ffa500", ha="right")
            ax.set_xlabel("log2(Odds Ratio)", fontsize=10)
            ax.set_ylabel("-log10(P ajustee)", fontsize=10)
            ax.set_title("Volcano plot : genes", fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=9, loc="upper right")
            ax.tick_params(labelsize=9)
            fig_mpl.tight_layout()
            story.append(fig_to_image(fig_mpl, width_mm=170))
        except Exception as e:
            story.append(Paragraph(f"[Erreur figure: {e}]", styles['CompSmall']))

        # Tableau top 20 genes
        story.append(PageBreak())
        story.append(Paragraph("Top 20 genes associes", styles['CompSub']))
        gene_headers = ["Gene", "N port.", "Comp+", "Comp-",
                        "Freq compl porteurs", "Freq compl non-port.",
                        "OR", "P brute", "P ajust."]
        gene_tbl = [gene_headers]
        for _, r in df_res_gene.head(20).iterrows():
            gene_tbl.append([
                str(r["Entity"])[:15],
                str(int(r["N_carriers"])),
                str(int(r["Carriers_with_compl"])),
                str(int(r["Carriers_without_compl"])),
                f"{r['Freq_compl_carriers']:.2f}",
                f"{r['Freq_compl_non_carriers']:.2f}",
                f"{r['Odds_Ratio']:.2f}",
                f"{r['P_value']:.4f}",
                f"{r['P_adjusted']:.4f}",
            ])
        col_widths_gene = [70, 40, 40, 40, 70, 70, 40, 50, 50]
        story.append(make_table(gene_tbl, col_widths=col_widths_gene, fontsize=7))

    # ========== METHODOLOGIE ==========
    story.append(PageBreak())
    story.append(Paragraph("Methodologie et interpretation", styles['CompSection']))

    story.append(Paragraph("Principe de l'analyse", styles['CompSub']))
    story.append(Paragraph(
        "Pour chaque variant (ou gene) present chez au moins N patients, un test exact de Fisher "
        "est applique sur une table de contingence 2x2 croisant la presence du variant (porteur "
        "oui/non) et le statut complication (Complication_any = 1 si au moins une des colonnes BO, "
        "PNP, MG ou FDSCS est a 1). L'odds ratio (OR) mesure la force de l'association : OR &gt; 1 "
        "= enrichissement chez les compliques, OR &lt; 1 = depletion.",
        styles['CompBody']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Correction multi-tests", styles['CompSub']))
    if correction_method == "Bonferroni":
        corr_txt = ("Correction de Bonferroni : p ajustee = p brute * N tests. Methode stricte "
                    "qui minimise les faux positifs mais reduit la puissance. Adaptee quand "
                    "on cherche des signaux robustes.")
    elif correction_method == "FDR (Benjamini-Hochberg)":
        corr_txt = ("Correction FDR (Benjamini-Hochberg) : controle le taux de faux positifs "
                    "attendu. Plus permissive que Bonferroni, adaptee a l'exploration quand on "
                    "genere des hypotheses sur de nombreux tests.")
    else:
        corr_txt = ("Aucune correction appliquee. Les p-values sont brutes. A utiliser uniquement "
                    "pour explorer des tendances ou sur un nombre tres limite de tests.")
    story.append(Paragraph(corr_txt, styles['CompBody']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Limitations et puissance statistique", styles['CompSub']))
    power_note = ""
    if n_elig < 30:
        power_note = ("<b>Cohorte tres petite ({} patients)</b> : la puissance statistique est "
                     "fortement limitee. Les tests multiples apres correction peuvent ne rien "
                     "retenir meme si des signaux biologiques existent.").format(n_elig)
    elif n_elig < 50:
        power_note = ("<b>Cohorte modeste ({} patients)</b> : puissance limitee pour les "
                     "correction strictes. Envisagez une validation independante.").format(n_elig)
    else:
        power_note = ("Cohorte de taille acceptable ({} patients).").format(n_elig)

    if n_compl < 5 or n_no_compl < 5:
        power_note += (" <b>ATTENTION</b> : un des groupes contient moins de 5 patients "
                      "(compliques={}, non-compliques={}), la puissance est critique.").format(n_compl, n_no_compl)

    story.append(Paragraph(power_note, styles['CompBody']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Niveaux d'analyse", styles['CompSub']))
    story.append(Paragraph(
        "<b>Niveau variant</b> : identifie les variants exacts associes. Peu de signal attendu "
        "car il est rare que plusieurs patients partagent exactement le meme variant dans un "
        "contexte somatique FFPE.",
        styles['CompBody']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Niveau gene</b> : identifie les genes ou une mutation delertere (Patho/LP/VUS) est "
        "enrichie chez les compliques. Plus sensible car il agrege les variants par gene. Les "
        "variants Benign/Likely Benign sont exclus car ils diluent le signal.",
        styles['CompBody']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Niveau pathway</b> (a venir) : identifie les voies biologiques enrichies. Plus "
        "sensible encore car agrege plusieurs genes d'une meme voie.",
        styles['CompBody']))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Rapport genere par Variant Explorer v4.0 - {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles['CompSmall']))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

