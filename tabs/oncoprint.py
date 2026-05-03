"""Onglet : OncoPrint"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import IMPACT_ORDER, ACMG_ORDER, NON_FUNCTIONAL_EFFECTS


def render(df_f, df, pathways_dict, api_key):
    st.markdown("## 🧬 OncoPrint - Profil mutationnel")
    st.markdown(
        "Visualisation de type **OncoPrint** (inspirée de la Figure 1B) montrant les gènes mutés "
        "par patient avec annotations cliniques. Chaque ligne représente un gène, chaque colonne un patient."
    )
    
    # ── PARAMÈTRES ──
    st.markdown("### ⚙️ Paramètres de visualisation")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        min_patients_onco = st.slider(
            "Gènes mutés chez au moins N patients",
            min_value=1, max_value=10, value=2,
            help="Afficher uniquement les gènes mutés chez au moins N patients"
        )
    
    with col_p2:
        max_genes_onco = st.slider(
            "Nombre max de gènes à afficher",
            min_value=10, max_value=100, value=30,
            help="Limiter l'affichage aux N gènes les plus fréquemment mutés"
        )
    
    with col_p3:
        show_annotations = st.checkbox(
            "Afficher les annotations cliniques",
            value=True,
            help="Afficher le sexe, l'âge, l'histologie et les complications au-dessus de la heatmap"
        )
    
    # ── FILTRAGE DES VARIANTS ──
    st.markdown("### 🧹 Filtres des variants")
    
    fcol1, fcol2, fcol3 = st.columns(3)
    
    with fcol1:
        onco_acmg_filter = st.multiselect(
            "Classifications ACMG",
            ACMG_ORDER,
            default=["Pathogenic", "Likely Pathogenic", "VUS"],
            help="Inclure uniquement ces classifications"
        )
    
    with fcol2:
        onco_impact_filter = st.multiselect(
            "Impacts fonctionnels",
            IMPACT_ORDER,
            default=["high", "moderate"],
            help="Inclure uniquement ces niveaux d'impact"
        )
    
    with fcol3:
        onco_exclude_synonymous = st.checkbox(
            "Exclure variants non-fonctionnels",
            value=True,
            help="Exclure synonymes, introniques, UTR, etc."
        )
    
    # Appliquer les filtres
    df_onco = df_f.copy()
    
    if len(onco_acmg_filter) > 0:
        df_onco = df_onco[df_onco['ACMG_class'].isin(onco_acmg_filter)]
    
    if len(onco_impact_filter) > 0:
        df_onco = df_onco[df_onco['Putative_impact'].isin(onco_impact_filter)]
    
    if onco_exclude_synonymous:
        df_onco = df_onco[~df_onco['Variant_effect'].isin(NON_FUNCTIONAL_EFFECTS)]
    
    if len(df_onco) == 0:
        st.warning("Aucun variant après filtrage. Assouplissez les filtres.")
    else:
        # ── CALCUL DE LA MATRICE ──
        st.markdown("---")
        st.markdown("### 📊 OncoPrint")
        
        # Compter les mutations par gène
        gene_mutation_counts = df_onco.groupby('Gene_symbol')['Pseudo'].nunique()
        
        # Filtrer les gènes fréquents
        frequent_genes = gene_mutation_counts[gene_mutation_counts >= min_patients_onco].sort_values(ascending=False)
        
        if len(frequent_genes) == 0:
            st.warning(f"Aucun gène muté chez ≥{min_patients_onco} patients avec les filtres actuels.")
        else:
            # Limiter au top N
            top_genes = frequent_genes.head(max_genes_onco).index.tolist()
            
            st.info(f"**{len(top_genes)} gènes** affichés (mutés chez ≥{min_patients_onco} patients)")
            
            # Liste de tous les patients
            all_patients = df_f['Pseudo'].unique().tolist()
            
            # Créer la matrice binaire gène × patient avec type de mutation
            mutation_matrix = pd.DataFrame('', index=top_genes, columns=all_patients)
            mutation_type_matrix = pd.DataFrame(0, index=top_genes, columns=all_patients)
            
            # Dictionnaire de priorité des types de mutations (pour affichage)
            mutation_priority = {
                'Frameshift': 5,
                'Nonsense': 4,
                'Splice': 3,
                'Missense': 2,
                'InDel': 1,
                'Other': 0
            }
            
            # Remplir la matrice
            for gene in top_genes:
                gene_data = df_onco[df_onco['Gene_symbol'] == gene]
                
                for patient in all_patients:
                    patient_mutations = gene_data[gene_data['Pseudo'] == patient]
                    
                    if len(patient_mutations) > 0:
                        # Déterminer le type de mutation le plus sévère
                        mutation_types = []
                        
                        for _, row in patient_mutations.iterrows():
                            effect = str(row.get('Variant_effect', '')).lower()
                            
                            if 'frameshift' in effect:
                                mutation_types.append(('Frameshift', mutation_priority['Frameshift']))
                            elif 'stop' in effect or 'nonsense' in effect:
                                mutation_types.append(('Nonsense', mutation_priority['Nonsense']))
                            elif 'splice' in effect:
                                mutation_types.append(('Splice', mutation_priority['Splice']))
                            elif 'missense' in effect:
                                mutation_types.append(('Missense', mutation_priority['Missense']))
                            elif 'del' in effect or 'ins' in effect or 'indel' in effect:
                                mutation_types.append(('InDel', mutation_priority['InDel']))
                            else:
                                mutation_types.append(('Other', mutation_priority['Other']))
                        
                        # Prendre le type le plus sévère
                        if mutation_types:
                            most_severe = max(mutation_types, key=lambda x: x[1])
                            mutation_matrix.loc[gene, patient] = most_severe[0]
                            mutation_type_matrix.loc[gene, patient] = most_severe[1]
            
            # ── ANNOTATIONS CLINIQUES ──
            patient_annotations = {}
            
            if show_annotations:
                for patient in all_patients:
                    patient_data = df_f[df_f['Pseudo'] == patient].iloc[0]
                    
                    # Sexe (chercher dans différentes colonnes possibles)
                    sex = 'Unknown'
                    for col in df_f.columns:
                        if 'sex' in col.lower() or 'genre' in col.lower():
                            sex_val = patient_data.get(col, '')
                            if pd.notna(sex_val) and sex_val != '':
                                sex = str(sex_val)
                                break
                    
                    # Âge
                    age = 'Unknown'
                    for col in df_f.columns:
                        if 'age' in col.lower() or 'âge' in col.lower():
                            age_val = patient_data.get(col, '')
                            if pd.notna(age_val):
                                try:
                                    age = f"{int(age_val)}"
                                except:
                                    age = str(age_val)
                                break
                    
                    # Histologie
                    histo = patient_data.get('Histo UCD', 'Unknown')
                    if pd.isna(histo) or histo == '':
                        histo = 'Unknown'
                    
                    # Complications
                    complications = []
                    for compl in ['BO', 'PNP', 'MG', 'FDSCS']:
                        if compl in df_f.columns:
                            val = patient_data.get(compl, 0)
                            try:
                                if float(val) == 1:
                                    complications.append(compl)
                            except:
                                pass
                    
                    compl_str = '+'.join(complications) if complications else 'Aucune'
                    
                    patient_annotations[patient] = {
                        'sex': sex,
                        'age': age,
                        'histo': histo,
                        'complications': compl_str
                    }
            
            # ── CRÉATION DE LA FIGURE ──
            # Calculer la hauteur selon le nombre de gènes
            height_per_gene = 18
            annotation_height = 80 if show_annotations else 0
            total_height = len(top_genes) * height_per_gene + annotation_height + 150
            
            fig = go.Figure()
            
            # Couleurs pour les types de mutations (style article)
            mutation_colors = {
                'Frameshift': '#8B0000',      # Rouge foncé
                'Nonsense': '#FF0000',         # Rouge
                'Splice': '#FF6B6B',           # Rouge clair
                'Missense': '#4CAF50',         # Vert
                'InDel': '#2196F3',            # Bleu
                'Other': '#FFC107',            # Jaune
                '': '#0a192f'                  # Fond (pas de mutation)
            }
            
            # Créer une matrice numérique pour les couleurs
            color_matrix = []
            hover_matrix = []
            
            for gene in top_genes:
                row_colors = []
                row_hover = []
                
                for patient in all_patients:
                    mut_type = mutation_matrix.loc[gene, patient]
                    
                    if mut_type == '':
                        row_colors.append(mutation_colors[''])
                        row_hover.append(f"Patient: {patient}<br>Gène: {gene}<br>Mutation: Aucune")
                    else:
                        row_colors.append(mutation_colors[mut_type])
                        
                        # Trouver les détails des mutations
                        mutations = df_onco[(df_onco['Gene_symbol'] == gene) & (df_onco['Pseudo'] == patient)]
                        
                        details = []
                        for _, m in mutations.iterrows():
                            variant = m.get('Variant', 'N/A')
                            hgvsp = m.get('hgvs.p', 'N/A')
                            acmg = m.get('ACMG_class', 'N/A')
                            details.append(f"{variant} ({hgvsp}) - {acmg}")
                        
                        detail_str = '<br>'.join(details[:3])  # Limiter à 3 mutations
                        if len(details) > 3:
                            detail_str += f"<br>... et {len(details)-3} autres"
                        
                        row_hover.append(
                            f"Patient: {patient}<br>Gène: {gene}<br>"
                            f"Type: {mut_type}<br>{detail_str}"
                        )
                
                color_matrix.append(row_colors)
                hover_matrix.append(row_hover)
            
            # Créer une heatmap personnalisée avec des rectangles
            # (car plotly heatmap ne supporte pas les couleurs directes par cellule)
            shapes = []
            annotations_fig = []
            
            x_step = 1
            y_step = 1
            
            for i, gene in enumerate(top_genes):
                y_pos = len(top_genes) - i - 1  # Inverser pour affichage
                
                for j, patient in enumerate(all_patients):
                    x_pos = j
                    color = color_matrix[i][j]
                    
                    # Ajouter un rectangle
                    shapes.append(
                        dict(
                            type='rect',
                            x0=x_pos - 0.5, x1=x_pos + 0.5,
                            y0=y_pos - 0.5, y1=y_pos + 0.5,
                            fillcolor=color,
                            line=dict(width=0.5, color='#333'),
                        )
                    )
            
            # Ajouter trace invisible pour le hover
            for i, gene in enumerate(top_genes):
                y_pos = len(top_genes) - i - 1
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(all_patients))),
                    y=[y_pos] * len(all_patients),
                    mode='markers',
                    marker=dict(size=1, opacity=0),
                    hovertext=hover_matrix[i],
                    hoverinfo='text',
                    showlegend=False
                ))
            
            # Configuration de la figure
            fig.update_layout(
                shapes=shapes,
                xaxis=dict(
                    ticktext=all_patients,
                    tickvals=list(range(len(all_patients))),
                    tickangle=-90,
                    side='bottom',
                    showgrid=False,
                    zeroline=False,
                    title=""
                ),
                yaxis=dict(
                    ticktext=top_genes,
                    tickvals=list(range(len(top_genes))),
                    showgrid=False,
                    zeroline=False,
                    title="Gène"
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark',
                height=total_height,
                margin=dict(l=120, r=150, t=annotation_height + 80, b=150),
                title=f"OncoPrint - {len(top_genes)} gènes × {len(all_patients)} patients",
                hovermode='closest'
            )
            
            # ── ANNOTATIONS PATIENTS (au-dessus) ──
            if show_annotations:
                # Sexe
                y_annot_sex = len(top_genes) + 0.5
                for j, patient in enumerate(all_patients):
                    sex = patient_annotations[patient]['sex']
                    sex_color = '#64ffda' if 'F' in sex.upper() or 'FEMME' in sex.upper() else '#ff6b6b'
                    
                    fig.add_shape(
                        type='rect',
                        x0=j - 0.5, x1=j + 0.5,
                        y0=y_annot_sex - 0.5, y1=y_annot_sex + 0.5,
                        fillcolor=sex_color,
                        line=dict(width=0.5, color='#333')
                    )
                
                # Âge
                y_annot_age = len(top_genes) + 1.5
                for j, patient in enumerate(all_patients):
                    age = patient_annotations[patient]['age']
                    try:
                        age_val = int(age)
                        # Gradient de couleur selon l'âge
                        if age_val < 20:
                            age_color = '#4ecdc4'
                        elif age_val < 40:
                            age_color = '#ffd93d'
                        else:
                            age_color = '#ff6b6b'
                    except:
                        age_color = '#666666'
                    
                    fig.add_shape(
                        type='rect',
                        x0=j - 0.5, x1=j + 0.5,
                        y0=y_annot_age - 0.5, y1=y_annot_age + 0.5,
                        fillcolor=age_color,
                        line=dict(width=0.5, color='#333')
                    )
                
                # Histologie
                y_annot_histo = len(top_genes) + 2.5
                histo_colors = {'HV': '#9b59b6', 'mixed': '#ffa500', 'PC': '#3498db', 'Unknown': '#666666'}
                for j, patient in enumerate(all_patients):
                    histo = patient_annotations[patient]['histo']
                    histo_color = histo_colors.get(histo, '#666666')
                    
                    fig.add_shape(
                        type='rect',
                        x0=j - 0.5, x1=j + 0.5,
                        y0=y_annot_histo - 0.5, y1=y_annot_histo + 0.5,
                        fillcolor=histo_color,
                        line=dict(width=0.5, color='#333')
                    )
                
                # Complications
                y_annot_compl = len(top_genes) + 3.5
                for j, patient in enumerate(all_patients):
                    compl = patient_annotations[patient]['complications']
                    compl_color = '#ff6b6b' if compl != 'Aucune' else '#4ecdc4'
                    
                    fig.add_shape(
                        type='rect',
                        x0=j - 0.5, x1=j + 0.5,
                        y0=y_annot_compl - 0.5, y1=y_annot_compl + 0.5,
                        fillcolor=compl_color,
                        line=dict(width=0.5, color='#333')
                    )
                
                # Labels des annotations (à gauche)
                fig.add_annotation(
                    x=-1, y=y_annot_sex,
                    text="Sexe", showarrow=False,
                    xanchor='right', font=dict(size=10, color='#ccd6f6')
                )
                fig.add_annotation(
                    x=-1, y=y_annot_age,
                    text="Âge", showarrow=False,
                    xanchor='right', font=dict(size=10, color='#ccd6f6')
                )
                fig.add_annotation(
                    x=-1, y=y_annot_histo,
                    text="Histo", showarrow=False,
                    xanchor='right', font=dict(size=10, color='#ccd6f6')
                )
                fig.add_annotation(
                    x=-1, y=y_annot_compl,
                    text="Compl.", showarrow=False,
                    xanchor='right', font=dict(size=10, color='#ccd6f6')
                )
            
            # ── LÉGENDE ──
            legend_y = -0.15
            legend_items = [
                ('Frameshift', mutation_colors['Frameshift']),
                ('Nonsense', mutation_colors['Nonsense']),
                ('Splice', mutation_colors['Splice']),
                ('Missense', mutation_colors['Missense']),
                ('InDel', mutation_colors['InDel']),
                ('Other', mutation_colors['Other']),
            ]
            
            legend_html = "<br>**Légende types de mutations :** "
            for label, color in legend_items:
                legend_html += f'<span style="color:{color}">⬛ {label}</span> &nbsp; '
            
            # Afficher la figure
            st.plotly_chart(fig, width='stretch')
            
            # Légende sous le graphe
            st.markdown(legend_html, unsafe_allow_html=True)
            
            if show_annotations:
                st.markdown("""
                **Légende annotations :**  
                - **Sexe :** <span style="color:#64ffda">⬛ Femme</span> | <span style="color:#ff6b6b">⬛ Homme</span>  
                - **Âge :** <span style="color:#4ecdc4">⬛ <20 ans</span> | <span style="color:#ffd93d">⬛ 20-40 ans</span> | <span style="color:#ff6b6b">⬛ >40 ans</span>  
                - **Histo :** <span style="color:#9b59b6">⬛ HV</span> | <span style="color:#ffa500">⬛ Mixed</span> | <span style="color:#3498db">⬛ PC</span>  
                - **Compl. :** <span style="color:#ff6b6b">⬛ Présentes</span> | <span style="color:#4ecdc4">⬛ Absentes</span>
                """, unsafe_allow_html=True)
            
            # ── STATISTIQUES ──
            st.markdown("---")
            st.markdown("### 📈 Statistiques")
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                total_mutations = (mutation_type_matrix > 0).sum().sum()
                st.metric("Total mutations affichées", int(total_mutations))
            
            with col_s2:
                patients_with_mutations = (mutation_type_matrix > 0).any(axis=0).sum()
                st.metric("Patients avec ≥1 mutation", patients_with_mutations)
            
            with col_s3:
                avg_mutations_per_patient = total_mutations / len(all_patients)
                st.metric("Moy. mutations/patient", f"{avg_mutations_per_patient:.1f}")
            
            with col_s4:
                avg_mutations_per_gene = (mutation_type_matrix > 0).sum(axis=1).mean()
                st.metric("Moy. patients/gène", f"{avg_mutations_per_gene:.1f}")
            
            # ── TABLEAU DÉTAILLÉ ──
            st.markdown("### 📋 Tableau détaillé des mutations")
            
            # Créer un tableau récapitulatif
            summary_data = []
            for gene in top_genes:
                n_patients_mutated = (mutation_type_matrix.loc[gene] > 0).sum()
                pct_patients = n_patients_mutated / len(all_patients) * 100
                
                # Compter par type de mutation
                mutation_counts = mutation_matrix.loc[gene].value_counts()
                
                summary_data.append({
                    'Gène': gene,
                    'N_patients': int(n_patients_mutated),
                    '% patients': f"{pct_patients:.1f}%",
                    'Frameshift': int(mutation_counts.get('Frameshift', 0)),
                    'Nonsense': int(mutation_counts.get('Nonsense', 0)),
                    'Splice': int(mutation_counts.get('Splice', 0)),
                    'Missense': int(mutation_counts.get('Missense', 0)),
                    'InDel': int(mutation_counts.get('InDel', 0)),
                    'Other': int(mutation_counts.get('Other', 0)),
                })
            
            df_summary = pd.DataFrame(summary_data)
            
            st.dataframe(
                df_summary,
                width='stretch',
                height=min(600, len(df_summary) * 35 + 38)
            )
            
            # ── EXPORT ──
            st.markdown("---")
            st.markdown("### 📥 Export")
            
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                csv_summary = df_summary.to_csv(index=False, sep=';')
                st.download_button(
                    "📥 Télécharger résumé (CSV)",
                    csv_summary,
                    "oncoprint_summary.csv",
                    "text/csv",
                    width='stretch'
                )
            
            with col_e2:
                # Matrice complète
                mutation_matrix_export = mutation_matrix.copy()
                mutation_matrix_export.index.name = 'Gene'
                csv_matrix = mutation_matrix_export.to_csv(sep=';')
                st.download_button(
                    "📥 Télécharger matrice complète (CSV)",
                    csv_matrix,
                    "oncoprint_matrix.csv",
                    "text/csv",
                    width='stretch'
                )



# VAF & CLONALITÉ
