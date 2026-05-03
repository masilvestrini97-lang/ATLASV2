# ATLASV2 — Refactoring (Étape 1)

## Structure

```
ATLASV2/
├── app.py                       # Point d'entrée — orchestration uniquement (51 lignes)
├── core/
│   ├── config.py                # Constantes (IMPACT_*, ACMG_*, FEATURE_LABELS, NON_FUNCTIONAL_EFFECTS, CLINICAL_COLS)
│   ├── styling.py               # set_page_config + CSS global
│   ├── data_loader.py           # load_data, classify_acmg, compute_impact_score, load_gmt*, get_relevant_pathways
│   ├── stats.py                 # compute_cooccurrence_matrix, compute_pairwise_fisher
│   ├── features.py              # build_patient_features (matrice patient × features)
│   ├── clustering.py            # compute_cluster_interpretation, get_gene_signature_per_cluster
│   ├── ai.py                    # build_ai_prompt, call_anthropic_api
│   └── reports.py               # generate_cluster_report, generate_complications_report (PDF)
├── ui/
│   └── sidebar.py               # render_sidebar() — upload, GMT, API key, filtres globaux
└── tabs/
    ├── overview.py              # 📊 Vue d'ensemble
    ├── patient.py               # 👤 Patient
    ├── acmg.py                  # 🏷️ ACMG
    ├── oncoprint.py             # 🧬 OncoPrint
    ├── complications.py         # 🎯 Complications
    └── qc.py                    # ⚖️ Homogénéité
```

## Utilisation

Identique à avant :
```bash
streamlit run app.py
```

## Garanties du refactoring

- **Aucun changement de comportement** : c'est un pur split par modules. Les fonctions, la logique, les widgets Streamlit, les widgets UI, les tabs et leurs sous-tabs sont identiques au fichier original.
- **Tous les imports résolvent** (vérifié par `pyflakes` + import chain test).
- **Aucun nom indéfini** introduit par le refactoring.

## Notes

Ton code source contenait quelques fonctions définies mais jamais appelées (build_patient_features, compute_cooccurrence_matrix, build_ai_prompt, call_anthropic_api, generate_cluster_report, compute_cluster_interpretation, get_gene_signature_per_cluster). Je les ai conservées dans `core/` pour qu'elles soient prêtes à l'emploi quand tu en auras besoin (notamment pour les onglets futurs : prédicteurs in silico, STRING, HPO).

Quelques warnings mineurs subsistent (variables locales non utilisées, f-strings sans placeholders, imports reportlab non utilisés dans `core/reports.py`) — ils étaient déjà dans le code original et ne sont pas des régressions.

## Validation suggérée

1. Lance `streamlit run app.py` avec ton CSV habituel
2. Parcours les 6 onglets pour vérifier qu'ils s'affichent correctement
3. Teste : génération d'un rapport PDF complications, l'analyse OncoPrint, les filtres globaux
4. Si tout est OK → on passe à l'**étape 2** : ajout des onglets MyVariant.info (REVEL/AlphaMissense), STRING (3 modes), HPO (Monarch)

## Prochaines étapes

Une fois cette étape validée, on ajoutera 3 nouveaux modules :
- `tabs/predictors.py` (option 5 — REVEL, AlphaMissense, SpliceAI via MyVariant.info, GRCh37/hg19)
- `tabs/network.py` (option 9 — STRING : modes cluster / patient / sélection)
- `tabs/hpo.py` (option 11 — Monarch Initiative API)

Avec un `core/api_clients.py` pour les appels HTTP avec cache local.
