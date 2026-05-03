# ATLASV2 — Etape 2 : ajout des nouveaux onglets

## Nouveautes

4 nouveaux onglets ajoutes :

- **🧮 Clustering** — pipeline complet KMeans/Agglomerative + UMAP + interpretation + PDF + IA optionnelle. Stocke les labels dans `st.session_state` pour reutilisation.
- **🔮 Predicteurs** — annotation a la demande via [MyVariant.info](https://myvariant.info) (REVEL, AlphaMissense, SpliceAI, MetaSVM/MetaLR, PolyPhen2, SIFT, CADD, ClinVar). GRCh37/hg19. Detecte les VUS reclassifiables.
- **🕸️ Reseau STRING** — interactions proteiques via [STRING-DB](https://string-db.org). 3 modes : cluster (reutilise l'onglet Clustering), patient, manuel. + enrichissement fonctionnel GO/KEGG/Reactome.
- **🩺 HPO** — phenotypes et maladies associees aux genes via [Monarch Initiative](https://monarchinitiative.org).

## Fichiers a uploader sur GitHub

### Nouveaux fichiers (5)
- `core/api_clients.py` — clients HTTP avec cache disque + session
- `tabs/clustering.py`
- `tabs/predictors.py`
- `tabs/network.py`
- `tabs/hpo.py`

### Fichiers a remplacer (2)
- `app.py` — nouvelle version avec 10 onglets
- `requirements.txt` — ajout de `requests`

## Cache API

Les appels API sont caches a deux niveaux :

1. **Disque** — fichier JSON dans `.api_cache/` (persistant entre sessions locales)
2. **Session Streamlit** — fallback si l'ecriture disque echoue (Streamlit Cloud)

Sur Streamlit Cloud, le cache disque est ephemere (reset a chaque redeploiement) mais le cache session compense pendant l'utilisation.

Chaque onglet a un bouton **🗑️ Vider le cache** pour relancer une requete fraiche.

## Workflow Clustering -> STRING

Pour utiliser le mode "cluster" de l'onglet STRING :
1. Aller sur l'onglet **🧮 Clustering**
2. Configurer (k=3 par defaut), cliquer **Lancer le clustering**
3. Aller sur l'onglet **🕸️ Reseau STRING**
4. Choisir le mode **Cluster** -> les labels du clustering apparaissent dans le selecteur

## Limites connues

- **Predicteurs** : seuls les SNVs (substitutions simples) sont annotes pour le moment. Les indels et variants complexes sont ignores.
- **STRING** : capped a 50 genes par requete pour rester lisible (warning si depassement, troncature aux 50 premiers).
- **HPO** : sur grosse cohorte (>100 genes), premier appel = ~30-60s. Les suivants utilisent le cache.

## Test rapide local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Si tout fonctionne en local mais pas sur Streamlit Cloud, verifier :
- `requirements.txt` est bien a la racine et inclut `requests`
- Pas de probleme de connexion sortante (les 3 APIs sont publiques mais Streamlit Cloud doit pouvoir les joindre — normalement aucun blocage)
