"""
Clients API pour MyVariant.info, STRING et Monarch (HPO).

Cache a deux niveaux :
- Cache disque persistant dans .api_cache/ (garde entre sessions locales)
- Cache session Streamlit en fallback (perdu au reboot)

Sur Streamlit Cloud, le cache disque est ephemere (reset a chaque
redeploiement) mais le cache session compense pendant l'utilisation.
"""
import json
import time
import hashlib
from pathlib import Path

import requests
import streamlit as st


# ─────────────────────────────────────────────
# CACHE A DEUX NIVEAUX
# ─────────────────────────────────────────────
CACHE_DIR = Path(".api_cache")


def _cache_path(name):
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _load_disk_cache(name):
    path = _cache_path(name)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_disk_cache(name, data):
    try:
        with open(_cache_path(name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        # Streamlit Cloud peut bloquer l'ecriture - on ignore
        pass


def _get_session_cache(name):
    """Cache session (initialise depuis le disque a la premiere utilisation)."""
    key = f"_api_cache_{name}"
    if key not in st.session_state:
        st.session_state[key] = _load_disk_cache(name)
    return st.session_state[key]


def _persist_session_cache(name):
    cache = _get_session_cache(name)
    _save_disk_cache(name, cache)


def clear_api_cache(name=None):
    """Vide le cache (disque + session). Si name est None, vide tout."""
    if name is None:
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                try:
                    f.unlink()
                except OSError:
                    pass
        keys = [k for k in st.session_state.keys() if k.startswith("_api_cache_")]
        for k in keys:
            del st.session_state[k]
    else:
        try:
            _cache_path(name).unlink(missing_ok=True)
        except OSError:
            pass
        key = f"_api_cache_{name}"
        if key in st.session_state:
            del st.session_state[key]


# ─────────────────────────────────────────────
# CONVERSION VARIANT -> HGVS GENOMIC
# ─────────────────────────────────────────────
def parse_variant_to_hgvs_g(variant_str):
    """
    Convertit le format `1:120572547:T>C` en HGVS genomic `chr1:g.120572547T>C`
    accepte par MyVariant.info.

    Pour l'instant, ne gere que les SNVs (substitutions simples).
    Indels et complexes -> None (ils ne seront pas annotes).
    """
    if not isinstance(variant_str, str):
        return None
    parts = variant_str.split(":")
    if len(parts) != 3:
        return None
    chrom, pos, alleles = parts
    chrom = chrom.replace("chr", "")
    if ">" not in alleles:
        return None
    ref, _, alt = alleles.partition(">")
    if not ref or not alt:
        return None
    if len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT":
        try:
            int(pos)
        except ValueError:
            return None
        return f"chr{chrom}:g.{pos}{ref}>{alt}"
    return None


# ─────────────────────────────────────────────
# MYVARIANT.INFO
# ─────────────────────────────────────────────
MYVARIANT_URL = "https://myvariant.info/v1/variant"

MYVARIANT_FIELDS = ",".join([
    "dbnsfp.revel.score",
    "dbnsfp.revel.rankscore",
    "dbnsfp.alphamissense.score",
    "dbnsfp.alphamissense.pred",
    "dbnsfp.metasvm.pred",
    "dbnsfp.metasvm.score",
    "dbnsfp.metalr.pred",
    "dbnsfp.metalr.score",
    "dbnsfp.polyphen2.hdiv.score",
    "dbnsfp.polyphen2.hdiv.pred",
    "dbnsfp.sift.score",
    "dbnsfp.sift.pred",
    "dbnsfp.cadd.phred",
    "dbnsfp.gnomad_exomes.af",
    "dbnsfp.gnomad_genomes.af",
    "dbnsfp.spliceai.ds_ag",
    "dbnsfp.spliceai.ds_al",
    "dbnsfp.spliceai.ds_dg",
    "dbnsfp.spliceai.ds_dl",
    "clinvar.rcv.clinical_significance",
])


def query_myvariant_batch(hgvs_ids, assembly="hg19", batch_size=800,
                          progress_callback=None):
    """
    Interroge MyVariant.info en batch sur une liste d'identifiants HGVS.
    Retourne {hgvs_id: dict_resultat}.
    """
    cache = _get_session_cache(f"myvariant_{assembly}")
    results = {}
    to_fetch = []

    for hgvs in hgvs_ids:
        if hgvs in cache:
            results[hgvs] = cache[hgvs]
        else:
            to_fetch.append(hgvs)

    if not to_fetch:
        return results

    n_batches = (len(to_fetch) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        batch = to_fetch[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        try:
            resp = requests.post(
                MYVARIANT_URL,
                data={
                    "ids": ",".join(batch),
                    "fields": MYVARIANT_FIELDS,
                    "assembly": assembly,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            for hgvs in batch:
                results[hgvs] = {"_error": str(e)}
                cache[hgvs] = {"_error": str(e)}
            if progress_callback:
                progress_callback((batch_idx + 1) / n_batches)
            continue

        for item in data:
            qid = item.get("query")
            if not qid:
                continue
            if item.get("notfound"):
                payload = {}
            else:
                payload = {k: v for k, v in item.items()
                           if k not in ("query", "_id", "_score")}
            results[qid] = payload
            cache[qid] = payload

        if progress_callback:
            progress_callback((batch_idx + 1) / n_batches)
        time.sleep(0.1)

    _persist_session_cache(f"myvariant_{assembly}")
    return results


def extract_myvariant_field(payload, *keys, default=None):
    """Extrait un champ imbrique : extract(p, 'dbnsfp', 'revel', 'score')."""
    cur = payload
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    if isinstance(cur, list):
        nums = [x for x in cur if isinstance(x, (int, float))]
        if nums:
            return max(nums)
        strs = [x for x in cur if isinstance(x, str)]
        if strs:
            return ";".join(sorted(set(strs)))
        return default
    return cur


# ─────────────────────────────────────────────
# STRING-DB
# ─────────────────────────────────────────────
STRING_BASE = "https://string-db.org/api"


def query_string_network(genes, required_score=400, species=9606):
    """
    Reseau d'interactions STRING pour une liste de symboles HGNC.
    required_score : 0-1000 (150=low, 400=medium, 700=high, 900=highest).
    Retourne {"nodes": [...], "edges": [...]}.
    """
    if not genes:
        return {"nodes": [], "edges": []}

    cache = _get_session_cache("string")
    cache_key = hashlib.sha256(
        f"{sorted(set(genes))}|{required_score}|{species}".encode()
    ).hexdigest()

    if cache_key in cache:
        return cache[cache_key]

    # 1. Mapping symboles -> STRING IDs
    try:
        resp_map = requests.post(
            f"{STRING_BASE}/json/get_string_ids",
            data={
                "identifiers": "\r".join(genes),
                "species": species,
                "limit": 1,
                "echo_query": 1,
            },
            timeout=30,
        )
        resp_map.raise_for_status()
        mapping = resp_map.json()
    except (requests.RequestException, ValueError) as e:
        return {"nodes": [], "edges": [], "_error": f"Mapping STRING echoue: {e}"}

    if not mapping:
        return {"nodes": [], "edges": [], "_error": "Aucun gene mappe sur STRING."}

    string_ids = [m["stringId"] for m in mapping if m.get("stringId")]
    pref_names = {m["stringId"]: m.get("preferredName", m.get("queryItem"))
                  for m in mapping if m.get("stringId")}

    # 2. Reseau
    try:
        resp_net = requests.post(
            f"{STRING_BASE}/json/network",
            data={
                "identifiers": "\r".join(string_ids),
                "species": species,
                "required_score": required_score,
            },
            timeout=60,
        )
        resp_net.raise_for_status()
        edges_raw = resp_net.json()
    except (requests.RequestException, ValueError) as e:
        return {"nodes": [], "edges": [], "_error": f"Network STRING echoue: {e}"}

    nodes_seen = set()
    nodes = []
    edges = []
    for e in edges_raw:
        a = e.get("preferredName_A")
        b = e.get("preferredName_B")
        score = e.get("score", 0)
        if a and a not in nodes_seen:
            nodes_seen.add(a)
            nodes.append({"id": a, "label": a})
        if b and b not in nodes_seen:
            nodes_seen.add(b)
            nodes.append({"id": b, "label": b})
        if a and b:
            edges.append({"source": a, "target": b, "score": float(score)})

    # Genes mappes mais isoles (pas d'interaction au-dessus du seuil)
    for stid, name in pref_names.items():
        if name and name not in nodes_seen:
            nodes_seen.add(name)
            nodes.append({"id": name, "label": name})

    result = {"nodes": nodes, "edges": edges}
    cache[cache_key] = result
    _persist_session_cache("string")
    return result


def get_string_enrichment(genes, species=9606):
    """Enrichissement fonctionnel STRING (GO, KEGG, Reactome, etc.)."""
    if not genes:
        return []

    cache = _get_session_cache("string_enrich")
    cache_key = hashlib.sha256(
        f"{sorted(set(genes))}|{species}".encode()
    ).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    try:
        resp = requests.post(
            f"{STRING_BASE}/json/enrichment",
            data={
                "identifiers": "\r".join(genes),
                "species": species,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    cache[cache_key] = data
    _persist_session_cache("string_enrich")
    return data


# ─────────────────────────────────────────────
# MONARCH (HPO + maladies)
# ─────────────────────────────────────────────
MONARCH_BASE = "https://api.monarchinitiative.org/v3/api"


def query_monarch_gene_phenotypes(gene_symbol):
    """Phenotypes HPO et maladies pour un gene (symbole HGNC humain)."""
    cache = _get_session_cache("monarch_hpo")
    if gene_symbol in cache:
        return cache[gene_symbol]

    result = {"hgnc_id": None, "phenotypes": [], "diseases": []}

    # 1. Search
    try:
        resp = requests.get(
            f"{MONARCH_BASE}/search",
            params={"q": gene_symbol, "category": "biolink:Gene", "limit": 5},
            timeout=20,
        )
        resp.raise_for_status()
        search_data = resp.json()
    except (requests.RequestException, ValueError):
        cache[gene_symbol] = result
        return result

    items = search_data.get("items", [])
    matched = None
    for it in items:
        if it.get("symbol") == gene_symbol or it.get("name") == gene_symbol:
            matched = it
            break
    if not matched and items:
        matched = items[0]
    if not matched:
        cache[gene_symbol] = result
        return result

    gene_id = matched.get("id")
    result["hgnc_id"] = gene_id

    # 2. Phenotypes
    try:
        resp_p = requests.get(
            f"{MONARCH_BASE}/entity/{gene_id}/biolink:GeneToPhenotypicFeatureAssociation",
            params={"limit": 200, "offset": 0},
            timeout=30,
        )
        resp_p.raise_for_status()
        pheno_data = resp_p.json()
        for a in pheno_data.get("items", []):
            obj = a.get("object_label") or a.get("object")
            obj_id = a.get("object")
            if obj and obj_id:
                result["phenotypes"].append({"hpo_id": obj_id, "label": obj})
    except (requests.RequestException, ValueError):
        pass

    # 3. Maladies
    try:
        resp_d = requests.get(
            f"{MONARCH_BASE}/entity/{gene_id}/biolink:CausalGeneToDiseaseAssociation",
            params={"limit": 100, "offset": 0},
            timeout=30,
        )
        resp_d.raise_for_status()
        dis_data = resp_d.json()
        for a in dis_data.get("items", []):
            lbl = a.get("object_label")
            iden = a.get("object")
            if lbl and iden:
                result["diseases"].append({"id": iden, "label": lbl})
    except (requests.RequestException, ValueError):
        pass

    cache[gene_symbol] = result
    _persist_session_cache("monarch_hpo")
    time.sleep(0.05)
    return result


def query_monarch_genes_batch(gene_symbols, progress_callback=None):
    """Itere sur une liste de genes avec progress callback."""
    out = {}
    genes = list(gene_symbols)
    n = len(genes)
    for i, g in enumerate(genes):
        out[g] = query_monarch_gene_phenotypes(g)
        if progress_callback:
            progress_callback((i + 1) / max(n, 1))
    return out
