"""Construction du prompt IA et appel API Anthropic pour interprétation des clusters."""
from core.config import FEATURE_LABELS

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

def build_ai_prompt(interpretations, gene_signatures, n_clusters):
    prompt = """Tu es un expert en génétique clinique spécialisé dans les variants somatiques 
issus de séquençage ciblé FFPE (thymomes / pathologies thymiques / maladies auto-immunes).

Méthodologie : variants filtrés par qualité (profondeur, AR, gnomAD NFE), exclusion des 
synonymes/introniques/UTR. Features en proportions (%) + scores d'impact composites + 
analyse de pathways (MSigDB). Gènes en features binaires.

"""
    for cid, interp in interpretations.items():
        prompt += f"\n{'='*50}\n## {cid} ({interp['n_patients']} patients: {', '.join(interp['patients'])})\n"
        prompt += "### Enrichi:\n"
        for f, z in interp['significant'].items():
            if z > 0:
                label = FEATURE_LABELS.get(f, f)
                prompt += f"  - {label}: {interp['cluster_mean'][f]:.2f} (glob: {interp['global_mean'][f]:.2f}, z={z:.2f})\n"
        prompt += "### Réduit:\n"
        for f, z in interp['significant'].items():
            if z < 0:
                label = FEATURE_LABELS.get(f, f)
                prompt += f"  - {label}: {interp['cluster_mean'][f]:.2f} (glob: {interp['global_mean'][f]:.2f}, z={z:.2f})\n"
        if cid in gene_signatures:
            gs = gene_signatures[cid]
            if len(gs["pathogenic_genes"]) > 0:
                prompt += "### Gènes pathogènes:\n"
                for g, c in gs["pathogenic_genes"].head(5).items(): prompt += f"  - {g}: {c}\n"
            if len(gs["enriched_genes"]) > 0:
                prompt += "### Gènes enrichis:\n"
                for g, r in gs["enriched_genes"].head(5).items(): prompt += f"  - {g}: x{r:.1f}\n"

    prompt += f"""
{'='*50}
Pour chaque cluster: 1) Signature dominante 2) Interprétation clinique (thymome, auto-immunité)
3) Gènes/pathways d'intérêt 4) Comparaison inter-clusters. Termine par une synthèse globale.
Réponds en français, sois précis et cliniquement pertinent."""
    return prompt


def call_anthropic_api(prompt, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=4000,
                                  messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text

