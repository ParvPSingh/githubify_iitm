"""Experiment 4: truly cross-domain pairs on ConceptNet 5.7 (English part).

ConceptNet is a general-knowledge graph (cars, diseases, music, animals ...)
where each edge records its source dataset and often a natural-language
"surfaceText", which we use as the hop's evidence. For each pair we cut out the
subgraph of all chains of up to 3 hops, load it into linkfinder, and compare the
ranking with and without the hub penalty.

Usage:  python3 experiments/conceptnet_crossdomain.py conceptnet_en.tsv [out_dir]
Build conceptnet_en.tsv from
  https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
keeping rows whose start and end both begin with /c/en/.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linkfinder import find_chains, rank  # noqa: E402
from linkfinder.graph import EDGE_KINDS, Edge, Evidence, EvidenceGraph, Node  # noqa: E402
from linkfinder.render import render_chain  # noqa: E402
from linkfinder.verify import HeuristicVerifier, verify_chain  # noqa: E402

KIND = {
    "Causes": "causal", "HasPrerequisite": "causal", "Entails": "causal", "CausesDesire": "associative",
    "PartOf": "mechanistic", "HasA": "mechanistic", "MadeOf": "mechanistic", "IsA": "mechanistic",
    "UsedFor": "mechanistic", "CapableOf": "mechanistic", "HasProperty": "mechanistic",
    "AtLocation": "mechanistic", "ReceivesAction": "mechanistic", "HasSubevent": "mechanistic",
    "HasFirstSubevent": "mechanistic", "HasLastSubevent": "mechanistic", "CreatedBy": "mechanistic",
    "MannerOf": "mechanistic", "LocatedNear": "mechanistic", "SymbolOf": "associative",
    "RelatedTo": "cooccurrence", "HasContext": "associative", "Desires": "associative",
    "MotivatedByGoal": "associative", "NotDesires": "associative",
    "SimilarTo": "analogical",
}  # Synonym, Antonym, DerivedFrom, FormOf, Etymology..., DistinctFrom, ExternalURL dropped (lexical)

PAIRS = [("cancer", "car"), ("cancer", "chickenpox"), ("music", "mathematics"),
         ("spider", "bridge"), ("bird", "airplane"), ("coffee", "sleep")]


def concept(uri: str) -> str:
    return uri.split("/")[3]


def load(path):
    t = time.time()
    adj = defaultdict(dict)  # u -> v -> (rel, n_sources, text, dataset, forward)
    for line in open(path, encoding="utf-8"):
        _, rel, a, b, meta = line.rstrip("\n").split("\t")
        rel = rel.split("/")[2]
        if rel not in KIND:
            continue
        a, b = concept(a), concept(b)
        if a == b:
            continue
        m = json.loads(meta)
        n = len(m.get("sources", [])) or 1
        text = m.get("surfaceText") or f"[[{a.replace('_', ' ')}]] {rel} [[{b.replace('_', ' ')}]]"
        text = text.replace("[[", "").replace("]]", "")
        cur = adj[a].get(b)
        if cur is None or EDGE_KINDS[KIND[rel]] > EDGE_KINDS[KIND[cur[0]]]:
            adj[a][b] = (rel, n, text, m.get("dataset", "?"), True)
            adj[b][a] = (rel, n, text, m.get("dataset", "?"), False)
    print(f"loaded {len(adj)} concepts, {sum(map(len, adj.values()))//2} edges in {time.time()-t:.0f}s")
    return adj


def subgraph(adj, s, t, use_related=True):
    """Nodes on any <=3-hop chain s..t, as a linkfinder EvidenceGraph. Degrees are
    taken from the full graph so hub penalties stay honest."""
    ok = (lambda r: True) if use_related else (lambda r: r != "RelatedTo")
    keep_edges = set()
    Ns, Nt = adj[s], adj[t]
    for u, e1 in Ns.items():
        if not ok(e1[0]):
            continue
        if u == t:
            keep_edges.add((s, t))
        if u in Nt and ok(Nt[u][0]):
            keep_edges |= {(s, u), (u, t)}
        Au = adj[u]
        small, big = (Au, Nt) if len(Au) < len(Nt) else (Nt, Au)
        for v in small:
            if v in big and v not in (s, t, u) and ok(Au[v][0]) and ok(Nt[v][0]):
                keep_edges |= {(s, u), (u, v), (v, t)}
    g = EvidenceGraph()
    for a, b in keep_edges:
        for x in (a, b):
            g.add_node(Node(x, x.replace("_", " "), "Concept", "conceptnet"))
        rel, n, text, ds, fwd = adj[a][b]
        src, dst = (a, b) if fwd else (b, a)
        g.add_edge(Edge(src, dst, rel, KIND[rel],
                        [Evidence(text, f"ConceptNet {ds}#{i}", None) for i in range(n)]))
    return g


class FullDegree:
    """Wrap a subgraph so degree() reports the node's degree in the full graph."""

    def __init__(self, g, adj):
        self.g, self.adj = g, adj

    def __getattr__(self, k):
        return getattr(self.g, k)

    def degree(self, node_id):
        return len(self.adj[node_id])


def main(path, out_dir):
    adj = load(path)
    out = {}
    report = []
    for a, c in PAIRS:
        for use_related in (True, False):
            g = subgraph(adj, a, c, use_related)
            if a not in g.nodes:
                continue
            chains = find_chains(g, a, c, max_hops=3)
            G = FullDegree(g, adj)
            top = [verify_chain(G, ch, HeuristicVerifier()) for ch in rank(G, chains, top_k=5)]
            nohub = rank(G, find_chains(g, a, c, max_hops=3), top_k=5, w=0.0)
            label = f"{a} ⇄ {c}" + ("" if use_related else " (RelatedTo removed)")
            hub_deg = lambda ch: max((len(adj[n]) for n in ch.nodes[1:-1]), default=0)  # noqa: E731
            out[label] = {
                "n_chains": len(chains),
                "top_by_linkfinder": [" → ".join(ch.nodes) + f"  [{', '.join(h.edge.relation for h in ch.hops)}]" for ch in top],
                "top_by_no_hub_penalty": [" → ".join(ch.nodes) for ch in nohub],
                "max_intermediate_degree_linkfinder": [hub_deg(ch) for ch in top],
                "max_intermediate_degree_no_hub_penalty": [hub_deg(ch) for ch in nohub],
            }
            report.append(f"===== {label}: {len(chains)} chains ≤3 hops =====")
            for i, ch in enumerate(top[:3], 1):
                report.append(render_chain(G, ch, i))
            report.append("-- without hub penalty, top 3: " + " | ".join(" → ".join(ch.nodes) for ch in nohub[:3]))
            report.append("")
    text = "\n".join(report)
    print(text)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "conceptnet_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (Path(out_dir) / "conceptnet_chains.txt").write_text(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "experiments/results")
