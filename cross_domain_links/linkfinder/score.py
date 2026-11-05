"""Ranking chains. Finding paths is easy; ranking them is the research problem.

score(chain) = specificity * evidence * domain_bonus * length_decay ** (hops - 1)

specificity and evidence are geometric means per hop, so chains of different
length compete fairly; a raw product would always favour the shortest chain,
which in practice is the one through a hub.

* specificity — degree-weighted path count (Himmelstein et al. 2017, Hetionet /
  Project Rephetio): each hop is down-weighted by (deg(u) * deg(v)) ** -w, so
  chains routed through hubs like "patients" or "protein" sink.
* evidence — per hop: kind weight (causal > mechanistic > associative >
  co-occurrence) times a saturating function of how many independent sources
  support it (1 - 0.5**n_sources).
* domain_bonus — mild preference for chains that actually cross domains, which
  is the point of the tool.
"""
from __future__ import annotations

from .graph import EDGE_KINDS, EvidenceGraph
from .paths import Chain


def hop_evidence(edge) -> float:
    n_sources = len({e.source for e in edge.evidence})
    return EDGE_KINDS[edge.kind] * (1 - 0.5 ** n_sources)


def score_chain(graph: EvidenceGraph, chain: Chain, w: float = 0.4,
                domain_bonus: float = 0.1, length_decay: float = 0.9) -> Chain:
    spec = 1.0
    ev = 1.0
    for h in chain.hops:
        spec *= (graph.degree(h.frm) * graph.degree(h.to)) ** (-w)
        ev *= hop_evidence(h.edge)
    k = len(chain.hops)
    spec, ev = spec ** (1 / k), ev ** (1 / k)
    domains = {graph.nodes[n].domain for n in chain.nodes}
    bonus = 1.0 + domain_bonus * (len(domains) - 1)
    chain.score = spec * ev * bonus * length_decay ** (k - 1)
    chain.parts = {
        "specificity": spec,  # per-hop geometric mean
        "evidence": ev,
        "domains": sorted(domains),
        "hops": len(chain.hops),
        "weakest_hop": min(chain.hops, key=lambda h: hop_evidence(h.edge)).edge.relation,
    }
    return chain


def rank(graph: EvidenceGraph, chains: list[Chain], top_k: int = 10,
         w: float = 0.4, diversify: bool = True) -> list[Chain]:
    """Score and sort. With `diversify`, keep at most one chain per bridge set:
    two chains through the same intermediate concepts are one explanation."""
    scored = sorted((score_chain(graph, c, w) for c in chains),
                    key=lambda c: c.score, reverse=True)
    if not diversify:
        return scored[:top_k]
    out, seen = [], set()
    for c in scored:
        key = frozenset(c.nodes[1:-1])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) == top_k:
            break
    return out
