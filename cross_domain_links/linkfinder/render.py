"""Turn a ranked chain into a human-readable, citable explanation."""
from __future__ import annotations

from .graph import EvidenceGraph
from .paths import Chain
from .verify import chain_status

_MARK = {"yes": "✔", "partial": "~", "no": "✘"}


def render_chain(graph: EvidenceGraph, chain: Chain, rank: int | None = None) -> str:
    name = lambda nid: graph.nodes[nid].name  # noqa: E731
    head = " → ".join(name(n) for n in chain.nodes)
    p = chain.parts or {}
    lines = [
        f"{'#' + str(rank) + ' ' if rank else ''}{head}",
        f"   score {chain.score:.3g} | specificity {p.get('specificity', 0):.3g} | "
        f"evidence {p.get('evidence', 0):.3g} | domains: {', '.join(p.get('domains', []))} | "
        f"status: {chain_status(chain)}",
    ]
    if chain.earliest_complete_year:
        lines.append(f"   chain fully supported from: {chain.earliest_complete_year}")
    for i, h in enumerate(chain.hops):
        v = chain.verdicts[i] if chain.verdicts else None
        mark = f"{_MARK[v.supported]} " if v else ""
        ev = h.edge.evidence[v.evidence_index] if v and v.evidence_index >= 0 else h.edge.evidence[0]
        year = f", {ev.year}" if ev.year else ""
        tag = f"{h.edge.relation} · {h.edge.kind}"
        arrow = f"—[{tag}]→" if h.forward else f"←[{tag}]—"
        lines.append(f"   {mark}{name(h.frm)} {arrow} {name(h.to)}")
        lines.append(f"       “{ev.text}”  [{ev.source}{year}]")
        extra = len(h.edge.evidence) - 1
        if extra > 0:
            lines.append(f"       (+{extra} more source{'s' if extra > 1 else ''})")
    return "\n".join(lines)
