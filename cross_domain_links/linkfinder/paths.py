"""Candidate chain generation between two concepts (closed discovery, A ... C)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .graph import Edge, EvidenceGraph


@dataclass
class Hop:
    edge: Edge
    frm: str   # node we came from
    to: str    # node we went to

    @property
    def forward(self) -> bool:
        """True if the hop follows the edge's stated direction."""
        return self.edge.src == self.frm


@dataclass
class Chain:
    hops: list[Hop]
    score: float = 0.0
    parts: dict | None = None
    verdicts: list | None = None

    @property
    def nodes(self) -> list[str]:
        return [self.hops[0].frm] + [h.to for h in self.hops]

    @property
    def earliest_complete_year(self) -> int | None:
        """First year in which every hop had at least one piece of evidence."""
        years = [h.edge.earliest_year for h in self.hops]
        return None if any(y is None for y in years) else max(years)


def _distances_to(graph: EvidenceGraph, target: str, limit: int, allowed) -> dict[str, int]:
    dist = {target: 0}
    q = deque([target])
    while q:
        u = q.popleft()
        if dist[u] >= limit:
            continue
        for e in graph.incident(u):
            if not allowed(e):
                continue
            v = e.dst if e.src == u else e.src
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def find_chains(
    graph: EvidenceGraph,
    source: str,
    target: str,
    max_hops: int = 5,
    include_analogical: bool = False,
    max_chains: int = 5000,
) -> list[Chain]:
    """Enumerate simple paths source -> target of at most `max_hops` edges.

    Edges are traversed in either direction (a chain is an explanation, not a
    directed causal claim); each Hop records whether it went with or against the
    edge's direction so the explanation can be phrased correctly.
    A backwards BFS from the target prunes branches that cannot reach it in time.
    """
    def allowed(e: Edge) -> bool:
        return include_analogical or e.kind != "analogical"

    dist = _distances_to(graph, target, max_hops, allowed)
    if source not in dist:
        return []

    chains: list[Chain] = []
    path: list[Hop] = []
    visited = {source}

    def dfs(u: str) -> None:
        if len(chains) >= max_chains:
            return
        if u == target:
            chains.append(Chain(hops=list(path)))
            return
        remaining = max_hops - len(path)
        for e in graph.incident(u):
            if not allowed(e):
                continue
            v = e.dst if e.src == u else e.src
            if v in visited or dist.get(v, max_hops + 1) > remaining - 1:
                continue
            visited.add(v)
            path.append(Hop(e, u, v))
            dfs(v)
            path.pop()
            visited.discard(v)

    dfs(source)
    return chains
