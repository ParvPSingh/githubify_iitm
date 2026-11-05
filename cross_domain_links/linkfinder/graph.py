"""Evidence graph: a knowledge graph where every edge carries its own provenance.

The design rule is simple: an edge without at least one Evidence record is not
allowed in. That is what lets a discovered chain "stand on its own" — every hop
can be traced back to a sentence, a record, or a document.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# How much a hop of each kind counts as evidence. "analogical" hops are kept in
# the graph (they are useful for idea generation) but never count as evidence.
EDGE_KINDS = {
    "causal": 1.0,        # A causes / treats / prevents C (stated by the source)
    "mechanistic": 0.9,   # part-of, contains, derived-from, is-a: structural facts
    "associative": 0.7,   # statistically associated, risk factor
    "cooccurrence": 0.3,  # mentioned in the same sentence; weakest signal
    "analogical": 0.0,    # shared abstract structure; a hint, not evidence
}


@dataclass
class Evidence:
    text: str                  # the sentence or statement that supports the edge
    source: str                # citation: PMID, DOI, report name, dataset record
    year: int | None = None    # when it became public; used for time-sliced eval
    url: str | None = None


@dataclass
class Node:
    id: str
    name: str
    type: str = "Concept"      # e.g. Disease, Chemical, Virus, Product, Component
    domain: str = "general"    # e.g. medicine, virology, automotive
    aliases: list[str] = field(default_factory=list)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    kind: str
    evidence: list[Evidence]

    def __post_init__(self) -> None:
        if self.kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind {self.kind!r}; use one of {sorted(EDGE_KINDS)}")
        if not self.evidence:
            raise ValueError(f"edge {self.src} -[{self.relation}]-> {self.dst} has no evidence")

    @property
    def earliest_year(self) -> int | None:
        years = [e.year for e in self.evidence if e.year is not None]
        return min(years) if years else None


class EvidenceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._adj: dict[str, list[Edge]] = {}
        self._edge_index: dict[tuple[str, str, str], Edge] = {}

    # ----- construction -------------------------------------------------
    def add_node(self, node: Node) -> Node:
        existing = self.nodes.get(node.id)
        if existing:
            for a in node.aliases:
                if a not in existing.aliases:
                    existing.aliases.append(a)
            return existing
        self.nodes[node.id] = node
        self._adj.setdefault(node.id, [])
        return node

    def add_edge(self, edge: Edge) -> Edge:
        for nid in (edge.src, edge.dst):
            if nid not in self.nodes:
                raise KeyError(f"edge references unknown node {nid!r}")
        key = (edge.src, edge.relation, edge.dst)
        existing = self._edge_index.get(key)
        if existing:  # merge evidence instead of duplicating the edge
            seen = {(e.source, e.text) for e in existing.evidence}
            existing.evidence.extend(e for e in edge.evidence if (e.source, e.text) not in seen)
            return existing
        self._edge_index[key] = edge
        self.edges.append(edge)
        self._adj[edge.src].append(edge)
        if edge.dst != edge.src:
            self._adj[edge.dst].append(edge)
        return edge

    # ----- queries ------------------------------------------------------
    def incident(self, node_id: str) -> list[Edge]:
        return self._adj.get(node_id, [])

    def degree(self, node_id: str) -> int:
        return len(self._adj.get(node_id, []))

    def resolve(self, name: str) -> str:
        """Map a user-typed name to a node id (id, name or alias; case-insensitive)."""
        q = name.strip().lower()
        for n in self.nodes.values():
            if q == n.id.lower() or q == n.name.lower() or q in (a.lower() for a in n.aliases):
                return n.id
        raise KeyError(f"no node matches {name!r}")

    def until(self, year: int) -> "EvidenceGraph":
        """Snapshot of the graph using only evidence published up to `year`.

        This is the basis for time-sliced evaluation: discover on the snapshot,
        then check whether the chain was later confirmed.
        """
        g = EvidenceGraph()
        for n in self.nodes.values():
            g.add_node(Node(**asdict(n)))
        for e in self.edges:
            ev = [x for x in e.evidence if x.year is not None and x.year <= year]
            if ev:
                g.add_edge(Edge(e.src, e.dst, e.relation, e.kind, ev))
        return g

    # ----- (de)serialisation -------------------------------------------
    def to_json(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [
                {**{k: v for k, v in asdict(e).items() if k != "evidence"},
                 "evidence": [asdict(x) for x in e.evidence]}
                for e in self.edges
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "EvidenceGraph":
        g = cls()
        for n in data["nodes"]:
            g.add_node(Node(**n))
        for e in data["edges"]:
            g.add_edge(Edge(e["src"], e["dst"], e["relation"], e["kind"],
                            [Evidence(**x) for x in e["evidence"]]))
        return g

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceGraph":
        return cls.from_json(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    def merge(self, others: Iterable["EvidenceGraph"]) -> "EvidenceGraph":
        for o in others:
            for n in o.nodes.values():
                self.add_node(Node(**asdict(n)))
            for e in o.edges:
                self.add_edge(Edge(e.src, e.dst, e.relation, e.kind, list(e.evidence)))
        return self
