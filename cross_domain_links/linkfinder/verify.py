"""Hop verification: the LLM checks evidence, it does not supply it.

Every hop in a chain must be supported by its own cited text. A verifier reads
(claim, evidence sentence) and returns a verdict. Chains with an unsupported hop
are dropped or flagged — this is what keeps the LLM from inventing links.

Two verifiers:
* HeuristicVerifier — offline, free: do both endpoint names (or aliases)
  appear in the evidence text? Catches mis-attributed evidence, not much else.
* ClaudeVerifier — asks Claude whether the sentence actually states the
  relation (entailment), with structured JSON output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .graph import Edge, EvidenceGraph
from .paths import Chain


@dataclass
class Verdict:
    supported: str       # "yes" | "partial" | "no"
    rationale: str
    evidence_index: int  # which Evidence record was used (-1 if none)


class Verifier(Protocol):
    def verify(self, graph: EvidenceGraph, edge: Edge) -> Verdict: ...


def _names(graph: EvidenceGraph, node_id: str) -> list[str]:
    n = graph.nodes[node_id]
    return [s.lower() for s in [n.name, *n.aliases]]


class HeuristicVerifier:
    def verify(self, graph: EvidenceGraph, edge: Edge) -> Verdict:
        a, b = _names(graph, edge.src), _names(graph, edge.dst)
        for i, ev in enumerate(edge.evidence):
            t = ev.text.lower()
            hit_a, hit_b = any(x in t for x in a), any(x in t for x in b)
            if hit_a and hit_b:
                return Verdict("yes", "both concepts are named in the evidence text", i)
            if hit_a or hit_b:
                partial = Verdict("partial", "only one concept is named in the evidence text", i)
                return partial
        return Verdict("no", "neither concept is named in any evidence text", -1)


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "string", "enum": ["yes", "partial", "no"]},
        "evidence_index": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["supported", "evidence_index", "rationale"],
    "additionalProperties": False,
}

PROMPT = """You are checking one hop of an evidence chain for a literature-based discovery tool.

Claim: "{a}" --[{rel}]--> "{b}"   (edge kind: {kind})

Evidence records:
{evidence}

Does at least one evidence record, read literally, state or directly imply the claim?
- "yes": a record states it.
- "partial": a record supports a weaker or narrower version (for example association, not causation; or only a subtype).
- "no": no record supports it. Do not use outside knowledge to fill the gap.
Return the index of the best record (-1 if none) and a one-sentence rationale."""


class ClaudeVerifier:
    """Requires `pip install anthropic` and an API key (or `ant auth login`)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low") -> None:
        import anthropic  # imported lazily so the rest of the package has no deps

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def verify(self, graph: EvidenceGraph, edge: Edge) -> Verdict:
        import json

        evidence = "\n".join(f"[{i}] {e.text} (source: {e.source})"
                             for i, e in enumerate(edge.evidence))
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            # If a safety classifier declines, the API re-runs on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
            messages=[{"role": "user", "content": PROMPT.format(
                a=graph.nodes[edge.src].name, b=graph.nodes[edge.dst].name,
                rel=edge.relation, kind=edge.kind, evidence=evidence)}],
        )
        if response.stop_reason == "refusal":
            return Verdict("no", "verifier declined to assess this hop", -1)
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return Verdict(data["supported"], data["rationale"], data["evidence_index"])


def verify_chain(graph: EvidenceGraph, chain: Chain, verifier: Verifier) -> Chain:
    chain.verdicts = [verifier.verify(graph, h.edge) for h in chain.hops]
    return chain


def chain_status(chain: Chain) -> str:
    """'verified' if every hop is supported, 'weak' if any hop is partial,
    'broken' if any hop is unsupported."""
    if not chain.verdicts:
        return "unverified"
    s = [v.supported for v in chain.verdicts]
    if "no" in s:
        return "broken"
    if "partial" in s:
        return "weak"
    return "verified"
