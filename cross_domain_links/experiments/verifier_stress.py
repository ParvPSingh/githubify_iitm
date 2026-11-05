"""Experiment 5: how much can the offline HeuristicVerifier catch?

For every non-hub, non-analogy edge of the demo graph, build three versions:
  correct   — its real evidence sentence
  swapped   — the evidence sentence of a random other edge (misattributed citation)
  negated   — its real sentence with the claim negated ("does not ...", "no evidence that ...")
and record whether the verifier says the hop is supported ("yes").
A good verifier says yes to `correct` and no to the other two.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linkfinder.graph import Edge, Evidence, EvidenceGraph  # noqa: E402
from linkfinder.verify import HeuristicVerifier  # noqa: E402

random.seed(0)
g = EvidenceGraph.load(Path(__file__).resolve().parents[1] / "data" / "demo_graph.json")
edges = [e for e in g.edges if e.kind not in ("analogical", "cooccurrence")]


def negate(text: str) -> str:
    for pat, rep in [(r"\bcauses\b", "does not cause"), (r"\bis an?\b", "is not a"),
                     (r"\bcan cause\b", "cannot cause"), (r"\bcan reactivate\b", "cannot reactivate"),
                     (r"\bare\b", "are not"), (r"\bis\b", "is not"), (r"\buse\b", "do not use"),
                     (r"\bwas\b", "was never")]:
        new, n = re.subn(pat, rep, text, count=1)
        if n:
            return new
    return "There is no evidence that " + text[0].lower() + text[1:]


v = HeuristicVerifier()
rates = {"correct": [], "swapped": [], "negated": []}
for e in edges:
    other = random.choice([x for x in g.edges if x is not e])
    for label, text in [("correct", e.evidence[0].text),
                        ("swapped", other.evidence[0].text),
                        ("negated", negate(e.evidence[0].text))]:
        probe = Edge(e.src, e.dst, e.relation, e.kind, [Evidence(text, "probe")])
        rates[label].append(v.verify(g, probe).supported == "yes")

print(f"{len(edges)} edges")
for k, xs in rates.items():
    print(f"{k:8s} verifier says 'supported': {sum(xs)}/{len(xs)} = {sum(xs)/len(xs):.0%}")
