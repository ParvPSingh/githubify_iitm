"""CLI.

  python -m linkfinder demo cancer chickenpox
  python -m linkfinder demo cancer automobile --max-hops 5
  python -m linkfinder demo cancer automobile --until 2000      # time-sliced snapshot
  python -m linkfinder demo cancer automobile --analogical      # include analogy hops
  python -m linkfinder demo cancer chickenpox --verifier claude # LLM hop checking
  python -m linkfinder pubtator "fish oil" "Raynaud disease" --save fish.json
  python -m linkfinder graph fish.json "fish oil" "Raynaud disease"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import discover
from .graph import EvidenceGraph
from .paths import find_chains
from .render import render_chain
from .verify import ClaudeVerifier, HeuristicVerifier, chain_status

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo_graph.json"


def main() -> None:
    ap = argparse.ArgumentParser(prog="linkfinder")
    ap.add_argument("mode", choices=["demo", "graph", "pubtator"])
    ap.add_argument("args", nargs="+", help="[graph.json] A C")
    ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--until", type=int, help="use only evidence published up to this year")
    ap.add_argument("--analogical", action="store_true", help="allow analogy hops")
    ap.add_argument("--verifier", choices=["heuristic", "claude"], default="heuristic")
    ap.add_argument("--show-broken", action="store_true")
    ap.add_argument("--save", help="write the built graph to this JSON file")
    ns = ap.parse_args()

    if ns.mode == "demo":
        graph, (a, c) = EvidenceGraph.load(DEMO), ns.args[-2:]
    elif ns.mode == "graph":
        graph, (a, c) = EvidenceGraph.load(ns.args[0]), ns.args[1:3]
    else:
        from .sources.pubtator import build_for_pair
        a, c = ns.args[-2:]
        graph = build_for_pair(a, c)
        print(f"built graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    if ns.save:
        graph.save(ns.save)
    if ns.until:
        graph = graph.until(ns.until)

    verifier = ClaudeVerifier() if ns.verifier == "claude" else HeuristicVerifier()
    chains = discover(graph, a, c, max_hops=ns.max_hops, top_k=ns.top, verifier=verifier)
    shown = [ch for ch in chains if ns.show_broken or chain_status(ch) != "broken"]
    print(f"\n{a} ⇄ {c}: {len(shown)} evidence chain(s)"
          + (f" using evidence up to {ns.until}" if ns.until else "") + "\n")
    for i, ch in enumerate(shown, 1):
        print(render_chain(graph, ch, i), end="\n\n")

    if ns.analogical:
        # Analogies score zero as evidence, so they are listed apart as hints.
        hints = [ch for ch in find_chains(graph, graph.resolve(a), graph.resolve(c),
                                          max_hops=2, include_analogical=True)
                 if any(h.edge.kind == "analogical" for h in ch.hops)]
        print(f"Analogies ({len(hints)}): hints for idea generation, not evidence\n")
        for ch in hints:
            print(render_chain(graph, ch), end="\n\n")


if __name__ == "__main__":
    main()
