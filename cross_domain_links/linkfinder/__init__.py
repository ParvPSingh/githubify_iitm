"""linkfinder: evidence-backed link discovery between distant concepts."""
from .graph import Edge, Evidence, EvidenceGraph, Node
from .paths import Chain, find_chains
from .score import rank
from .verify import ClaudeVerifier, HeuristicVerifier, chain_status, verify_chain

__all__ = ["Edge", "Evidence", "EvidenceGraph", "Node", "Chain", "find_chains", "rank",
           "ClaudeVerifier", "HeuristicVerifier", "chain_status", "verify_chain", "discover"]


def discover(graph: EvidenceGraph, a: str, c: str, max_hops: int = 5, top_k: int = 10,
             verifier=None, include_analogical: bool = False) -> list[Chain]:
    """Full pipeline: candidate chains -> rank -> verify each hop -> keep the ones that hold."""
    src, dst = graph.resolve(a), graph.resolve(c)
    chains = find_chains(graph, src, dst, max_hops=max_hops,
                         include_analogical=include_analogical)
    ranked = rank(graph, chains, top_k=top_k)
    verifier = verifier or HeuristicVerifier()
    return [verify_chain(graph, ch, verifier) for ch in ranked]
