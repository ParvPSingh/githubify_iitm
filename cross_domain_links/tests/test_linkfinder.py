import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from linkfinder import discover, find_chains  # noqa: E402
from linkfinder.graph import Edge, Evidence, EvidenceGraph, Node  # noqa: E402
from linkfinder.sources.pubtator import documents_to_graph  # noqa: E402
from linkfinder.verify import chain_status  # noqa: E402

DEMO = EvidenceGraph.load(ROOT / "data" / "demo_graph.json")


def names(chain):
    return [DEMO.nodes[n].name for n in chain.nodes]


class TestGraph(unittest.TestCase):
    def test_edge_requires_evidence(self):
        with self.assertRaises(ValueError):
            Edge("a", "b", "r", "causal", [])

    def test_resolve_alias(self):
        self.assertEqual(DEMO.resolve("Shingles"), "herpes_zoster")
        self.assertEqual(DEMO.resolve("car"), "automobile")

    def test_until_drops_later_evidence(self):
        old = DEMO.until(2000)
        self.assertNotIn(("benzene", "aml"), {(e.src, e.dst) for e in old.edges})


class TestDiscovery(unittest.TestCase):
    def test_analogy_excluded_by_default(self):
        chains = find_chains(DEMO, "cancer", "automobile", max_hops=1)
        self.assertEqual(chains, [])
        chains = find_chains(DEMO, "cancer", "automobile", max_hops=1, include_analogical=True)
        self.assertEqual(len(chains), 1)

    def test_hub_chain_is_not_top(self):
        top = discover(DEMO, "cancer", "automobile", max_hops=5, top_k=10)
        self.assertNotIn("patients", top[0].nodes)
        hub_ranks = [i for i, c in enumerate(top) if "patients" in c.nodes]
        self.assertTrue(all(i > 0 for i in hub_ranks))

    def test_expected_mechanistic_bridges_found(self):
        top = discover(DEMO, "cancer", "automobile", max_hops=5, top_k=10)
        bridges = {frozenset(c.nodes[1:-1]) for c in top}
        self.assertIn(frozenset({"lung_cancer", "diesel_exhaust", "diesel_engine"}), bridges)
        self.assertIn(frozenset({"mesothelioma", "asbestos", "brake_lining"}), bridges)

    def test_chickenpox_chain_verified(self):
        top = discover(DEMO, "cancer", "chickenpox", max_hops=4, top_k=5)
        self.assertEqual(names(top[0]), ["cancer", "herpes zoster", "varicella-zoster virus", "chickenpox"])
        self.assertEqual(chain_status(top[0]), "verified")

    def test_time_slice(self):
        old = DEMO.until(2010)
        top = discover(old, "cancer", "automobile", max_hops=5, top_k=10)
        bridges = {frozenset(c.nodes[1:-1]) for c in top}
        # diesel exhaust was classified Group 1 in 2012, so it should not appear yet
        self.assertNotIn(frozenset({"lung_cancer", "diesel_exhaust", "diesel_engine"}), bridges)


class TestPubTatorParsing(unittest.TestCase):
    def test_sentence_cooccurrence(self):
        text = "Fish oil reduces blood viscosity. Raynaud disease involves high blood viscosity."
        doc = {"id": "123", "date": "1986-01-01", "passages": [{"offset": 0, "text": text, "annotations": [
            {"text": "Fish oil", "infons": {"identifier": "MESH:D005395", "type": "Chemical"}, "locations": [{"offset": 0, "length": 8}]},
            {"text": "blood viscosity", "infons": {"identifier": "MESH:D001809", "type": "Disease"}, "locations": [{"offset": 17, "length": 15}]},
            {"text": "Raynaud disease", "infons": {"identifier": "MESH:D011928", "type": "Disease"}, "locations": [{"offset": 34, "length": 15}]},
            {"text": "blood viscosity", "infons": {"identifier": "MESH:D001809", "type": "Disease"}, "locations": [{"offset": 65, "length": 15}]},
        ]}]}
        g = documents_to_graph([doc])
        pairs = {frozenset((e.src, e.dst)) for e in g.edges}
        self.assertEqual(len(pairs), 2)  # fish oil–viscosity, Raynaud–viscosity; no direct fish oil–Raynaud
        chains = find_chains(g, "Chemical:MESH:D005395", "Disease:MESH:D011928", max_hops=2)
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].hops[0].edge.evidence[0].source, "PMID:123")
        self.assertEqual(chains[0].hops[0].edge.evidence[0].year, 1986)


if __name__ == "__main__":
    unittest.main()
