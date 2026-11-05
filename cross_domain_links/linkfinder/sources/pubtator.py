"""Build an evidence graph from PubMed via the PubTator3 API (NCBI).

PubTator3 tags genes, diseases, chemicals, species, variants and cell lines in
PubMed abstracts and normalises them to MeSH / NCBI identifiers. We turn
sentence-level co-occurrence of tagged entities into `cooccurrence` edges whose
evidence is the sentence itself plus the PMID. That gives per-hop provenance
for free; stronger edge kinds can come later from a relation extractor or from
SemMedDB.

For closed discovery (A ... C) pull literature for A and for C separately and
merge: bridge concepts B appear where the two literatures share entities —
Swanson's ABC model.

NOTE: written against the public PubTator3 API but not run in the environment
it was written in (no network access to NCBI there). Expect to adjust field
names if the API has changed. Respect NCBI's rate limit (~3 requests/second).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

from ..graph import Edge, Evidence, EvidenceGraph, Node

BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_DOMAIN = {"Disease": "medicine", "Chemical": "chemistry", "Gene": "molecular biology",
           "Species": "biology", "Mutation": "genetics", "Variant": "genetics",
           "CellLine": "biology"}


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "linkfinder-prototype"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode()
    time.sleep(0.35)
    return json.loads(body)


def search_pmids(query: str, max_pmids: int = 200) -> list[str]:
    pmids: list[str] = []
    page = 1
    while len(pmids) < max_pmids:
        data = _get(f"{BASE}/search/?text={urllib.parse.quote(query)}&page={page}")
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            break
        pmids += [str(r["pmid"]) for r in results if "pmid" in r]
        page += 1
    return pmids[:max_pmids]


def fetch_documents(pmids: list[str], batch: int = 100) -> list[dict]:
    docs: list[dict] = []
    for i in range(0, len(pmids), batch):
        ids = ",".join(pmids[i:i + batch])
        data = _get(f"{BASE}/publications/export/biocjson?pmids={ids}")
        if isinstance(data, dict):
            data = data.get("PubTator3", data.get("documents", []))
        docs += data
    return docs


def _year(doc: dict) -> int | None:
    for key in ("date", "year"):
        m = re.match(r"(\d{4})", str(doc.get(key, "")))
        if m:
            return int(m.group(1))
    for p in doc.get("passages", []):
        m = re.match(r"(\d{4})", str(p.get("infons", {}).get("year", "")))
        if m:
            return int(m.group(1))
    return None


def documents_to_graph(docs: list[dict], min_support: int = 1) -> EvidenceGraph:
    g = EvidenceGraph()
    pending: dict[tuple[str, str], list[Evidence]] = {}
    for doc in docs:
        pmid = str(doc.get("id") or doc.get("pmid") or doc.get("_id", "")).split("|")[0]
        year = _year(doc)
        for p in doc.get("passages", []):
            text, base = p.get("text", ""), p.get("offset", 0)
            # sentence spans relative to the document
            spans, start = [], 0
            for piece in _SENT.split(text):
                s = text.find(piece, start)
                spans.append((base + s, base + s + len(piece), piece))
                start = s + len(piece)
            for s0, s1, sentence in spans:
                ents: dict[str, Node] = {}
                for a in p.get("annotations", []):
                    inf = a.get("infons", {})
                    ident, etype = inf.get("identifier"), inf.get("type", "Concept")
                    locs = a.get("locations", [])
                    if not ident or ident == "-" or not locs:
                        continue
                    off = locs[0].get("offset", -1)
                    if s0 <= off < s1:
                        nid = f"{etype}:{ident}"
                        ents.setdefault(nid, Node(nid, a.get("text", ident), etype,
                                                  _DOMAIN.get(etype, "biomedicine"),
                                                  [a.get("text", "")]))
                ids = sorted(ents)
                for n in ents.values():
                    g.add_node(n)
                for i, x in enumerate(ids):
                    for y in ids[i + 1:]:
                        pending.setdefault((x, y), []).append(
                            Evidence(sentence.strip(), f"PMID:{pmid}", year,
                                     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
    for (x, y), ev in pending.items():
        if len({e.source for e in ev}) >= min_support:
            g.add_edge(Edge(x, y, "co-mentioned with", "cooccurrence", ev))
    return g


def build_for_pair(a: str, c: str, per_side: int = 200, min_support: int = 1) -> EvidenceGraph:
    """Closed discovery: literature for A plus literature for C, merged."""
    docs = fetch_documents(search_pmids(a, per_side)) + fetch_documents(search_pmids(c, per_side))
    return documents_to_graph(docs, min_support=min_support)
