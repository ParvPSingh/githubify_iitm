# linkfinder: evidence-backed links between distant concepts

Give it two concepts from different worlds (say *cancer* and *automobile*). It returns ranked chains
`A → B₁ → … → C` where **every hop carries its own citation**, so each result can be checked hop by hop
without trusting the model.

```
#1 cancer → mesothelioma → asbestos → brake lining → automobile
   score 0.183 | specificity 0.408 | evidence 0.511 | domains: automotive, materials, oncology | status: verified
   ✔ cancer ←[is a · mechanistic]— mesothelioma
   ✔ mesothelioma ←[causes · causal]— asbestos
       “Asbestos (all forms) causes mesothelioma and cancer of the lung, larynx and ovary.”  [IARC Monographs Vol. 100C (2012), 2012]
   ✔ asbestos ←[historically contained · mechanistic]— brake lining
   ✔ brake lining ←[has part · mechanistic]— automobile
```

This is a research prototype of Swanson-style closed literature-based discovery (the ABC model),
with three additions: typed hops, hub-aware ranking, and an LLM that **checks** evidence instead of supplying it.

## Pipeline

1. **Evidence graph** (`graph.py`): nodes carry a domain; an edge cannot exist without at least one
   `Evidence(text, source, year)`. Edges have a *kind*: `causal`, `mechanistic`, `associative`,
   `cooccurrence`, or `analogical`. Analogies are kept as hints and never count as evidence.
2. **Candidate chains** (`paths.py`): all simple paths up to *k* hops, pruned by a backwards BFS from the target.
3. **Ranking** (`score.py`): per-hop geometric mean of
   - *specificity*, a degree-weighted path count (Hetionet / Project Rephetio) that sinks chains routed through hubs;
   - *evidence strength*, from the edge kind and the number of independent sources;
   - a small bonus for chains that cross more domains, and a length decay.
   Chains through the same bridge concepts are collapsed into one.
4. **Hop verification** (`verify.py`): each hop is checked against its own cited text. The offline
   `HeuristicVerifier` only checks that both concepts are named. `ClaudeVerifier` asks an LLM whether the sentence
   actually states the relation (yes / partial / no, structured JSON). Chains with an unsupported hop are dropped.
5. **Time slicing** (`EvidenceGraph.until(year)`): rebuild the graph from evidence published up to a year. Discover
   on the snapshot, then check whether the chain was later confirmed. This is the evaluation protocol.

## Run it

Needs Python 3.10+, with no dependencies for the demo.

```bash
cd cross_domain_links
python3 -m linkfinder demo cancer chickenpox
python3 -m linkfinder demo cancer automobile --max-hops 5
python3 -m linkfinder demo cancer automobile --until 2010   # what was discoverable in 2010?
python3 -m linkfinder demo cancer automobile --analogical   # also list analogy hints
python3 -m unittest tests/test_linkfinder.py
```

Optional, real literature and LLM verification:

```bash
pip install anthropic            # then set ANTHROPIC_API_KEY
python3 -m linkfinder demo cancer chickenpox --verifier claude
python3 -m linkfinder pubtator "fish oil" "Raynaud disease" --save fish.json   # needs access to ncbi.nlm.nih.gov
python3 -m linkfinder graph fish.json "fish oil" "Raynaud"
```

## Status and honest limits

- `data/demo_graph.json` (built by `data/build_demo.py`) has 26 hand-curated nodes. The facts are well established,
  but the statements are paraphrased for the demo. Check the primary source before citing any of them.
- The PubTator3 adapter (`sources/pubtator.py`) is written against the public API but **has not been run against the
  live service** (the build environment had no access to NCBI). Its parser is unit-tested on a synthetic document.
  It only produces `cooccurrence` edges. Typed edges need a relation extractor or SemMedDB.
- `ClaudeVerifier` has not been exercised here (no API key in the build environment).
- The ranking weights (`w=0.4`, kind weights, decay) are hand-set. Learning them from time-sliced data is the
  obvious next step.
- In the `--until 2010` demo, co-occurrence chains through the `patients` hub fill the lower ranks once the
  strong chains are removed. A minimum-evidence threshold per hop is needed.

## Next steps

1. Load SemMedDB or PubTator relations for typed, sentence-cited biomedical edges. Replicate Swanson's
   fish oil → Raynaud and magnesium → migraine from pre-1986 / pre-1988 literature.
2. Add a learned scorer (NBFNet or a path-based RL walker) and compare it with the hand-set score on a
   time-sliced benchmark.
3. Extend beyond biomedicine: Wikidata IDs as the shared key, plus LLM extraction from patents and industry
   reports with sentence-level provenance.
4. Expert rating study comparing these chains against LLM-only, shortest-path and random-path (SciAgents-style)
   baselines on validity, novelty and usefulness.
