# Experiment results

Run 2026-11-05. Raw outputs are in `results/`. Data was not committed (Hetionet 16 MB, ConceptNet 500 MB). Each
script's docstring says where to get it.

## Setup

| Data | Size | Used for |
|---|---|---|
| Hetionet v1.0 | 47,031 nodes, 2.25M edges, 11 node types | E1–E3, E6: a real biomedical graph with known answers |
| ConceptNet 5.7, English | 809k concepts, 2.25M edges, after dropping lexical relations | E4: truly cross-domain pairs |
| Demo graph | 26 nodes | E5: verifier stress test |

PubMed/PubTator, OpenAlex, Wikidata and Europe PMC were not reachable from the build environment, and no LLM API
key was available, so the PubTator adapter and the Claude verifier remain untested.

## E1: Do chains carry real signal? (Hetionet, held-out treatments)

All 1,145 Compound–treats/palliates–Disease edges were removed. Each scorer then had to tell the true pairs from
1,145 **degree-matched** decoys: the same disease, paired with a compound of similar connectivity. Degree matching
removes the easy shortcut that popular drugs tend to treat popular diseases. Chains of up to 3 hops.

| Scorer | AUROC | AUPRC |
|---|---|---|
| Logistic regression on all path features (5-fold CV) | 0.670 | 0.674 |
| Best single chain, no hub penalty | 0.661 | 0.681 |
| Best single chain, evidence kinds ignored | 0.660 | 0.651 |
| Mean of top-5 chains | 0.660 | 0.658 |
| Evidence-weighted DWPC (sum over all chains) | 0.658 | 0.629 |
| Best single chain (linkfinder, w = 0.4) | 0.650 | 0.642 |
| DWPC (Rephetio-style, sum over all chains) | 0.638 | 0.596 |
| Spectral embedding (black box, rank 100) | 0.621 | 0.586 |
| Shortest path length | 0.611 | 0.565 |
| Path count | 0.558 | 0.535 |

What this shows:
- **One readable chain is as predictive as aggregate scores and better than the black box.** The best single chain
  scores 0.65–0.67. Summing over all chains gives 0.64–0.66, and the spectral baseline 0.62. On this task,
  interpretability costs nothing.
- **Counting paths is nearly useless (0.56).** Penalising hubs (DWPC, 0.64) and weighting by evidence type
  (0.66) both help.
- **Absolute numbers are modest.** Degree-matched decoys are hard, and these scorers are unsupervised. They are not
  comparable to Rephetio's published 0.97, which used supervised per-metapath features and different negatives.
- **"resembles" (similarity) edges never appeared in a top chain.** The no-analogy ablation is identical.

## E2: Small world (Hetionet)

- 73% of random node pairs connect within 3 hops; 17% within 2.
- True drug–disease pairs have a median of **252** distinct ≤3-hop chains, and 90th-percentile random pairs have
  over 2,000. Path finding is trivial; **ranking is the problem.**
- The biggest hubs are anatomy terms (brain, testis, adrenal gland, heart, lung: about 14–16k edges each, mostly
  gene expression).
- The top chain passes through a top-1% hub in 47% of cases without the hub penalty, and 11% with w = 0.4.

## E6: How strong should the hub penalty be? (Hetionet)

"Textbook" means the top chain for a true pair is *drug →binds→ target gene →associated with→ disease*.

| w | AUROC (best chain) | Textbook mechanism at #1 | Median degree of bridge nodes |
|---|---|---|---|
| 0.00 | 0.661 | 44% | 516 |
| 0.05 | 0.671 | **44%** | 385 |
| **0.10** | **0.674** | 42% | 276 |
| 0.15 | 0.671 | 37% | 239 |
| 0.20 | 0.667 | 32% | 188 |
| 0.30 | 0.656 | 18% | 162 |
| 0.40 | 0.650 | 15% | 149 |

- The prototype's original setting, w = 0.4 (Rephetio's value, chosen there for *aggregated* features), is about
  4× too strong for picking a *single* explanation. It produces an **obscurity bias**: chains through rare,
  uninformative nodes win.
- Example: tamoxifen ⇄ breast cancer at w = 0.4 ranks *FMO3 → SPRR1A* first. Without the penalty it ranks
  *binds ESR1 → associated with breast cancer* first, the actual mechanism. Methotrexate ⇄ rheumatoid arthritis
  shows the same pattern (TYMS and MTHFR win without the penalty, obscure transporters win with it).
- **Action taken:** linkfinder's default is now w = 0.1. Explanation quality should also be scored by a separate
  criterion, such as the textbook rate here, not only by AUROC.

## E4: Truly cross-domain pairs (ConceptNet)

Pairs tested: cancer ⇄ car, cancer ⇄ chickenpox, music ⇄ mathematics, spider ⇄ bridge, bird ⇄ airplane,
coffee ⇄ sleep (`results/conceptnet_chains.txt`). Run with w = 0.4.

- **Coverage is the main limit.** None of the real mechanistic links between cancer and cars (asbestos brakes, diesel
  exhaust, benzene, platinum catalysts) exist in ConceptNet. The best it finds is commonsense such as
  *cancer ←causes— smoking —needs→ lighter —found in→ car*. Cancer ⇄ chickenpox only meets at the generic node
  "disease". A general-knowledge graph is too shallow for discovery; the bridges need domain sources (papers,
  patents, regulatory reports).
- **Word sense breaks cross-domain chains.** "spider" (snooker rest, card game, slang for heroin) and "bridge" (card
  game, song section, structure) produce chains that are true word by word and meaningless as a whole, such as
  *spider → billiards → bridge*. Cross-domain linking needs entity disambiguation before path finding.
- **The obscurity bias shows here too.** For music ⇄ mathematics the penalised ranking picks *riemannian*, *supersaw*,
  *gerbe*. Without the penalty it picks *triangle* and *geometry*, generic but sensible. E6's lower w addresses this.
- **Untyped "RelatedTo" edges are most of the noise.** Dropping them cut chains by 4–6× (cancer ⇄ car: 206 → 51)
  and removed most nonsense.

## E5: What can the offline verifier catch? (demo graph, 31 edges)

| Evidence shown to verifier | Accepted as support |
|---|---|
| The edge's real sentence | 31 / 31 (100%) |
| Another edge's sentence (misattributed citation) | 1 / 31 (3%) |
| The real sentence, negated ("does not cause …") | **31 / 31 (100%)** |

The name-matching verifier catches misattributed citations but not contradictions. Entailment checking (the
`ClaudeVerifier`, or an NLI model) is required, not optional. In ConceptNet the check is also circular, because the
evidence text is generated from the edge itself. Per-hop verification only means something when the evidence is
independent text.

## What to do next, in priority order

1. **Disambiguate entities before linking.** Wikidata or UMLS IDs instead of strings.
2. **Get typed, cited domain edges.** SemMedDB or PubTator relations for biomedicine, and LLM extraction from
   patents and industry reports for other domains. ConceptNet is not enough.
3. **Run the LLM or NLI verifier on the negation test** and target close to 0% acceptance of negated evidence.
4. **Learn the scorer** (hub weight, kind weights, length decay) on time-sliced data, and keep a separate
   explanation-quality metric.
5. **Swanson replication** once PubMed access is available.
