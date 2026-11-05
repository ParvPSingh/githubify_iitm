"""Experiment 6: how strong should the hub penalty be?

For hub-penalty exponent w in a grid, on Hetionet's held-out treatment pairs:
  * AUROC of the best single chain (positives vs degree-matched decoys)
  * textbook rate: share of true pairs whose top chain is the direct
    drug -[binds]- target gene -[associates/up/down]- disease mechanism
  * obscurity: median degree of the intermediate nodes in the top chain
Usage: python3 experiments/hub_penalty_sweep.py hetionet-v1.0.json.bz2 [out_dir]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hetionet_eval as H  # noqa: E402


def main(path, out_dir):
    names, types, adj, held = H.load(path)
    deg = {u: len(adj.get(u, {})) for u in range(len(names))}
    for u in range(len(names)):
        adj.setdefault(u, {})
    random.seed(0)
    compounds = sorted((u for u in range(len(names)) if types[u] == "Compound"), key=lambda u: deg[u])
    rank = {u: i for i, u in enumerate(compounds)}
    pos = sorted(held)
    neg = []
    for c, d in pos:
        for _ in range(100):
            c2 = compounds[min(max(rank[c] + random.randint(-15, 15), 0), len(compounds) - 1)]
            if c2 != c and (c2, d) not in held:
                neg.append((c2, d))
                break
    pairs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    cache = {}
    for s, t in pairs:
        ps = H.chains(adj, s, t)
        parts = []
        for p in ps:
            ev, hubs = 1.0, 1.0
            for a, b in zip(p, p[1:]):
                rel, n = adj[a][b]
                ev *= H.hop_ev(rel, n)
                hubs *= deg[a] * deg[b]
            parts.append((p, ev, np.log(hubs), len(p) - 1))
        cache[(s, t)] = parts

    rows = []
    for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        scores, textbook, obsc = [], [], []
        for i, (s, t) in enumerate(pairs):
            parts = cache[(s, t)]
            if not parts:
                scores.append(0.0)
                continue
            best = max(parts, key=lambda x: (x[1] ** (1 / x[3])) * np.exp(-w * x[2] / x[3]) * 0.9 ** (x[3] - 1))
            p, ev, lh, k = best
            scores.append((ev ** (1 / k)) * np.exp(-w * lh / k) * 0.9 ** (k - 1))
            if i < len(pos):
                rels = [adj[a][b][0] for a, b in zip(p, p[1:])]
                textbook.append(k == 2 and rels[0] == "binds" and types[p[1]] == "Gene"
                                and rels[1] in ("associates", "upregulates", "downregulates"))
                obsc.append(np.median([deg[u] for u in p[1:-1]]) if k > 1 else 0)
        rows.append({"w": w, "auroc": roc_auc_score(y, scores),
                     "textbook_rate": float(np.mean(textbook)),
                     "median_intermediate_degree": float(np.median(obsc))})
        print(rows[-1])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "hub_penalty_sweep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "experiments/results")
