"""Experiment 1-3 on Hetionet v1.0 (47,031 nodes, 2.25M edges, 11 node types).

E1  Does chain scoring carry real signal?  Hide every Compound–treats–Disease and
    Compound–palliates–Disease edge, then ask each scorer to tell true
    (drug, disease) pairs from degree-matched decoys. Report AUROC / AUPRC.
    Scorers: path count, DWPC (hub-penalised), evidence-weighted DWPC, the
    best single chain (linkfinder's score), ablations, and a black-box
    spectral baseline.
E2  Small-world check: how many chains connect random node pairs?
E3  Case studies: top chains for well-known held-out drug–disease pairs.

Usage:  python3 experiments/hetionet_eval.py path/to/hetionet-v1.0.json.bz2 [out_dir]
Get the data: https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/json/hetionet-v1.0.json.bz2
"""
from __future__ import annotations

import bz2
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linkfinder.graph import EDGE_KINDS  # noqa: E402

random.seed(0)
np.random.seed(0)

# Hetionet relation -> linkfinder edge kind
KIND = {
    "treats": "causal", "palliates": "causal", "causes": "causal",
    "binds": "mechanistic", "interacts": "mechanistic", "regulates": "mechanistic",
    "upregulates": "mechanistic", "downregulates": "mechanistic",
    "participates": "mechanistic", "includes": "mechanistic",
    "associates": "associative", "localizes": "associative", "presents": "associative",
    "expresses": "associative", "covaries": "associative",
    "resembles": "analogical",   # similarity: a hint, not evidence
}
HELD_OUT = {"treats", "palliates"}


def load(path):
    t = time.time()
    d = json.load(bz2.open(path))
    ids = {}
    names, types = [], []
    for n in d["nodes"]:
        ids[(n["kind"], n["identifier"])] = len(names)
        names.append(n["name"])
        types.append(n["kind"])
    adj = defaultdict(dict)  # u -> {v: (relation, n_sources)}
    held = set()
    for e in d["edges"]:
        u = ids[tuple(e["source_id"])]
        v = ids[tuple(e["target_id"])]
        rel = e["kind"]
        if rel in HELD_OUT:
            held.add((u, v))
            continue
        data = e["data"]
        n_src = max(len(data.get("sources", [])), 1)
        n_src += min(len(data.get("pubmed_ids", [])), 3)
        # keep the strongest description if two relations join the same pair
        prev = adj[u].get(v)
        if prev is None or EDGE_KINDS[KIND[rel]] > EDGE_KINDS[KIND[prev[0]]]:
            adj[u][v] = (rel, n_src)
            adj[v][u] = (rel, n_src)
    print(f"loaded {len(names)} nodes, {sum(len(x) for x in adj.values())//2} pairs, "
          f"{len(held)} held-out treatment edges in {time.time()-t:.0f}s")
    return names, types, adj, held


def hop_ev(rel, n_src, use_kinds=True):
    w = EDGE_KINDS[KIND[rel]] if use_kinds else 1.0
    return w * (1 - 0.5 ** n_src)


def chains(adj, s, t, max_hops=3, allow_analogical=True):
    """All simple paths s..t with up to 3 hops, via meet-in-the-middle."""
    ok = (lambda r: True) if allow_analogical else (lambda r: KIND[r] != "analogical")
    Ns, Nt = adj[s], adj[t]
    out = []
    if t in Ns and ok(Ns[t][0]):
        out.append((s, t))
    for u, (r1, _) in Ns.items():
        if u == t or not ok(r1):
            continue
        if u in Nt and ok(Nt[u][0]):
            out.append((s, u, t))
        if max_hops < 3:
            continue
        Au = adj[u]
        small, big = (Au, Nt) if len(Au) < len(Nt) else (Nt, Au)
        for v in small:
            if v in big and v not in (s, t, u) and ok(Au[v][0]) and ok(Nt[v][0]):
                out.append((s, u, v, t))
    return out


def chain_parts(adj, deg, path, w=0.4, use_kinds=True):
    spec, ev = 1.0, 1.0
    for a, b in zip(path, path[1:]):
        rel, n = adj[a][b]
        spec *= (deg[a] * deg[b]) ** (-w)
        ev *= hop_ev(rel, n, use_kinds)
    return spec, ev, len(path) - 1


def linkfinder_score(spec, ev, k, decay=0.9):
    return (spec ** (1 / k)) * (ev ** (1 / k)) * decay ** (k - 1)


def features(adj, deg, s, t):
    ps = chains(adj, s, t)
    f = {"path_count": len(ps), "dwpc": 0.0, "ev_dwpc": 0.0,
         "best_chain": 0.0, "best_chain_no_hub": 0.0, "best_chain_no_kinds": 0.0,
         "best_chain_no_analogy": 0.0, "top5_mean": 0.0, "neg_shortest": -9.0}
    if not ps:
        return f, []
    scored = []
    for p in ps:
        spec, ev, k = chain_parts(adj, deg, p)
        sc = linkfinder_score(spec, ev, k)
        scored.append((sc, p))
        f["dwpc"] += spec
        f["ev_dwpc"] += spec * ev
        f["best_chain_no_hub"] = max(f["best_chain_no_hub"],
                                     linkfinder_score(1.0, ev, k))
        s2, e2, _ = chain_parts(adj, deg, p, use_kinds=False)
        f["best_chain_no_kinds"] = max(f["best_chain_no_kinds"], linkfinder_score(spec, e2, k))
        if not any(KIND[adj[a][b][0]] == "analogical" for a, b in zip(p, p[1:])):
            f["best_chain_no_analogy"] = max(f["best_chain_no_analogy"], sc)
    scored.sort(reverse=True)
    f["best_chain"] = scored[0][0]
    f["top5_mean"] = float(np.mean([x for x, _ in scored[:5]]))
    f["neg_shortest"] = -min(len(p) - 1 for p in ps)
    return f, scored


def spectral_scores(names, adj, pairs, k=100):
    """Black-box baseline: rank-k reconstruction of 3-step walks on the
    normalised adjacency (no paths, no explanations)."""
    n = len(names)
    rows, cols = [], []
    for u, nb in adj.items():
        for v in nb:
            rows.append(u)
            cols.append(v)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1
    Dm = diags(1 / np.sqrt(d))
    S = Dm @ A @ Dm
    vals, vecs = eigsh(S, k=k, which="LA")
    return np.array([float(np.sum(vals ** 3 * vecs[s] * vecs[t])) for s, t in pairs])


def main(path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, types, adj, held = load(path)
    deg = {u: len(nb) for u, nb in adj.items()}
    for u in range(len(names)):
        deg.setdefault(u, 0)
        adj.setdefault(u, {})

    # ---------- E1: held-out treatment recovery ----------
    compounds = sorted((u for u in range(len(names)) if types[u] == "Compound"), key=lambda u: deg[u])
    rank = {u: i for i, u in enumerate(compounds)}
    positives = sorted(held)
    negatives = []
    for c, d in positives:  # degree-matched decoy: a compound of similar degree
        for _ in range(100):
            j = rank[c] + random.randint(-15, 15)
            c2 = compounds[min(max(j, 0), len(compounds) - 1)]
            if c2 != c and (c2, d) not in held:
                negatives.append((c2, d))
                break
    pairs = positives + negatives
    y = np.array([1] * len(positives) + [0] * len(negatives))
    t0 = time.time()
    feats = [features(adj, deg, s, t)[0] for s, t in pairs]
    print(f"E1: scored {len(pairs)} pairs in {time.time()-t0:.0f}s")
    keys = list(feats[0])
    X = np.array([[f[k] for k in keys] for f in feats])
    spec = spectral_scores(names, adj, pairs)

    results = {}
    for i, k in enumerate(keys):
        results[k] = (roc_auc_score(y, X[:, i]), average_precision_score(y, X[:, i]))
    results["spectral_blackbox"] = (roc_auc_score(y, spec), average_precision_score(y, spec))
    # supervised combination (5-fold CV) of the interpretable features
    Xl = np.log1p(np.abs(X)) * np.sign(X)
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xl, y):
        m = LogisticRegression(max_iter=2000).fit(Xl[tr], y[tr])
        pred[te] = m.predict_proba(Xl[te])[:, 1]
    results["logreg_all_path_features"] = (roc_auc_score(y, pred), average_precision_score(y, pred))

    print("\nE1 held-out treatment recovery (positives vs degree-matched decoys)")
    print(f"{'scorer':32s} {'AUROC':>6s} {'AUPRC':>6s}")
    for k, (a, p) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:32s} {a:6.3f} {p:6.3f}")

    # ---------- E2: small world ----------
    nodes = [u for u in range(len(names)) if deg[u] > 0]
    sw = []
    for _ in range(300):
        s, t = random.sample(nodes, 2)
        ps = chains(adj, s, t)
        sw.append((len(ps), min((len(p) - 1 for p in ps), default=99)))
    counts = np.array([c for c, _ in sw])
    short = np.array([L for _, L in sw])
    e2 = {
        "pairs": len(sw),
        "connected_within_2": float(np.mean(short <= 2)),
        "connected_within_3": float(np.mean(short <= 3)),
        "median_chains_le3": float(np.median(counts)),
        "p90_chains_le3": float(np.percentile(counts, 90)),
        "positive_pairs_median_chains": float(np.median(X[: len(positives), keys.index("path_count")])),
    }
    hubs = sorted(deg, key=deg.get, reverse=True)[:5]
    e2["top_hubs"] = [(names[h], types[h], deg[h]) for h in hubs]
    print("\nE2 small world:", json.dumps(e2, indent=1))

    # hub use in top chains: by linkfinder score vs unpenalised
    hub_cut = np.percentile([deg[u] for u in nodes], 99)
    hub_frac = {"linkfinder": [], "no_hub_penalty": []}
    for s, t in positives[:300]:
        ps = chains(adj, s, t)
        if not ps:
            continue
        parts = [(p, *chain_parts(adj, deg, p)) for p in ps]
        best = max(parts, key=lambda x: linkfinder_score(x[1], x[2], x[3]))[0]
        best_nh = max(parts, key=lambda x: linkfinder_score(1.0, x[2], x[3]))[0]
        hub_frac["linkfinder"].append(any(deg[u] > hub_cut for u in best[1:-1]))
        hub_frac["no_hub_penalty"].append(any(deg[u] > hub_cut for u in best_nh[1:-1]))
    e2["top_chain_uses_top1pct_hub"] = {k: float(np.mean(v)) for k, v in hub_frac.items()}
    print("top chain passes through a top-1% hub:", e2["top_chain_uses_top1pct_hub"])

    # ---------- E3: case studies ----------
    lookup = {(names[s].lower(), names[t].lower()): (s, t) for s, t in positives}
    wanted = [("metformin", "type 2 diabetes mellitus"), ("tamoxifen", "breast cancer"),
              ("methotrexate", "rheumatoid arthritis"), ("levodopa", "parkinson's disease"),
              ("cisplatin", "testicular cancer")]
    cases = {}
    for key in wanted:
        if key not in lookup:
            continue
        s, t = lookup[key]
        _, scored = features(adj, deg, s, t)
        def fmt(p):
            hops = []
            for a, b in zip(p, p[1:]):
                hops.append(f"{names[a]} [{types[a]}] -{adj[a][b][0]}-")
            return " ".join(hops) + f" {names[p[-1]]}"
        plain = sorted(scored, key=lambda x: -linkfinder_score(1.0, chain_parts(adj, deg, x[1])[1], len(x[1]) - 1))
        cases[" ⇄ ".join(key)] = {
            "n_chains": len(scored),
            "linkfinder_top3": [fmt(p) for _, p in scored[:3]],
            "no_hub_penalty_top3": [fmt(p) for _, p in plain[:3]],
        }
    print("\nE3 case studies:", json.dumps(cases, indent=1, ensure_ascii=False))

    (out_dir / "hetionet_results.json").write_text(json.dumps(
        {"E1": {k: {"auroc": a, "auprc": p} for k, (a, p) in results.items()},
         "E1_n": {"positives": len(positives), "negatives": len(negatives)},
         "E2": e2, "E3": cases}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "experiments/results")
