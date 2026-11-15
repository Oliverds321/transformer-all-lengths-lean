"""Figure and table data for the manuscript, from exact integer arithmetic.

For each position-free model (seeds 0 to 4) and each gap g, with the query token
equal to the maximum m (m >= 24):

  tolerance(m, g): the largest N such that the region certificate of
      NNProofs.MaxOfK.CheckPF holds for "other tokens at most m - g, at most N of
      them" (so the model is provably correct with any N such tokens);
  failure(m, g):   the least k such that the failure certificate holds, for
      some wrong answer j, for k copies of the token m - g followed by m (so the
      model is provably wrong).

Wherever failure = tolerance + 1, tolerance(m, g) is exactly the largest N for
which every input with query m and at most N other tokens, all at most m - g,
is answered correctly: the failing input belongs to that class with N + 1.

For each gap the script records, over the maxima m, the medians of both counts,
the median and largest ratio failure / tolerance for the same m, and the number
of maxima at which the two are adjacent (failure = tolerance + 1). M_LO = 24
keeps m - g >= 0 for every gap up to 24.

The same integer bounds as in the Lean development are used (weights = integers
/ 2^10, exponential bounds with 256 fractional bits). The pairs with
failure = tolerance + 1 are the exact thresholds that the Lean development checks
in the kernel (the tables nTight written by maxofk/export_pf.py; theorems
threshold and tightCounts); the medians are computed by this script.

usage: python make_figures.py        (run from paper/; writes figure_gap.pdf, figure_data.json)
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "maxofk"))

import exact_bound  # noqa: E402
from export_lean import P_SQ, lean_exp_bounds  # noqa: E402
from model import MaxOfK  # noqa: E402

M_LO = 24
GAPS = list(range(1, 25))
BIG = 10 ** 40


def analyse(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    a = ck["args"]
    model = MaxOfK(a["vocab"], a["ctx"], a["d_model"], pos=False)
    model.load_state_dict(ck["state"])
    V = a["vocab"]
    w = exact_bound.quantise(model, 10)
    S = w["S"]
    D = S ** 4
    E = w["E"]
    EQKE = E.dot(w["WQ"]).dot(E.dot(w["WK"]).T)
    EVOU = E.dot(w["WV"]).dot(w["WO"]).dot(w["WU"])
    DIR = E.dot(w["WU"]) * (S * S)
    tol = {g: [] for g in GAPS}
    fail = {g: [] for g in GAPS}
    for m in range(M_LO, V):
        c = max(EQKE[m])
        assert c - min(EQKE[m]) <= D << P_SQ
        b = [lean_exp_bounds(c - EQKE[m][t], D) for t in range(m + 1)]
        lo = np.array([x[0] for x in b], dtype=object)[:, None]
        hi = np.array([x[1] for x in b], dtype=object)[:, None]
        v = (EVOU[:m + 1, m:m + 1] - EVOU[:m + 1]) + (DIR[m][m] - DIR[m])[None, :]
        wlo = np.where(v >= 0, lo * v, hi * v)
        whi = np.where(v >= 0, hi * v, lo * v)
        wrong = np.arange(V) != m
        base = wlo[m][wrong]
        zero = np.zeros((1, V), dtype=object)
        gmin = np.minimum.accumulate(np.concatenate([zero, wlo[:m]], axis=0), axis=0)[:, wrong]   # index b
        for g in GAPS:
            bb = m - g + 1                      # other tokens below bb, i.e. at most m - g
            if bb < 0:
                continue
            if not (base > 0).all():
                tol[g].append(-1)
            else:
                gg = gmin[bb]
                neg = gg < 0
                tol[g].append(min((int(x) - 1) // int(-y) for x, y in zip(base[neg], gg[neg])) if neg.any() else BIG)
            if bb >= 1:
                t = m - g                        # k copies of token t, then m; best wrong answer j
                best = BIG
                for j in range(V):
                    if j == m:
                        continue
                    A, B = int(whi[t][j]), int(whi[m][j])
                    if A < 0:
                        best = min(best, max(0, -((-B) // (-A))))
                    elif A == 0 and B <= 0:
                        best = 0
                fail[g].append(best)
    return tol, fail


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for seed in range(5):
        path = os.path.join(HERE, "..", "maxofk", "checkpoints", f"maxofk_v64_k10_s{seed}_nopos.pt")
        if not os.path.exists(path):
            continue
        tol, fail = analyse(path)
        d = {k: [] for k in ("tolerance_median", "failure_median", "ratio_median", "ratio_max",
                             "adjacent", "maxima", "tolerance_min", "failure_min")}
        for g in GAPS:
            # per maximum m: both certificates exist and are finite
            pairs = [(t, f_) for t, f_ in zip(tol[g], fail[g]) if 0 < t < BIG and f_ < BIG]
            ratios = sorted(f_ / t for t, f_ in pairs)
            d["tolerance_median"].append(sorted(t for t, _ in pairs)[len(pairs) // 2])
            d["failure_median"].append(sorted(f_ for _, f_ in pairs)[len(pairs) // 2])
            d["ratio_median"].append(ratios[len(ratios) // 2])
            d["ratio_max"].append(ratios[-1])
            d["adjacent"].append(sum(1 for t, f_ in pairs if f_ == t + 1))
            d["maxima"].append(len(pairs))
            d["tolerance_min"].append(min(tol[g]))
            d["failure_min"].append(min(fail[g]))
        d["gap"] = GAPS
        data[seed] = d
        print(f"seed {seed}: adjacent / maxima by gap:", list(zip(d["adjacent"][:8], d["maxima"][:8])))
        print(f"          largest ratio failure/tolerance by gap:", [f"{x:.2f}" for x in d["ratio_max"][:10]])
    with open(os.path.join(HERE, "figure_data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({str(k): {kk: [str(x) for x in vv] for kk, vv in v.items()} for k, v in data.items()}, f, indent=1)

    plt.rcParams.update({"font.size": 8.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    gmax = 16
    for seed, d in data.items():
        kw = dict(lw=1.6 if seed == 0 else 0.7, alpha=1.0 if seed == 0 else 0.55)
        gs = d["gap"][:gmax]
        ax.semilogy(gs, d["tolerance_median"][:gmax], color="#1f5fa8",
                    label="proved correct with up to $N$ other tokens" if seed == 0 else None, **kw)
        ax.semilogy(gs, d["failure_median"][:gmax], color="#b5462f", ls="--",
                    label="proved wrong from $k$ copies" if seed == 0 else None, **kw)
        frac = [a_ / mx for a_, mx in zip(d["adjacent"][:gmax], d["maxima"][:gmax])]
        bx.plot(gs, frac, color="#2e8b57", marker="o" if seed == 0 else None, ms=2.5,
                label="seed 0" if seed == 0 else ("seeds 1 to 4" if seed == 1 else None), **kw)
    ax.set_xlabel("gap $g$")
    ax.set_ylabel("median over maxima $m\\geq 24$")
    ax.set_title("(a) the two certificates", fontsize=8.5)
    bx.set_xlabel("gap $g$")
    bx.set_ylabel("fraction of maxima with $k=N+1$")
    bx.set_title("(b) maxima at which they are adjacent", fontsize=8.5)
    for a in (ax, bx):
        a.grid(True, which="major", lw=0.3, alpha=0.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    bx.legend(fontsize=7, frameon=False, loc="lower left")
    bx.set_ylim(0, 1.03)
    for a in (ax, bx):
        a.set_xticks([1, 4, 8, 12, 16])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figure_gap.pdf"))
    print("wrote figure_gap.pdf")


if __name__ == "__main__":
    main()
