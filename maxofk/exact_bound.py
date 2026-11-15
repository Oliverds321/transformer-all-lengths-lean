"""The fixed-length accuracy bound in exact integer arithmetic.

The statement is about the QUANTISED model: every weight is an integer divided by
S = 2**bits (W_Q is first divided by sqrt(d_model), so that attention scores are
rational), with real-valued softmax. Nothing here uses floating point after the
weights have been rounded. This module computes the certificate (the largest
certified bound for each query token and maximum) and the number of inputs it
covers; the Lean development checks the same computation in the kernel.

Tables (integers; D = S**4 is the common denominator of scores and of logits):

    score(q; t, i) = (EQKE[q][t] + EQKP[q][i]) / D
    logit_j        = (direct[q][j] + sum_i a_i (EVOU[t_i][j] + PVOU[i][j])) / D

Exponentials. For a query token q put c = max_t EQKE[q][t] and c' = max_i EQKP[q][i].
The program computes integers lo <= 2**F exp((EQKE[q][t] - c) / D) <= hi, and the same
for positions, by an alternating Taylor sum for exp(-z), 0 <= z <= 1, in exact
rationals, followed by repeated squaring with outward rounding.

A case (q, m, r) consists of the sequences with last token q, maximum m and every
other token at most r. Certificate condition: for every wrong answer j and every
admissible position i* of the maximum,

    sum_i  L(t_i, i) > 0,     L(t, i) = v * (lo_t lo_i if v >= 0 else hi_t hi_i),
                              v = (EVOU[t][m] - EVOU[t][j]) + (PVOU[i][m] - PVOU[i][j])
                                  + direct[q][m] - direct[q][j]

where the other positions take the token of {0..r} + {m} that minimises L. Since
L(t, i) <= 2**(2F) exp(score - c - c') v, positivity of the sum implies that the
logit of m exceeds the logit of j.

usage: python exact_bound.py checkpoints/maxofk_v64_k4_s0.pt [--bits 10] [--frac 256]
"""
import argparse
import math
import time
from fractions import Fraction

import numpy as np
import torch

from model import MaxOfK


def quantise(model, bits):
    s = 1 << bits
    d = model.d_model

    def r(t):
        return np.array([[int(round(x * s)) for x in row] for row in t.double().tolist()], dtype=object)

    return {
        "E": r(model.E), "P": r(model.P), "WQ": r(model.WQ / math.sqrt(d)), "WK": r(model.WK),
        "WV": r(model.WV), "WO": r(model.WO), "WU": r(model.WU), "S": s,
    }


def tables(w, ctx):
    E, P, s = w["E"], w["P"], w["S"]
    xq = E + P[ctx - 1][None, :]
    qk = w["WQ"].dot(w["WK"].T)
    ov = w["WV"].dot(w["WO"]).dot(w["WU"])
    return {
        "EQKE": xq.dot(qk).dot(E.T), "EQKP": xq.dot(qk).dot(P.T),
        "EVOU": E.dot(ov), "PVOU": P.dot(ov), "direct": xq.dot(w["WU"]) * (s * s),
    }


def exp_bounds(num, den, frac, terms=60, guard=96):
    """Integers (lo, hi) with lo <= 2**frac * exp(-num/den) <= hi, for num >= 0, den > 0."""
    if num == 0:
        return 1 << frac, 1 << frac
    p = 0
    while num > (den << p):
        p += 1
    z = Fraction(num, den << p)                      # 0 < z <= 1
    g = frac + guard
    term, total, sums = Fraction(1), Fraction(0), []
    for k in range(terms + 1):
        total += term
        sums.append(total)
        term = -term * z / (k + 1)
    # partial sums of an alternating series with decreasing terms: odd index below, even above
    lo_f, hi_f = sums[terms - 1], sums[terms]        # terms is even
    lo = (lo_f.numerator << g) // lo_f.denominator
    hi = -((-hi_f.numerator << g) // hi_f.denominator)
    for _ in range(p):
        lo = (lo * lo) >> g
        hi = -((-(hi * hi)) >> g)
    return lo >> guard, -((-hi) >> guard)


def certified_r(tab, ctx, frac):
    EQKE, EQKP, EVOU, PVOU, direct = (tab[k] for k in ("EQKE", "EQKP", "EVOU", "PVOU", "direct"))
    V, K = EQKE.shape[0], ctx
    den = None
    rmax = [[-2] * V for _ in range(V)]
    wrong = ~np.eye(V, dtype=bool)
    for q in range(V):
        c = max(EQKE[q])
        cp = max(EQKP[q])
        tok = [exp_bounds(c - EQKE[q][t], tab["D"], frac) for t in range(V)]
        pos = [exp_bounds(cp - EQKP[q][i], tab["D"], frac) for i in range(K)]
        lo = np.array([[tok[t][0] * pos[i][0] for i in range(K)] for t in range(V)], dtype=object)   # (t, i)
        hi = np.array([[tok[t][1] * pos[i][1] for i in range(K)] for t in range(V)], dtype=object)
        for m in range(q, V):
            dv_tok = EVOU[:m + 1, m:m + 1] - EVOU[:m + 1]                       # (t, j)
            dv_pos = PVOU[:, m:m + 1] - PVOU                                     # (i, j)
            dd = direct[q][m] - direct[q]                                        # (j,)
            v = dv_tok[:, None, :] + dv_pos[None, :, :] + dd[None, None, :]      # (t, i, j)
            f = np.where(v >= 0, lo[:m + 1, :, None] * v, hi[:m + 1, :, None] * v)
            f_max = f[m]
            if m:
                g = np.concatenate([f_max[None], np.minimum(np.minimum.accumulate(f[:m], axis=0), f_max[None])])
            else:
                g = f_max[None]
            free = g[:, :K - 1, :]
            if q == m:
                total = free.sum(axis=1) + f_max[K - 1][None, :]
                ok = (total > 0)[:, wrong[m]].all(axis=1)
            else:
                base = free.sum(axis=1) + f[q, K - 1][None, :]
                total = base[:, None, :] - free + f_max[None, :K - 1, :]
                ok = (total > 0)[:, :, wrong[m]].all(axis=2).all(axis=1)
            start = 0 if q == m else q + 1
            bad = np.nonzero(~ok[start:])[0]
            top = start + (int(bad[0]) if len(bad) else m + 1 - start) - 1
            if top >= start:
                rmax[q][m] = top - 1
    return rmax


def count_certified(rmax, ctx):
    V, K = len(rmax), ctx
    total = 0
    for m in range(V):
        for q in range(m + 1):
            r = rmax[q][m]
            if r >= -1:
                total += (r + 2) ** (K - 1) if q == m else (r + 2) ** (K - 1) - (r + 1) ** (K - 1)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--bits", type=int, default=10)
    ap.add_argument("--frac", type=int, default=256)
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    a = ck["args"]
    model = MaxOfK(a["vocab"], a["ctx"], a["d_model"])
    model.load_state_dict(ck["state"])
    V, K = a["vocab"], a["ctx"]

    start = time.time()
    w = quantise(model, args.bits)
    tab = tables(w, K)
    tab["D"] = w["S"] ** 4
    rmax = certified_r(tab, K, args.frac)
    cert = count_certified(rmax, K)
    elapsed = time.time() - start
    total = V ** K
    print(f"{args.checkpoint}: V={V} K={K}, weights = integers / 2^{args.bits}, exp to {args.frac} fractional bits")
    print(f"exactly certified: {cert} of {total}")
    print(f"  = {Fraction(cert, total).limit_denominator(10 ** 12)} ~ {cert / total:.8f}   ({elapsed:.0f}s)")
    bits_e = max(int(x).bit_length() for x in np.abs(tab["EQKE"]).ravel())
    print(f"largest score table entry: {bits_e} bits; denominator D = 2^{4 * args.bits}")

    # sanity check of the quantisation (floating point, sampled; not part of the proof)
    qm = MaxOfK(V, K, a["d_model"]).double()
    with torch.no_grad():
        for name in ("E", "P", "WK", "WV", "WO", "WU"):
            getattr(qm, name).copy_(torch.tensor(np.array(w[name], dtype=np.float64) / w["S"]))
        qm.WQ.copy_(torch.tensor(np.array(w["WQ"], dtype=np.float64) / w["S"]) * math.sqrt(a["d_model"]))
        g = torch.Generator().manual_seed(5)
        t = torch.randint(0, V, (1 << 19, K), generator=g)
        acc_q = (qm(t).argmax(-1) == t.max(-1).values).double().mean().item()
        acc_o = (model.double()(t).argmax(-1) == t.max(-1).values).double().mean().item()
    print(f"sampled accuracy (2^19 inputs, float64): quantised model {acc_q:.6f}, original {acc_o:.6f}")


if __name__ == "__main__":
    main()
