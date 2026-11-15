"""All-lengths analysis of a Max-of-K model without positional embeddings.

A float64 computation, not a proof. Without positional embeddings the logits
depend on the input only through the query token q (the last token) and the
number of times n_t each token t occurs among the other positions. For a wrong
answer j, the model prefers the maximum m to j exactly when

    F_j(q) + sum_t n_t F_j(t) > 0,     F_j(t) = exp(score(q, t)) * v_j(t),
    v_j(t) = EVOU[t][m] - EVOU[t][j] + direct[q][m] - direct[q][j].

So correctness on any input of any length is a system of linear inequalities in
the counts. Consequences computed here:

* Nmax(q, m, b): if all tokens other than the maximum are below b, the model is
  correct whenever there are at most Nmax such tokens, at every length;
* the smallest number of copies of m - 1 that outvote one copy of m (a failure);
* a lower bound on the accuracy on uniformly random inputs of a given length,
  compared with sampling.

usage: python lengths.py checkpoints/maxofk_v64_k10_s0_nopos.pt
"""
import math
import sys
from math import comb

import torch

from model import MaxOfK

INF = 10 ** 30


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    a = ck["args"]
    model = MaxOfK(a["vocab"], a["ctx"], a["d_model"], pos=not a.get("no_pos", False))
    model.load_state_dict(ck["state"])
    return model.double(), a


def nmax_table(tab):
    """nmax[q][m][b] for b = 0..m (others below b); -1 if even N = 0 is not certified."""
    EQKE, EVOU, direct = tab["EQKE"], tab["EVOU"], tab["direct"]
    V = EQKE.shape[0]
    out = [[None] * V for _ in range(V)]
    for m in range(V):
        wrong = torch.arange(V) != m
        for q in range(m + 1):
            v = (EVOU[:m + 1, m:m + 1] - EVOU[:m + 1]) + (direct[q, m] - direct[q])[None, :]   # (t, j)
            F = torch.exp(EQKE[q, :m + 1] - EQKE[q, m])[:, None] * v                           # (t, j)
            base = (F[m] if q == m else F[m] + F[q])[wrong]
            row = []
            g = torch.full((int(wrong.sum()),), float("inf"), dtype=F.dtype)
            for b in range(m + 1):
                if b:
                    g = torch.minimum(g, F[b - 1][wrong])
                if not bool((base > 0).all()) or not bool((F[m][wrong] >= 0).all()):
                    row.append(-1)
                    continue
                neg = g < 0
                if not bool(neg.any()):
                    row.append(INF)
                else:
                    n = torch.ceil(base[neg] / (-g[neg])) - 1          # largest N with base + N g > 0
                    row.append(int(n.min().item()))
            out[q][m] = row
    return out


def certified_count(nmax, V, length):
    """Sequences of the given length (uniform over V**length) inside the certified region."""
    free = length - 1
    total = 0
    for m in range(V):
        for q in range(m + 1):
            row = nmax[q][m]
            for c in range(0 if q == m else 1, free + 1):
                n = free - c
                ways = comb(free, c)
                if n == 0:
                    if row[0] >= 0:
                        total += ways
                    continue
                for r in range(m):                      # largest other token is exactly r
                    if row[r + 1] >= n:
                        total += ways * ((r + 1) ** n - r ** n)
    return total


def main():
    path = sys.argv[1]
    model, a = load(path)
    V = a["vocab"]
    tab = model.tables()
    nmax = nmax_table(tab)

    print(f"{path}: V={V}, trained at length {a['ctx']}, no positional embedding: {a.get('no_pos', False)}")
    print("\nIf every other token is at least g below the maximum, the model is correct with up to N other tokens:")
    print("  gap g | worst case over (q, m) | median")
    for g in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30):
        vals = [nmax[q][m][m - g + 1] for m in range(g - 1, V) for q in range(m + 1) if m - g + 1 >= 0]
        vals.sort()
        fmt = lambda x: "unbounded" if x >= INF else ("none" if x < 0 else f"{x:.3g}")
        print(f"  {g:5d} | {fmt(vals[0]):>22s} | {fmt(vals[len(vals) // 2])}")

    # explicit failures: one copy of m (as the query), N copies of m - 1
    fails = []
    for m in range(1, V):
        row = nmax[m][m]
        if 0 <= row[m] < INF:
            fails.append((row[m] + 1, m))
    if fails:
        n0, m0 = min(fails)
        print(f"\nsmallest adversarial bound: with query {m0}, {n0} other tokens below {m0} may defeat the maximum "
              f"(range over m: {min(fails)[0]} to {max(fails)[0]})")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        with torch.no_grad():
            for n in sorted({n0 - 1, n0, n0 + 1, 2 * n0}):
                if n < 1:
                    continue
                seq = torch.tensor([[m0 - 1] * n + [m0]], device=dev)
                pred = model(seq).argmax(-1).item()
                print(f"  input: {n} copies of {m0 - 1} then {m0}  ->  model answers {pred} ({'correct' if pred == m0 else 'WRONG'})")

    print("\nuniformly random inputs:  length | certified lower bound | sampled accuracy (2^17 inputs)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    g = torch.Generator(device=dev).manual_seed(11)
    for length in (4, 10, 20, 50, 100, 300, 1000):
        cert = certified_count(nmax, V, length)
        bound = cert / V ** length
        with torch.no_grad():
            correct = 0
            for _ in range(8):
                t = torch.randint(0, V, (1 << 14, length), device=dev, generator=g)
                correct += (model(t).argmax(-1) == t.max(-1).values).sum().item()
        print(f"  {length:28d} | {bound:21.6f} | {correct / (1 << 17):.6f}")


if __name__ == "__main__":
    main()
