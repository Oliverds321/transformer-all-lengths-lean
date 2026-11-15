"""Compare a floating-point checkpoint with its quantised model on random inputs.

The theorems are about the quantised model (weights = integers / 2^10, W_Q
divided by sqrt(d_model) before rounding). This script samples uniformly random
inputs of several lengths and counts how often the two models give different
answers. It is an experiment in floating point, not part of the proofs.

usage: python compare_quantised.py checkpoints/maxofk_v64_k10_s0_nopos.pt [more checkpoints]
"""
import math
import sys

import numpy as np
import torch

import exact_bound
from model import MaxOfK


def load_pair(path, bits=10):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    a = ck["args"]
    pos = not a.get("no_pos", False)
    model = MaxOfK(a["vocab"], a["ctx"], a["d_model"], pos=pos)
    model.load_state_dict(ck["state"])
    model = model.double()
    w = exact_bound.quantise(model, bits)
    quant = MaxOfK(a["vocab"], a["ctx"], a["d_model"], pos=pos).double()
    with torch.no_grad():
        names = ("E", "P", "WK", "WV", "WO", "WU") if pos else ("E", "WK", "WV", "WO", "WU")
        for name in names:
            getattr(quant, name).copy_(torch.tensor(np.array(w[name], dtype=np.float64) / w["S"]))
        quant.WQ.copy_(torch.tensor(np.array(w["WQ"], dtype=np.float64) / w["S"]) * math.sqrt(a["d_model"]))
    return model, quant, a, pos


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for path in sys.argv[1:]:
        model, quant, a, pos = load_pair(path)
        model, quant = model.to(dev), quant.to(dev)
        gen = torch.Generator(device=dev).manual_seed(123)
        lengths = (4, 10, 100, 1000) if not pos else (a["ctx"],)
        row = []
        with torch.no_grad():
            for length in lengths:
                n = 1 << 19 if length <= 100 else 1 << 16
                differ = right_f = right_q = 0
                for _ in range(n >> 14):
                    t = torch.randint(0, a["vocab"], (1 << 14, length), device=dev, generator=gen)
                    pf, pq, mx = model(t).argmax(-1), quant(t).argmax(-1), t.max(-1).values
                    differ += (pf != pq).sum().item()
                    right_f += (pf == mx).sum().item()
                    right_q += (pq == mx).sum().item()
                row.append(f"length {length}: float {right_f / n:.6f}, quantised {right_q / n:.6f}, "
                           f"different answers {differ} of {n}")
            if not pos:
                # every input of length 4
                V = a["vocab"]
                n = V ** 4
                differ = right_f = right_q = 0
                tail = torch.cartesian_prod(*[torch.arange(V, device=dev)] * 3)        # (V^3, 3)
                for first in range(V):
                    t = torch.cat([torch.full((V ** 3, 1), first, device=dev), tail], dim=1)
                    pf, pq, mx = model(t).argmax(-1), quant(t).argmax(-1), t.max(-1).values
                    differ += (pf != pq).sum().item()
                    right_f += (pf == mx).sum().item()
                    right_q += (pq == mx).sum().item()
                row.append(f"length 4, all {n} inputs: float {right_f / n:.6f} ({right_f}), "
                           f"quantised {right_q / n:.6f} ({right_q}), different answers {differ}")
        print(path)
        for r in row:
            print("  " + r)


if __name__ == "__main__":
    main()
