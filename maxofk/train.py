"""Train a Max-of-K transformer on uniformly random sequences.

usage: python train.py --ctx 4 [--vocab 64] [--d-model 32] [--steps 30000] [--seed 0]
Writes checkpoints/maxofk_v{vocab}_k{ctx}_s{seed}.pt and prints sampled accuracy
(and the exact accuracy over all vocab**ctx inputs when that is at most 2**28).
"""
import argparse
import itertools
import os
import time

import torch
import torch.nn.functional as F

from model import MaxOfK


def exact_accuracy(model, device, chunk=1 << 20):
    v, k = model.vocab, model.ctx
    total = v ** k
    correct = 0
    for start in range(0, total, chunk):
        idx = torch.arange(start, min(start + chunk, total), device=device)
        toks = torch.stack([(idx // v ** (k - 1 - j)) % v for j in range(k)], dim=1)
        correct += (model(toks).argmax(-1) == toks.max(-1).values).sum().item()
    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pos", action="store_true", help="no positional embedding")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = MaxOfK(args.vocab, args.ctx, args.d_model, pos=not args.no_pos).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    start = time.time()
    for step in range(1, args.steps + 1):
        toks = torch.randint(0, args.vocab, (args.batch, args.ctx), device=device, generator=gen)
        loss = F.cross_entropy(model(toks), toks.max(-1).values)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 5000 == 0 or step == args.steps:
            print(f"step {step:6d}  loss {loss.item():.3e}  {time.time() - start:.0f}s", flush=True)

    model.eval()
    with torch.no_grad():
        toks = torch.randint(0, args.vocab, (1 << 20, args.ctx), device=device, generator=gen)
        acc = (model(toks).argmax(-1) == toks.max(-1).values).float().mean().item()
        print(f"sampled accuracy on 2^20 inputs: {acc:.6f}")
        if args.vocab ** args.ctx <= 1 << 28:
            c, t = exact_accuracy(model, device)
            print(f"exact accuracy over all {t} inputs: {c}/{t} = {c / t:.8f}")

    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/maxofk_v{args.vocab}_k{args.ctx}_s{args.seed}{'_nopos' if args.no_pos else ''}.pt"
    torch.save({"state": model.state_dict(), "args": vars(args)}, path)
    print("saved", path, f"({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
