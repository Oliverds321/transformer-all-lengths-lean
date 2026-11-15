"""One-layer, one-head, attention-only transformer for Max-of-K.

Input: K tokens in {0, ..., V-1}. Output: logits over V read at the last position;
the target is the maximum of the K tokens. No layer norm, no MLP, no biases.

    x_i     = E[t_i] + P[i]                                   (d_model)
    score_i = (x_K W_Q) . (x_i W_K) / sqrt(d_model)
    a       = softmax(score)                                   over the K positions
    out     = x_K + sum_i a_i (x_i W_V W_O)
    logits  = out W_U
"""
import math

import torch
import torch.nn as nn


class MaxOfK(nn.Module):
    def __init__(self, vocab=64, ctx=4, d_model=32, pos=True):
        super().__init__()
        self.vocab, self.ctx, self.d_model, self.pos = vocab, ctx, d_model, pos
        s = 1 / math.sqrt(d_model)
        self.E = nn.Parameter(torch.randn(vocab, d_model) * s)
        if pos:
            self.P = nn.Parameter(torch.randn(ctx, d_model) * s)
        else:                                       # no positional embedding: any input length
            self.register_buffer("P", torch.zeros(1, d_model))
        self.WQ = nn.Parameter(torch.randn(d_model, d_model) * s)
        self.WK = nn.Parameter(torch.randn(d_model, d_model) * s)
        self.WV = nn.Parameter(torch.randn(d_model, d_model) * s)
        self.WO = nn.Parameter(torch.randn(d_model, d_model) * s)
        self.WU = nn.Parameter(torch.randn(d_model, vocab) * s)

    def forward(self, tokens):                      # tokens: (batch, ctx) int64
        x = self.E[tokens] + self.P                 # (b, ctx, d)
        q = x[:, -1] @ self.WQ                      # (b, d)
        k = x @ self.WK                             # (b, ctx, d)
        score = torch.einsum("bd,bid->bi", q, k) / math.sqrt(self.d_model)
        a = torch.softmax(score, dim=-1)
        v = x @ self.WV @ self.WO                   # (b, ctx, d)
        out = x[:, -1] + torch.einsum("bi,bid->bd", a, v)
        return out @ self.WU

    @torch.no_grad()
    def tables(self):
        """The matrices a proof works with, in float64.

        score(q, t, i) = EQKE[q, t] + EQKP[q, i]      attention score of query token q
                                                      (at the last position) on token t at i
        logits         = direct[q] + sum_i a_i (EVOU[t_i] + PVOU[i])
        """
        d = self.d_model
        E, P = self.E.double(), self.P.double()
        qk = self.WQ.double() @ self.WK.double().T / math.sqrt(d)
        ov = self.WV.double() @ self.WO.double() @ self.WU.double()
        xq = E + P[-1]                              # residual of the query position, per token
        return {
            "EQKE": xq @ qk @ E.T,                  # (V, V)
            "EQKP": xq @ qk @ P.T,                  # (V, ctx)
            "EVOU": E @ ov,                         # (V, V)
            "PVOU": P @ ov,                         # (ctx, V)
            "direct": xq @ self.WU.double(),        # (V, V)
        }
