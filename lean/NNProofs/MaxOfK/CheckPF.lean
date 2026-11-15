import NNProofs.MaxOfK.Check

/-!
# The integer checker for a model without positional embeddings

Without positional embeddings the tables do not depend on the position, and the
same weights define the model at every input length. The certificates below are
therefore independent of the length:

* `region q m b N`: every input, of any length, whose query token is `q`, that
  contains `m`, whose other tokens are all below `b`, and that has at most `N`
  such other tokens outside the query position, is answered with `m`;
* `failSum`: an upper bound of the quantity whose sign decides between two
  answers, for a given vector of token counts.

Pure integer computations with no imports beyond the fixed-length checker.
-/

namespace NNProofs.MaxOfK

/-- An upper bound of `E * v` when `lo ≤ E ≤ hi`. -/
def ub (lo hi : Nat) (v : Int) : Int :=
  if 0 ≤ v then (hi : Int) * v else (lo : Int) * v

/-- Integer tables of a model without positional embeddings (common denominator `D`):
`score(q; t) = eqke q t / D`, `value(t; j) = evou t j / D`, `direct(q; j) = dir q j / D`,
and `tokLo q t ≤ 2^F exp((eqke q t - cTok q) / D) ≤ tokHi q t`. -/
structure IntPF where
  V : Nat
  D : Nat
  F : Nat
  eqke : Nat → Nat → Int
  evou : Nat → Nat → Int
  dir : Nat → Nat → Int
  cTok : Nat → Int
  tokLo : Nat → Nat → Nat
  tokHi : Nat → Nat → Nat

namespace IntPF

variable (M : IntPF)

/-- `D` times the contribution of token `t` to the gap between logits `m` and `j`. -/
def v (q m j t : Nat) : Int :=
  M.evou t m - M.evou t j + (M.dir q m - M.dir q j)

/-- Lower and upper bounds of the scaled term of one occurrence of token `t`. -/
def wLo (q m j t : Nat) : Int := lb (M.tokLo q t) (M.tokHi q t) (M.v q m j t)
def wHi (q m j t : Nat) : Int := ub (M.tokLo q t) (M.tokHi q t) (M.v q m j t)

/-- The worst term of a token below `b`, or zero if all are positive. -/
def gLo (q m b j : Nat) : Int := minTo b (fun t => M.wLo q m j t) 0

/-- The region certificate against one wrong answer `j`. -/
def regionJ (q m b N j : Nat) : Bool :=
  decide (0 ≤ M.wLo q m j m) &&
    (if q = m then decide (0 < M.wLo q m j m + (N : Int) * M.gLo q m b j)
     else decide (0 < M.wLo q m j m + M.wLo q m j q + (N : Int) * M.gLo q m b j))

/-- The region certificate: query `q`, maximum `m`, other tokens below `b`, at most `N` of them. -/
def region (q m b N : Nat) : Bool :=
  allTo M.V fun j => j == m || M.regionJ q m b N j

/-- Upper bound of the deciding sum for the token counts `n` (query position included). -/
def failSum (q m j : Nat) (n : Nat → Nat) : Int :=
  sumTo M.V fun t => (n t : Int) * M.wHi q m j t

end IntPF

/-- One row of a region table: `bt q m = 0` (no certificate) or the certificate
`region q m (bt q m - 1) N` holds. -/
def regRow (M : IntPF) (bt : Nat → Nat → Nat) (N q m : Nat) : Bool :=
  bt q m == 0 || M.region q m (bt q m - 1) N

/-- One row of a failure table: `nf m = 0` (no claim) or `nf m - 1` copies of the
token `m - 1` outweigh the query token `m` in favour of the answer `m - 1`. -/
def failRow (M : IntPF) (nf : Nat → Nat) (m : Nat) : Bool :=
  nf m == 0 ||
    (decide (M.wHi m m (m - 1) (m - 1) ≤ 0) &&
      decide (((nf m - 1 : Nat) : Int) * M.wHi m m (m - 1) (m - 1) + M.wHi m m (m - 1) m ≤ 0))

end NNProofs.MaxOfK
