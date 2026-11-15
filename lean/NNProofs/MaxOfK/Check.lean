/-!
# The integer checker

Pure natural-number and integer computations, with no imports, so that the
kernel can evaluate them and the modules that do so stay small. Soundness is
proved in `NNProofs.MaxOfK.Sound`.

A model is given by integer tables with a common denominator `D`:

    score(q; t, i) = (eqke q t + eqkp q i) / D
    value(t, i; j) = (evou t j + pvou i j) / D
    direct(q; j)   = dir q j / D

and by integer bounds for the exponentials of the scores relative to reference
values `cTok q`, `cPos q`, with `F` fractional bits:

    tokLo q t ≤ 2^F exp((eqke q t - cTok q) / D) ≤ tokHi q t
    posLo q i ≤ 2^F exp((eqkp q i - cPos q) / D) ≤ posHi q i

The context has `K + 1` positions `0, ..., K`; the query token sits at position `K`.
-/

namespace NNProofs.MaxOfK

/-- `f 0 + ... + f (n - 1)`. -/
def sumTo : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumTo n f + f n

/-- `p 0 && ... && p (n - 1)`. -/
def allTo : Nat → (Nat → Bool) → Bool
  | 0, _ => true
  | n + 1, p => allTo n p && p n

/-- The least of `a, f 0, ..., f (n - 1)`. -/
def minTo : Nat → (Nat → Int) → Int → Int
  | 0, _, a => a
  | n + 1, f, a => min (minTo n f a) (f n)

/-- A lower bound of `E * v` when `lo ≤ E ≤ hi`. -/
def lb (lo hi : Nat) (v : Int) : Int :=
  if 0 ≤ v then (lo : Int) * v else (hi : Int) * v

structure IntModel where
  V : Nat
  K : Nat
  D : Nat
  F : Nat
  eqke : Nat → Nat → Int
  eqkp : Nat → Nat → Int
  evou : Nat → Nat → Int
  pvou : Nat → Nat → Int
  dir : Nat → Nat → Int
  cTok : Nat → Int
  cPos : Nat → Int
  tokLo : Nat → Nat → Nat
  tokHi : Nat → Nat → Nat
  posLo : Nat → Nat → Nat
  posHi : Nat → Nat → Nat

namespace IntModel

variable (M : IntModel)

/-- `D` times the contribution of token `t` at position `i` to the gap between
logits `m` and `j`, for query token `q`. -/
def v (q m j t i : Nat) : Int :=
  M.evou t m - M.evou t j + (M.pvou i m - M.pvou i j) + (M.dir q m - M.dir q j)

/-- Lower bound of the scaled term of token `t` at position `i`. -/
def w (q m j t i : Nat) : Int :=
  lb (M.tokLo q t * M.posLo q i) (M.tokHi q t * M.posHi q i) (M.v q m j t i)

/-- Worst term at a free position `i`: the token is `m` or below `b`. -/
def g (q m b j i : Nat) : Int :=
  minTo b (fun t => M.w q m j t i) (M.w q m j m i)

/-- The case (query `q`, maximum `m`, all other tokens below `b`) against the wrong answer `j`. -/
def checkJ (q m b j : Nat) : Bool :=
  if q = m then
    decide (0 < sumTo M.K (fun i => M.g q m b j i) + M.w q m j m M.K)
  else
    allTo M.K fun i0 =>
      decide (0 < sumTo M.K (fun i => M.g q m b j i) - M.g q m b j i0
        + M.w q m j m i0 + M.w q m j q M.K)

/-- The case against every wrong answer. -/
def check (q m b : Nat) : Bool :=
  allTo M.V fun j => j == m || M.checkJ q m b j

end IntModel

/-! ## Packed tables

A table is stored as one natural number: entry `k` occupies bits `w * k` to
`w * k + w - 1`. Signed entries are stored with an offset. -/

def unpack (w : Nat) (packed : Nat) (k : Nat) : Nat :=
  (packed >>> (w * k)) % 2 ^ w

def unpackInt (w : Nat) (off : Nat) (packed : Nat) (k : Nat) : Int :=
  (unpack w packed k : Int) - (off : Int)

end NNProofs.MaxOfK
