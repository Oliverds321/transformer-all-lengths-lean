import NNProofs.MaxOfK.Check

/-!
# Integer weights and matrix-product checks

The weights of the quantised transformer are integers with a common scale `S`
(the real weight is the integer divided by `S`). The checks below verify, by
kernel evaluation, that one integer table is the matrix product of two others.
No imports beyond the checker, so the kernel-evaluated modules stay small.
-/

namespace NNProofs.MaxOfK

structure IntWeights where
  d : Nat
  S : Nat
  E : Nat → Nat → Int
  P : Nat → Nat → Int
  WQ : Nat → Nat → Int
  WK : Nat → Nat → Int
  WV : Nat → Nat → Int
  WO : Nat → Nat → Int
  WU : Nat → Nat → Int

/-- Row `r` of `C` equals row `r` of the product: `C r c = ∑ a < d, A r a * B a c` for `c < cols`. -/
def rowMulOK (d cols : Nat) (A B C : Nat → Nat → Int) (r : Nat) : Bool :=
  allTo cols fun c => decide (C r c = sumTo d fun a => A r a * B a c)

/-- The same with a scale factor: `C r c = (∑ a < d, A r a * B a c) * k`. -/
def rowMulScaleOK (d cols : Nat) (k : Int) (A B C : Nat → Nat → Int) (r : Nat) : Bool :=
  allTo cols fun c => decide (C r c = (sumTo d fun a => A r a * B a c) * k)

/-- A property of rows, checked on the chunk of `c` rows starting at `k * c`. -/
def chunkOK (c : Nat) (p : Nat → Bool) (k : Nat) : Bool :=
  allTo c fun r => p (k * c + r)

end NNProofs.MaxOfK
