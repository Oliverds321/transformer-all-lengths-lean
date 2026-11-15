import Mathlib
import NNProofs.MaxOfK.Model

/-!
# The transformer and its table form

A one-layer, one-head, attention-only transformer without biases or layer
normalisation, read at the last position, defined from its weight matrices. Its
logits coincide with the logits of the table-form model of
`NNProofs.MaxOfK.Model` for the tables `Transformer.toTables`.
-/

namespace NNProofs.MaxOfK

open Finset

/-- Weights: token embedding `E`, positional embedding `P`, query, key, value and
output matrices of the head, and the unembedding `WU`. -/
structure Transformer (V K d : ℕ) where
  E : Fin V → Fin d → ℝ
  P : Fin (K + 1) → Fin d → ℝ
  WQ : Fin d → Fin d → ℝ
  WK : Fin d → Fin d → ℝ
  WV : Fin d → Fin d → ℝ
  WO : Fin d → Fin d → ℝ
  WU : Fin d → Fin V → ℝ

namespace Transformer

variable {V K d : ℕ} (N : Transformer V K d)

/-- Residual stream before the head: token plus position embedding. -/
def resid (t : Fin V) (i : Fin (K + 1)) (a : Fin d) : ℝ := N.E t a + N.P i a

/-- Attention score of a query residual on a key residual. -/
def scoreOf (xq xk : Fin d → ℝ) : ℝ :=
  ∑ a, (∑ c, xq c * N.WQ c a) * (∑ c, xk c * N.WK c a)

/-- The head's value-output map applied to a residual, then unembedded to logit `j`. -/
def ovLogit (x : Fin d → ℝ) (j : Fin V) : ℝ :=
  ∑ a, (∑ e, (∑ c, x c * N.WV c e) * N.WO e a) * N.WU a j

/-- Attention score of the last position on position `i`. -/
def score (s : Fin (K + 1) → Fin V) (i : Fin (K + 1)) : ℝ :=
  N.scoreOf (N.resid (s (Fin.last K)) (Fin.last K)) (N.resid (s i) i)

/-- Attention weights: softmax of the scores over the positions. -/
noncomputable def attn (s : Fin (K + 1) → Fin V) (i : Fin (K + 1)) : ℝ :=
  Real.exp (N.score s i) / ∑ i', Real.exp (N.score s i')

/-- Residual stream after the head, at the last position. -/
noncomputable def out (s : Fin (K + 1) → Fin V) (a : Fin d) : ℝ :=
  N.resid (s (Fin.last K)) (Fin.last K) a +
    ∑ i, N.attn s i * ∑ e, (∑ c, N.resid (s i) i c * N.WV c e) * N.WO e a

/-- Logit `j` of the transformer on the input `s`. -/
noncomputable def logit (s : Fin (K + 1) → Fin V) (j : Fin V) : ℝ :=
  ∑ a, N.out s a * N.WU a j

/-- The tables of the transformer. -/
def toTables : Tables V (K + 1) where
  score q t i := N.scoreOf (N.resid q (Fin.last K)) (N.resid t i)
  value t i j := N.ovLogit (N.resid t i) j
  direct q j := ∑ a, N.resid q (Fin.last K) a * N.WU a j

/-- The transformer computes exactly the logits of its table form. -/
theorem logit_eq (s : Fin (K + 1) → Fin V) (j : Fin V) :
    N.logit s j = N.toTables.logit s j := by
  have hw : ∀ i, N.toTables.weight s i = Real.exp (N.score s i) := fun i => rfl
  unfold logit out Tables.logit
  simp only [hw]
  have h1 : ∑ a, (N.resid (s (Fin.last K)) (Fin.last K) a +
        ∑ i, N.attn s i * ∑ e, (∑ c, N.resid (s i) i c * N.WV c e) * N.WO e a) * N.WU a j =
      (∑ a, N.resid (s (Fin.last K)) (Fin.last K) a * N.WU a j) +
        ∑ i, N.attn s i * N.ovLogit (N.resid (s i) i) j := by
    have hsplit : ∀ a, (N.resid (s (Fin.last K)) (Fin.last K) a +
          ∑ i, N.attn s i * ∑ e, (∑ c, N.resid (s i) i c * N.WV c e) * N.WO e a) * N.WU a j =
        N.resid (s (Fin.last K)) (Fin.last K) a * N.WU a j +
          ∑ i, N.attn s i * ((∑ e, (∑ c, N.resid (s i) i c * N.WV c e) * N.WO e a) * N.WU a j) := by
      intro a
      rw [add_mul, Finset.sum_mul]
      congr 1
      exact Finset.sum_congr rfl fun i _ => mul_assoc _ _ _
    rw [Finset.sum_congr rfl fun a _ => hsplit a, Finset.sum_add_distrib, Finset.sum_comm]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.mul_sum]
    rfl
  rw [h1]
  have h2 : ∑ i, N.attn s i * N.ovLogit (N.resid (s i) i) j =
      (∑ i, Real.exp (N.score s i) * N.toTables.value (s i) i j) / ∑ i, Real.exp (N.score s i) := by
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun i _ => ?_
    unfold attn
    rw [div_mul_eq_mul_div]
    rfl
  rw [h2]
  rfl

end Transformer

end NNProofs.MaxOfK
