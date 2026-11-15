import Mathlib

/-!
# A one-layer attention-only model in table form

The output of a one-layer, one-head, attention-only transformer that reads its
answer at the last position depends on the weights only through three tables:

* `score q t i`: attention score of the query token `q` (the token at the last
  position) on token `t` at position `i`;
* `value t i j`: what token `t` at position `i` contributes to logit `j` through
  the head;
* `direct q j`: the direct path from the query token to logit `j`.

This file defines the logits from the tables, with real-valued softmax, and
proves the fact the accuracy bound rests on: the sign of a logit gap is the sign
of a sum over positions of a function of the token at that position alone.
-/

namespace NNProofs.MaxOfK

open Finset

/-- The tables of a model with vocabulary `V` and context length `K`. -/
structure Tables (V K : ℕ) where
  score : Fin V → Fin V → Fin K → ℝ
  value : Fin V → Fin K → Fin V → ℝ
  direct : Fin V → Fin V → ℝ

variable {V K : ℕ} (T : Tables V (K + 1))

/-- The query token: the token at the last position. -/
def query (s : Fin (K + 1) → Fin V) : Fin V := s (Fin.last K)

/-- Unnormalised attention weight on position `i`. -/
noncomputable def Tables.weight (s : Fin (K + 1) → Fin V) (i : Fin (K + 1)) : ℝ :=
  Real.exp (T.score (query s) (s i) i)

/-- Logit `j` on the input `s`: direct path plus the attention-weighted values. -/
noncomputable def Tables.logit (s : Fin (K + 1) → Fin V) (j : Fin V) : ℝ :=
  T.direct (query s) j + (∑ i, T.weight s i * T.value (s i) i j) / ∑ i, T.weight s i

/-- What position `i` holding token `t` contributes to the gap between logits `m`
and `j`, when the query token is `q`. -/
def Tables.gapTerm (q m j t : Fin V) (i : Fin (K + 1)) : ℝ :=
  T.value t i m - T.value t i j + (T.direct q m - T.direct q j)

theorem Tables.weight_pos (s : Fin (K + 1) → Fin V) (i : Fin (K + 1)) : 0 < T.weight s i :=
  Real.exp_pos _

theorem Tables.sum_weight_pos (s : Fin (K + 1) → Fin V) : 0 < ∑ i, T.weight s i :=
  Finset.sum_pos (fun i _ => T.weight_pos s i) Finset.univ_nonempty

/-- The gap between two logits is a positive multiple of a sum over positions. -/
theorem Tables.logit_sub (s : Fin (K + 1) → Fin V) (j m : Fin V) :
    T.logit s m - T.logit s j =
      (∑ i, T.weight s i * T.gapTerm (query s) m j (s i) i) / ∑ i, T.weight s i := by
  have hW : (∑ i, T.weight s i) ≠ 0 := (T.sum_weight_pos s).ne'
  have h1 : ∑ i, T.weight s i * T.gapTerm (query s) m j (s i) i =
      (∑ i, T.weight s i * T.value (s i) i m) - (∑ i, T.weight s i * T.value (s i) i j) +
        (T.direct (query s) m - T.direct (query s) j) * ∑ i, T.weight s i := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    unfold Tables.gapTerm
    ring
  rw [h1]
  unfold Tables.logit
  field_simp
  ring

/-- **Separability.** Logit `m` exceeds logit `j` exactly when the sum over
positions of `exp(score) * gapTerm` is positive. -/
theorem Tables.logit_lt_iff (s : Fin (K + 1) → Fin V) (j m : Fin V) :
    T.logit s j < T.logit s m ↔
      0 < ∑ i, T.weight s i * T.gapTerm (query s) m j (s i) i := by
  rw [← sub_pos, T.logit_sub s j m]
  constructor
  · intro h
    by_contra hneg
    exact absurd h (not_lt.mpr (div_nonpos_of_nonpos_of_nonneg (not_lt.mp hneg) (T.sum_weight_pos s).le))
  · intro h
    exact div_pos h (T.sum_weight_pos s)

/-- The model answers `m` on `s`: logit `m` is strictly above every other logit
(a tie counts as a wrong answer). -/
def Tables.Answers (s : Fin (K + 1) → Fin V) (m : Fin V) : Prop :=
  ∀ j, j ≠ m → T.logit s j < T.logit s m

/-- A lower bound for each position's term gives the answer. `L t i` may be any
lower bound of `c * exp(score) * gapTerm` with one positive constant `c` for the
whole sequence. -/
theorem Tables.answers_of_lower_bounds (s : Fin (K + 1) → Fin V) (m : Fin V)
    (c : ℝ) (hc : 0 < c) (L : Fin V → Fin (K + 1) → ℝ)
    (hL : ∀ j, j ≠ m → ∀ i, L j i ≤ c * (T.weight s i * T.gapTerm (query s) m j (s i) i))
    (hsum : ∀ j, j ≠ m → 0 < ∑ i, L j i) : T.Answers s m := by
  intro j hj
  rw [T.logit_lt_iff s j m]
  have h1 : ∑ i, L j i ≤ c * ∑ i, T.weight s i * T.gapTerm (query s) m j (s i) i := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => hL j hj i
  have h2 : 0 < c * ∑ i, T.weight s i * T.gapTerm (query s) m j (s i) i :=
    lt_of_lt_of_le (hsum j hj) h1
  exact (mul_pos_iff_of_pos_left hc).mp h2

end NNProofs.MaxOfK
