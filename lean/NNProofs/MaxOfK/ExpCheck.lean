import Mathlib
import NNProofs.ExpBound
import NNProofs.MaxOfK.Sound

/-!
# Checking the exponential bounds of a model

`M.expCheckQ n p q` compares the literal bounds stored in the model for the query
token `q` with the bounds computed by `ExpBound.expBounds`. If it evaluates to
`true` for every `q`, the model's bounds are valid (`IntModel.ExpOK`).
-/

namespace NNProofs.MaxOfK

open NNProofs.ExpBound

/-- The literal bounds `lo, hi` for `2 ^ F * exp ((x - c) / D)` are implied by the computed ones. -/
def expEntryOK (F n p D : ℕ) (x c : ℤ) (lo hi : ℕ) : Bool :=
  decide (x ≤ c) && decide ((c - x).toNat ≤ D * 2 ^ p) &&
    decide (lo ≤ (expBounds F n p (c - x).toNat D).1) &&
    decide ((expBounds F n p (c - x).toNat D).2 ≤ hi)

theorem expEntryOK_spec {F n p D : ℕ} {x c : ℤ} {lo hi : ℕ} (hD : 0 < D) (hn : 0 < n)
    (h : expEntryOK F n p D x c lo hi = true) :
    (lo : ℝ) ≤ 2 ^ F * Real.exp (((x - c : ℤ) : ℝ) / D) ∧
      2 ^ F * Real.exp (((x - c : ℤ) : ℝ) / D) ≤ (hi : ℝ) := by
  unfold expEntryOK at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  have hspec := expBounds_spec F n p (c - x).toNat D hD hn h2
  have hy : (((c - x).toNat : ℕ) : ℝ) = ((c - x : ℤ) : ℝ) := by
    have : (((c - x).toNat : ℕ) : ℤ) = c - x := Int.toNat_of_nonneg (by omega)
    exact_mod_cast this
  have harg : -(((c - x).toNat : ℕ) : ℝ) / D = ((x - c : ℤ) : ℝ) / D := by
    rw [hy]; push_cast; ring
  rw [harg] at hspec
  have h3' : (lo : ℝ) ≤ (((expBounds F n p (c - x).toNat D).1 : ℕ) : ℝ) := by exact_mod_cast h3
  have h4' : (((expBounds F n p (c - x).toNat D).2 : ℕ) : ℝ) ≤ (hi : ℝ) := by exact_mod_cast h4
  exact ⟨le_trans h3' hspec.1, le_trans hspec.2 h4'⟩

/-- All exponential bounds that belong to the query token `q`. -/
def IntModel.expCheckQ (M : IntModel) (n p q : ℕ) : Bool :=
  (allTo M.V fun t =>
      expEntryOK M.F n p M.D (M.eqke q t) (M.cTok q) (M.tokLo q t) (M.tokHi q t)) &&
    (allTo (M.K + 1) fun i =>
      expEntryOK M.F n p M.D (M.eqkp q i) (M.cPos q) (M.posLo q i) (M.posHi q i))

theorem IntModel.expOK_of_check (M : IntModel) (n p : ℕ) (hD : 0 < M.D) (hn : 0 < n)
    (h : ∀ q, q < M.V → M.expCheckQ n p q = true) : M.ExpOK where
  D_pos := hD
  tok q t hq ht := by
    have h1 := h q hq
    unfold IntModel.expCheckQ at h1
    rw [Bool.and_eq_true] at h1
    exact expEntryOK_spec hD hn ((allTo_iff _ _).mp h1.1 t ht)
  pos q i hq hi := by
    have h1 := h q hq
    unfold IntModel.expCheckQ at h1
    rw [Bool.and_eq_true] at h1
    exact expEntryOK_spec hD hn ((allTo_iff _ _).mp h1.2 i (Nat.lt_succ_of_le hi))

end NNProofs.MaxOfK
