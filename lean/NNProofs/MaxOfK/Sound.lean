import Mathlib
import NNProofs.MaxOfK.Model
import NNProofs.MaxOfK.Check

/-!
# Soundness of the integer checker

If the exponential bounds of an `IntModel` are valid and `check q m b` evaluates
to `true`, then the real-valued model answers `m` on every sequence whose query
token is `q`, that contains `m`, and all of whose other tokens are below `b`.
-/

namespace NNProofs.MaxOfK

open Finset

theorem sumTo_eq (n : ℕ) (f : ℕ → ℤ) : sumTo n f = ∑ i ∈ range n, f i := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => rw [sumTo, Finset.sum_range_succ, ih]

theorem allTo_iff (n : ℕ) (p : ℕ → Bool) : allTo n p = true ↔ ∀ i, i < n → p i = true := by
  induction n with
  | zero => simp [allTo]
  | succ n ih =>
    rw [allTo, Bool.and_eq_true, ih]
    constructor
    · rintro ⟨h1, h2⟩ i hi
      rcases Nat.lt_succ_iff_lt_or_eq.mp hi with h | h
      · exact h1 i h
      · rw [h]; exact h2
    · intro h
      exact ⟨fun i hi => h i (Nat.lt_succ_of_lt hi), h n (Nat.lt_succ_self n)⟩

theorem minTo_le_init (n : ℕ) (f : ℕ → ℤ) (a : ℤ) : minTo n f a ≤ a := by
  induction n with
  | zero => simp [minTo]
  | succ n ih => rw [minTo]; exact le_trans (Int.min_le_left _ _) ih

theorem minTo_le (n : ℕ) (f : ℕ → ℤ) (a : ℤ) (i : ℕ) (hi : i < n) : minTo n f a ≤ f i := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [minTo]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hi with h | h
    · exact le_trans (Int.min_le_left _ _) (ih h)
    · rw [h]; exact Int.min_le_right _ _

theorem lb_le {lo hi : ℕ} {v : ℤ} {E : ℝ} (h1 : (lo : ℝ) ≤ E) (h2 : E ≤ (hi : ℝ)) :
    ((lb lo hi v : ℤ) : ℝ) ≤ E * (v : ℝ) := by
  unfold lb
  split_ifs with hv
  · push_cast
    have h0 : (0 : ℝ) ≤ (v : ℝ) := by exact_mod_cast hv
    exact mul_le_mul_of_nonneg_right h1 h0
  · push_cast
    have h0 : (v : ℝ) ≤ 0 := by exact_mod_cast (not_le.mp hv).le
    exact mul_le_mul_of_nonpos_right h2 h0

namespace IntModel

variable (M : IntModel)

/-- The real-valued model described by the integer tables. -/
noncomputable def tables : Tables M.V (M.K + 1) where
  score q t i := ((M.eqke q t + M.eqkp q i : ℤ) : ℝ) / M.D
  value t i j := ((M.evou t j + M.pvou i j : ℤ) : ℝ) / M.D
  direct q j := ((M.dir q j : ℤ) : ℝ) / M.D

/-- The exponential bounds of the model are valid. -/
structure ExpOK : Prop where
  D_pos : 0 < M.D
  tok : ∀ q t, q < M.V → t < M.V →
    (M.tokLo q t : ℝ) ≤ 2 ^ M.F * Real.exp (((M.eqke q t - M.cTok q : ℤ) : ℝ) / M.D) ∧
      2 ^ M.F * Real.exp (((M.eqke q t - M.cTok q : ℤ) : ℝ) / M.D) ≤ (M.tokHi q t : ℝ)
  pos : ∀ q i, q < M.V → i ≤ M.K →
    (M.posLo q i : ℝ) ≤ 2 ^ M.F * Real.exp (((M.eqkp q i - M.cPos q : ℤ) : ℝ) / M.D) ∧
      2 ^ M.F * Real.exp (((M.eqkp q i - M.cPos q : ℤ) : ℝ) / M.D) ≤ (M.posHi q i : ℝ)

/-- The positive constant that relates the integer terms to the real ones. -/
noncomputable def cst (q : ℕ) : ℝ :=
  2 ^ M.F * 2 ^ M.F * Real.exp (-((M.cTok q + M.cPos q : ℤ) : ℝ) / M.D) * M.D

theorem gapTerm_eq (q m j t : Fin M.V) (i : Fin (M.K + 1)) :
    M.tables.gapTerm q m j t i = ((M.v q m j t i : ℤ) : ℝ) / M.D := by
  unfold Tables.gapTerm tables v
  push_cast
  ring

theorem w_le (h : M.ExpOK) (s : Fin (M.K + 1) → Fin M.V) (m j : Fin M.V) (i : Fin (M.K + 1)) :
    ((M.w (query s) m j (s i) i : ℤ) : ℝ) ≤
      M.cst (query s) * (M.tables.weight s i * M.tables.gapTerm (query s) m j (s i) i) := by
  have hD : (0 : ℝ) < M.D := by exact_mod_cast h.D_pos
  obtain ⟨t1, t2⟩ := h.tok (query s) (s i) (query s).isLt (s i).isLt
  obtain ⟨p1, p2⟩ := h.pos (query s) i (query s).isLt (Nat.lt_succ_iff.mp i.isLt)
  set A : ℝ := 2 ^ M.F * Real.exp (((M.eqke (query s) (s i) - M.cTok (query s) : ℤ) : ℝ) / M.D) with hA
  set B : ℝ := 2 ^ M.F * Real.exp (((M.eqkp (query s) i - M.cPos (query s) : ℤ) : ℝ) / M.D) with hB
  have hA0 : 0 ≤ A := by positivity
  have hB0 : 0 ≤ B := by positivity
  have hlo : ((M.tokLo (query s) (s i) * M.posLo (query s) i : ℕ) : ℝ) ≤ A * B := by
    push_cast
    exact mul_le_mul t1 p1 (Nat.cast_nonneg _) hA0
  have hhi : A * B ≤ ((M.tokHi (query s) (s i) * M.posHi (query s) i : ℕ) : ℝ) := by
    push_cast
    exact mul_le_mul t2 p2 hB0 (Nat.cast_nonneg _)
  have hw := lb_le (v := M.v (query s) m j (s i) i) hlo hhi
  have key : A * B * ((M.v (query s) m j (s i) i : ℤ) : ℝ) =
      M.cst (query s) * (M.tables.weight s i * M.tables.gapTerm (query s) m j (s i) i) := by
    rw [M.gapTerm_eq]
    have hexp : Real.exp (((M.eqke (query s) (s i) - M.cTok (query s) : ℤ) : ℝ) / M.D) *
        Real.exp (((M.eqkp (query s) i - M.cPos (query s) : ℤ) : ℝ) / M.D) =
        Real.exp (-((M.cTok (query s) + M.cPos (query s) : ℤ) : ℝ) / M.D) * M.tables.weight s i := by
      unfold Tables.weight tables
      rw [← Real.exp_add, ← Real.exp_add]
      congr 1
      push_cast
      ring
    have hAB : A * B = 2 ^ M.F * 2 ^ M.F *
        (Real.exp (-((M.cTok (query s) + M.cPos (query s) : ℤ) : ℝ) / M.D) * M.tables.weight s i) := by
      rw [← hexp, hA, hB]; ring
    rw [hAB]
    unfold cst
    field_simp
  unfold w
  rw [← key]
  exact hw

theorem cst_pos (h : M.ExpOK) (q : ℕ) : 0 < M.cst q := by
  have hD : (0 : ℝ) < M.D := by exact_mod_cast h.D_pos
  unfold cst
  positivity

/-- **Soundness of one case.** -/
theorem case_sound (h : M.ExpOK) (q m : Fin M.V) (b : ℕ) (hc : M.check q m b = true)
    (s : Fin (M.K + 1) → Fin M.V) (hq : query s = q) (hmax : ∃ i, s i = m)
    (hothers : ∀ i, s i = m ∨ (s i : ℕ) < b) : M.tables.Answers s m := by
  apply M.tables.answers_of_lower_bounds s m (M.cst q) (M.cst_pos h q)
    (fun j i => ((M.w q m j (s i) i : ℤ) : ℝ))
  · intro j _ i
    have hw := M.w_le h s m j i
    rw [hq] at hw
    have hg : M.tables.gapTerm (query s) m j (s i) i = M.tables.gapTerm q m j (s i) i := by rw [hq]
    rw [hg]
    exact hw
  · intro j hj
    have hjm : (j : ℕ) ≠ (m : ℕ) := fun e => hj (Fin.ext e)
    have hcj : M.checkJ q m b j = true := by
      have h1 := (allTo_iff _ _).mp hc j j.isLt
      rcases Bool.or_eq_true _ _ |>.mp h1 with h2 | h2
      · exact absurd (by simpa using h2) hjm
      · exact h2
    -- the integer inequality
    have hint : (0 : ℤ) < ∑ i : Fin (M.K + 1), M.w q m j (s i) i := by
      rw [Fin.sum_univ_castSucc]
      have hlast : s (Fin.last M.K) = q := hq
      have hG : ∀ i : Fin M.K, M.g q m b j i ≤ M.w q m j (s (Fin.castSucc i)) i := by
        intro i
        unfold g
        rcases hothers (Fin.castSucc i) with e | e
        · rw [e]; exact minTo_le_init _ _ _
        · exact minTo_le _ (fun t => M.w q m j t i) _ _ e
      have hsumG : sumTo M.K (fun i => M.g q m b j i) = ∑ i : Fin M.K, M.g q m b j i := by
        rw [sumTo_eq, Finset.sum_range]
      have hle : ∑ i : Fin M.K, M.g q m b j i ≤ ∑ i : Fin M.K, M.w q m j (s (Fin.castSucc i)) i :=
        Finset.sum_le_sum fun i _ => hG i
      simp only [Fin.val_castSucc, Fin.val_last, hlast]
      by_cases hqm : (q : ℕ) = (m : ℕ)
      · unfold checkJ at hcj
        rw [if_pos hqm, hsumG] at hcj
        have h2 := of_decide_eq_true hcj
        have hqm' : q = m := Fin.ext hqm
        rw [hqm']
        rw [hqm'] at h2 hle
        omega
      · unfold checkJ at hcj
        rw [if_neg hqm] at hcj
        obtain ⟨i, hi⟩ := hmax
        have hne : i ≠ Fin.last M.K := by
          intro e
          rw [e, hlast] at hi
          exact hqm (congrArg Fin.val hi)
        obtain ⟨i0, hi0⟩ := Fin.exists_castSucc_eq.mpr hne
        have h2 := of_decide_eq_true ((allTo_iff _ _).mp hcj i0 i0.isLt)
        rw [hsumG] at h2
        have hterm : M.w q m j (s (Fin.castSucc i0)) i0 = M.w q m j m i0 := by rw [hi0, hi]
        have hsingle : M.w q m j (s (Fin.castSucc i0)) i0 - M.g q m b j i0 ≤
            ∑ i : Fin M.K, (M.w q m j (s (Fin.castSucc i)) i - M.g q m b j i) :=
          Finset.single_le_sum (f := fun i : Fin M.K => M.w q m j (s (Fin.castSucc i)) i - M.g q m b j i)
            (fun i _ => sub_nonneg.mpr (hG i)) (Finset.mem_univ i0)
        rw [Finset.sum_sub_distrib] at hsingle
        omega
    have hcast : ((∑ i : Fin (M.K + 1), M.w q m j (s i) i : ℤ) : ℝ) =
        ∑ i : Fin (M.K + 1), ((M.w q m j (s i) i : ℤ) : ℝ) := by push_cast; rfl
    rw [← hcast]
    exact_mod_cast hint

end IntModel

end NNProofs.MaxOfK
