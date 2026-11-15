import Mathlib
import NNProofs.MaxOfK.Sound
import NNProofs.MaxOfK.ExpCheck
import NNProofs.MaxOfK.CheckPF

/-!
# A model without positional embeddings, at every input length

`M.tables L` is the real-valued model on inputs of length `L + 1`; the weights do
not depend on `L`. This file proves

* `answers_iff_counts`: on every input of every length, whether the model answers
  `m` is decided by linear inequalities in the token counts of the input;
* `region_sound`: a region certificate proves correctness at every length;
* `not_answers_of_upper`: an upper-bound certificate proves a wrong answer.
-/

namespace NNProofs.MaxOfK

open Finset

theorem le_ub {lo hi : ℕ} {v : ℤ} {E : ℝ} (h1 : (lo : ℝ) ≤ E) (h2 : E ≤ (hi : ℝ)) :
    E * (v : ℝ) ≤ ((ub lo hi v : ℤ) : ℝ) := by
  unfold ub
  split_ifs with hv
  · push_cast
    have h0 : (0 : ℝ) ≤ (v : ℝ) := by exact_mod_cast hv
    exact mul_le_mul_of_nonneg_right h2 h0
  · push_cast
    have h0 : (v : ℝ) ≤ 0 := by exact_mod_cast (not_le.mp hv).le
    exact mul_le_mul_of_nonpos_right h1 h0

/-- A sum over positions of a function of the token is a sum over tokens weighted by counts. -/
theorem sum_by_token {n V : ℕ} (s : Fin n → Fin V) (f : Fin V → ℝ) :
    ∑ i, f (s i) = ∑ t, ((Finset.univ.filter fun i => s i = t).card : ℝ) * f t := by
  rw [← Finset.sum_fiberwise_of_maps_to (g := s) (t := Finset.univ) (fun _ _ => Finset.mem_univ _)]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Finset.sum_congr rfl (fun i hi => by rw [(Finset.mem_filter.mp hi).2]), Finset.sum_const,
    nsmul_eq_mul]

namespace IntPF

variable (M : IntPF)

/-- The real-valued model on inputs of length `L + 1`. -/
noncomputable def tables (L : ℕ) : Tables M.V (L + 1) where
  score q t _ := ((M.eqke q t : ℤ) : ℝ) / M.D
  value t _ j := ((M.evou t j : ℤ) : ℝ) / M.D
  direct q j := ((M.dir q j : ℤ) : ℝ) / M.D

/-- The exponential bounds of the model are valid. -/
structure ExpOK : Prop where
  D_pos : 0 < M.D
  tok : ∀ q t, q < M.V → t < M.V →
    (M.tokLo q t : ℝ) ≤ 2 ^ M.F * Real.exp (((M.eqke q t - M.cTok q : ℤ) : ℝ) / M.D) ∧
      2 ^ M.F * Real.exp (((M.eqke q t - M.cTok q : ℤ) : ℝ) / M.D) ≤ (M.tokHi q t : ℝ)

/-- The exponential bounds that belong to the query token `q`, checked against `expBounds`. -/
def expCheckQ (n p q : ℕ) : Bool :=
  allTo M.V fun t =>
    expEntryOK M.F n p M.D (M.eqke q t) (M.cTok q) (M.tokLo q t) (M.tokHi q t)

theorem expOK_of_check (n p : ℕ) (hD : 0 < M.D) (hn : 0 < n)
    (h : ∀ q, q < M.V → M.expCheckQ n p q = true) : M.ExpOK where
  D_pos := hD
  tok q t hq ht := expEntryOK_spec hD hn ((allTo_iff _ _).mp (h q hq) t ht)

/-- The positive constant that relates the integer terms to the real ones. -/
noncomputable def cst (q : ℕ) : ℝ :=
  2 ^ M.F * Real.exp (-((M.cTok q : ℤ) : ℝ) / M.D) * M.D

theorem cst_pos (h : M.ExpOK) (q : ℕ) : 0 < M.cst q := by
  have hD : (0 : ℝ) < M.D := by exact_mod_cast h.D_pos
  unfold cst
  positivity

theorem gapTerm_eq (L : ℕ) (q m j t : Fin M.V) (i : Fin (L + 1)) :
    (M.tables L).gapTerm q m j t i = ((M.v q m j t : ℤ) : ℝ) / M.D := by
  unfold Tables.gapTerm tables v
  push_cast
  ring

/-- The scaled real term of one position, as (scaled exponential) times (integer value). -/
theorem term_eq (h : M.ExpOK) (L : ℕ) (s : Fin (L + 1) → Fin M.V) (m j : Fin M.V)
    (i : Fin (L + 1)) :
    M.cst (query s) * ((M.tables L).weight s i * (M.tables L).gapTerm (query s) m j (s i) i) =
      (2 ^ M.F * Real.exp (((M.eqke (query s) (s i) - M.cTok (query s) : ℤ) : ℝ) / M.D)) *
        ((M.v (query s) m j (s i) : ℤ) : ℝ) := by
  have hD : (M.D : ℝ) ≠ 0 := by exact_mod_cast h.D_pos.ne'
  rw [M.gapTerm_eq]
  have hexp : Real.exp (((M.eqke (query s) (s i) - M.cTok (query s) : ℤ) : ℝ) / M.D) =
      Real.exp (-((M.cTok (query s) : ℤ) : ℝ) / M.D) * (M.tables L).weight s i := by
    unfold Tables.weight tables
    rw [← Real.exp_add]
    congr 1
    push_cast
    ring
  rw [hexp]
  unfold cst
  field_simp

theorem wLo_le (h : M.ExpOK) (L : ℕ) (s : Fin (L + 1) → Fin M.V) (m j : Fin M.V)
    (i : Fin (L + 1)) :
    ((M.wLo (query s) m j (s i) : ℤ) : ℝ) ≤
      M.cst (query s) * ((M.tables L).weight s i * (M.tables L).gapTerm (query s) m j (s i) i) := by
  obtain ⟨t1, t2⟩ := h.tok (query s) (s i) (query s).isLt (s i).isLt
  rw [M.term_eq h]
  exact lb_le t1 t2

theorem le_wHi (h : M.ExpOK) (L : ℕ) (s : Fin (L + 1) → Fin M.V) (m j : Fin M.V)
    (i : Fin (L + 1)) :
    M.cst (query s) * ((M.tables L).weight s i * (M.tables L).gapTerm (query s) m j (s i) i) ≤
      ((M.wHi (query s) m j (s i) : ℤ) : ℝ) := by
  obtain ⟨t1, t2⟩ := h.tok (query s) (s i) (query s).isLt (s i).isLt
  rw [M.term_eq h]
  exact le_ub t1 t2

/-- **Exact characterisation at every length.** The model answers `m` on `s` exactly
when, for every other token `j`, a linear form in the token counts of `s` is positive. -/
theorem answers_iff_counts (L : ℕ) (s : Fin (L + 1) → Fin M.V) (m : Fin M.V) :
    (M.tables L).Answers s m ↔ ∀ j, j ≠ m →
      0 < ∑ t : Fin M.V, ((Finset.univ.filter fun i => s i = t).card : ℝ) *
        (Real.exp (((M.eqke (query s) t : ℤ) : ℝ) / M.D) *
          (((M.v (query s) m j t : ℤ) : ℝ) / M.D)) := by
  unfold Tables.Answers
  refine forall_congr' fun j => forall_congr' fun _ => ?_
  rw [(M.tables L).logit_lt_iff s j m, ← sum_by_token s
    (fun t => Real.exp (((M.eqke (query s) t : ℤ) : ℝ) / M.D) *
      (((M.v (query s) m j t : ℤ) : ℝ) / M.D))]
  have e : ∀ i, (M.tables L).weight s i * (M.tables L).gapTerm (query s) m j (s i) i =
      Real.exp (((M.eqke (query s) (s i) : ℤ) : ℝ) / M.D) *
        (((M.v (query s) m j (s i) : ℤ) : ℝ) / M.D) := by
    intro i
    rw [M.gapTerm_eq]
    rfl
  rw [Finset.sum_congr rfl fun i _ => e i]

/-- **Soundness of a region certificate, at every length.** -/
theorem region_sound (h : M.ExpOK) (q m : Fin M.V) (b N : ℕ) (hc : M.region q m b N = true)
    (L : ℕ) (s : Fin (L + 1) → Fin M.V) (hq : query s = q) (hmax : ∃ i, s i = m)
    (hothers : ∀ i, s i = m ∨ (s i : ℕ) < b)
    (hN : (Finset.univ.filter fun i : Fin L => s (Fin.castSucc i) ≠ m).card ≤ N) :
    (M.tables L).Answers s m := by
  apply (M.tables L).answers_of_lower_bounds s m (M.cst q) (M.cst_pos h q)
    (fun j i => ((M.wLo q m j (s i) : ℤ) : ℝ))
  · intro j _ i
    have hw := M.wLo_le h L s m j i
    rw [hq] at hw
    have hg : (M.tables L).gapTerm (query s) m j (s i) i = (M.tables L).gapTerm q m j (s i) i := by
      rw [hq]
    rw [hg]
    exact hw
  · intro j hj
    have hjm : (j : ℕ) ≠ (m : ℕ) := fun e => hj (Fin.ext e)
    have hcj : M.regionJ q m b N j = true := by
      have h1 := (allTo_iff _ _).mp hc j j.isLt
      rcases Bool.or_eq_true _ _ |>.mp h1 with h2 | h2
      · exact absurd (by simpa using h2) hjm
      · exact h2
    unfold regionJ at hcj
    rw [Bool.and_eq_true] at hcj
    obtain ⟨hm0', hcond⟩ := hcj
    have hm0 : 0 ≤ M.wLo q m j m := of_decide_eq_true hm0'
    have hG0 : M.gLo q m b j ≤ 0 := minTo_le_init _ _ _
    have hint : (0 : ℤ) < ∑ i : Fin (L + 1), M.wLo q m j (s i) := by
      rw [Fin.sum_univ_castSucc]
      have hlast : s (Fin.last L) = q := hq
      rw [hlast]
      -- split the free positions into copies of `m` and the others
      rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i : Fin L => s (Fin.castSucc i) = m)]
      have hA : ∀ i ∈ Finset.univ.filter (fun i : Fin L => s (Fin.castSucc i) = m),
          (0 : ℤ) ≤ M.wLo q m j (s (Fin.castSucc i)) := by
        intro i hi
        rw [(Finset.mem_filter.mp hi).2]
        exact hm0
      have hA0 : (0 : ℤ) ≤ ∑ i ∈ Finset.univ.filter (fun i : Fin L => s (Fin.castSucc i) = m),
          M.wLo q m j (s (Fin.castSucc i)) := Finset.sum_nonneg hA
      have hB : (N : ℤ) * M.gLo q m b j ≤
          ∑ i ∈ Finset.univ.filter (fun i : Fin L => ¬ s (Fin.castSucc i) = m),
            M.wLo q m j (s (Fin.castSucc i)) := by
        have h1 : (Finset.univ.filter (fun i : Fin L => ¬ s (Fin.castSucc i) = m)).card • M.gLo q m b j ≤
            ∑ i ∈ Finset.univ.filter (fun i : Fin L => ¬ s (Fin.castSucc i) = m),
              M.wLo q m j (s (Fin.castSucc i)) := by
          apply Finset.card_nsmul_le_sum
          intro i hi
          have hne := (Finset.mem_filter.mp hi).2
          rcases hothers (Fin.castSucc i) with e | e
          · exact absurd e hne
          · exact minTo_le _ (fun t => M.wLo q m j t) _ _ e
        rw [nsmul_eq_mul] at h1
        have h2 : (N : ℤ) * M.gLo q m b j ≤
            ((Finset.univ.filter (fun i : Fin L => ¬ s (Fin.castSucc i) = m)).card : ℤ) *
              M.gLo q m b j :=
          mul_le_mul_of_nonpos_right (by exact_mod_cast hN) hG0
        exact le_trans h2 h1
      by_cases hqm : (q : ℕ) = (m : ℕ)
      · rw [if_pos hqm] at hcond
        have h2 := of_decide_eq_true hcond
        have e : M.wLo q m j (q : ℕ) = M.wLo q m j (m : ℕ) := by rw [hqm]
        rw [e]
        omega
      · rw [if_neg hqm] at hcond
        have h2 := of_decide_eq_true hcond
        obtain ⟨i, hi⟩ := hmax
        have hne : i ≠ Fin.last L := by
          intro e
          rw [e, hlast] at hi
          exact hqm (congrArg Fin.val hi)
        obtain ⟨i0, hi0⟩ := Fin.exists_castSucc_eq.mpr hne
        have hmem : i0 ∈ Finset.univ.filter (fun i : Fin L => s (Fin.castSucc i) = m) := by
          rw [Finset.mem_filter]
          exact ⟨Finset.mem_univ _, by rw [hi0]; exact hi⟩
        have hsingle := Finset.single_le_sum hA hmem
        have hterm : M.wLo q m j (s (Fin.castSucc i0)) = M.wLo q m j m := by rw [hi0, hi]
        rw [hterm] at hsingle
        omega
    have hcast : ((∑ i : Fin (L + 1), M.wLo q m j (s i) : ℤ) : ℝ) =
        ∑ i : Fin (L + 1), ((M.wLo q m j (s i) : ℤ) : ℝ) := by push_cast; rfl
    rw [← hcast]
    exact_mod_cast hint

/-- **A wrong answer.** If the upper bounds of the terms sum to at most zero for some
`j ≠ m`, then logit `m` is not above logit `j`: the model does not answer `m`. -/
theorem not_answers_of_upper (h : M.ExpOK) (L : ℕ) (s : Fin (L + 1) → Fin M.V) (m j : Fin M.V)
    (hj : j ≠ m) (hsum : ∑ i, M.wHi (query s) m j (s i) ≤ 0) : ¬ (M.tables L).Answers s m := by
  intro hA
  have h1 := ((M.tables L).logit_lt_iff s j m).mp (hA j hj)
  have h2 : 0 < M.cst (query s) *
      ∑ i, (M.tables L).weight s i * (M.tables L).gapTerm (query s) m j (s i) i :=
    mul_pos (M.cst_pos h _) h1
  rw [Finset.mul_sum] at h2
  have h3 : ∑ i, M.cst (query s) *
      ((M.tables L).weight s i * (M.tables L).gapTerm (query s) m j (s i) i) ≤
      ∑ i, ((M.wHi (query s) m j (s i) : ℤ) : ℝ) :=
    Finset.sum_le_sum fun i _ => M.le_wHi h L s m j i
  have h4 : (∑ i, ((M.wHi (query s) m j (s i) : ℤ) : ℝ)) ≤ 0 := by
    have : ((∑ i, M.wHi (query s) m j (s i) : ℤ) : ℝ) ≤ 0 := by exact_mod_cast hsum
    push_cast at this
    exact this
  linarith

end IntPF

end NNProofs.MaxOfK
