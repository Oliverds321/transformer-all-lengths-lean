import Mathlib
import NNProofs.MaxOfK.Model

/-!
# Counting the sequences covered by a certificate table

A certificate table `bt` assigns to a query token `q` and a maximum `m` a number
`bt q m`: zero means "not certified", and `c > 0` means that every sequence with
query token `q`, maximum `m` and all other tokens below `c - 1` is certified.
This file counts the covered sequences:

    #covered = ∑ q, ∑ m, cnt K q m (bt q m).
-/

namespace NNProofs.MaxOfK

open Finset

variable {V K : ℕ}

/-- The largest token of a sequence. -/
def seqMax (s : Fin (K + 1) → Fin V) : Fin V := Finset.univ.sup' Finset.univ_nonempty s

theorem le_seqMax (s : Fin (K + 1) → Fin V) (i : Fin (K + 1)) : s i ≤ seqMax s :=
  Finset.le_sup' s (Finset.mem_univ i)

theorem exists_eq_seqMax (s : Fin (K + 1) → Fin V) : ∃ i, s i = seqMax s := by
  obtain ⟨i, _, hi⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty s
  exact ⟨i, hi.symm⟩

theorem seqMax_eq_of (s : Fin (K + 1) → Fin V) (m : Fin V) (hm : ∃ i, s i = m)
    (hle : ∀ i, s i ≤ m) : seqMax s = m := by
  apply le_antisymm
  · exact Finset.sup'_le _ _ fun i _ => hle i
  · obtain ⟨i, hi⟩ := hm
    rw [← hi]
    exact le_seqMax s i

/-- Sequences with query token `q`, maximum `m`, and every other token below `b`. -/
def caseSet (K : ℕ) (q m : Fin V) (b : ℕ) : Finset (Fin (K + 1) → Fin V) :=
  Finset.univ.filter fun s => query s = q ∧ seqMax s = m ∧ ∀ i, s i = m ∨ (s i : ℕ) < b

theorem card_caseSet (q m : Fin V) (b : ℕ) (hb : b ≤ (m : ℕ)) :
    (caseSet K q m b).card =
      if q = m then (b + 1) ^ K else if (q : ℕ) < b then (b + 1) ^ K - b ^ K else 0 := by
  classical
  have hbV : b < V := lt_of_le_of_lt hb m.isLt
  set A0 : Finset (Fin V) := Finset.Iio ⟨b, hbV⟩ with hA0
  have hA0mem : ∀ t : Fin V, t ∈ A0 ↔ (t : ℕ) < b := by
    intro t
    simp [hA0, Fin.lt_def]
  have hA0card : A0.card = b := by rw [hA0, Fin.card_Iio]
  have hmA0 : m ∉ A0 := by rw [hA0mem]; omega
  set A : Finset (Fin V) := insert m A0 with hA
  have hAcard : A.card = b + 1 := by rw [hA, Finset.card_insert_of_notMem hmA0, hA0card]
  have hAmem : ∀ t : Fin V, t ∈ A ↔ t = m ∨ (t : ℕ) < b := by
    intro t
    rw [hA, Finset.mem_insert, hA0mem]
  let T1 : Fin (K + 1) → Finset (Fin V) := fun i => if i = Fin.last K then {q} else A
  let T0 : Fin (K + 1) → Finset (Fin V) := fun i => if i = Fin.last K then {q} else A0
  have hT1card : (Fintype.piFinset T1).card = (b + 1) ^ K := by
    rw [Fintype.card_piFinset, Fin.prod_univ_castSucc]
    simp [T1, hAcard, (Fin.castSucc_lt_last _).ne]
  have hT0card : (Fintype.piFinset T0).card = b ^ K := by
    rw [Fintype.card_piFinset, Fin.prod_univ_castSucc]
    simp [T0, hA0card, (Fin.castSucc_lt_last _).ne]
  have hsub : Fintype.piFinset T0 ⊆ Fintype.piFinset T1 := by
    apply Fintype.piFinset_subset
    intro i
    simp only [T0, T1]
    split_ifs
    · exact subset_rfl
    · rw [hA]; exact Finset.subset_insert _ _
  by_cases hqm : q = m
  · rw [if_pos hqm, ← hT1card]
    congr 1
    ext s
    simp only [caseSet, Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
    constructor
    · rintro ⟨h1, _, h3⟩ i
      simp only [T1]
      split_ifs with hi
      · rw [hi, Finset.mem_singleton]; exact h1
      · exact (hAmem _).mpr (h3 i)
    · intro h
      have hlast : query s = q := by
        have h' := h (Fin.last K)
        simp [T1] at h'
        exact h'
      have hall : ∀ i, s i = m ∨ (s i : ℕ) < b := by
        intro i
        by_cases hi : i = Fin.last K
        · left; rw [hi]; exact hlast.trans hqm
        · have h' := h i
          simp only [T1, if_neg hi] at h'
          exact (hAmem _).mp h'
      refine ⟨hlast, ?_, hall⟩
      apply seqMax_eq_of
      · exact ⟨Fin.last K, hlast.trans hqm⟩
      · intro i
        rcases hall i with e | e
        · exact e.le
        · exact Fin.le_def.mpr (by omega)
  · rw [if_neg hqm]
    by_cases hqb : (q : ℕ) < b
    · rw [if_pos hqb, ← hT1card, ← hT0card, ← Finset.card_sdiff_of_subset hsub]
      congr 1
      ext s
      simp only [caseSet, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_sdiff,
        Fintype.mem_piFinset]
      constructor
      · rintro ⟨h1, h2, h3⟩
        refine ⟨?_, ?_⟩
        · intro i
          simp only [T1]
          split_ifs with hi
          · rw [hi, Finset.mem_singleton]; exact h1
          · exact (hAmem _).mpr (h3 i)
        · intro h0
          obtain ⟨i, hi⟩ := exists_eq_seqMax s
          rw [h2] at hi
          have h' := h0 i
          simp only [T0] at h'
          split_ifs at h' with hl
          · rw [Finset.mem_singleton, hi] at h'; exact hqm h'.symm
          · rw [hi] at h'; exact hmA0 h'
      · rintro ⟨h1, h0⟩
        have hlast : query s = q := by
          have h' := h1 (Fin.last K)
          simp [T1] at h'
          exact h'
        have hall : ∀ i, s i = m ∨ (s i : ℕ) < b := by
          intro i
          by_cases hi : i = Fin.last K
          · right
            rw [hi]
            have e : s (Fin.last K) = q := hlast
            rw [e]; exact hqb
          · have h' := h1 i
            simp only [T1, if_neg hi] at h'
            exact (hAmem _).mp h'
        refine ⟨hlast, ?_, hall⟩
        apply seqMax_eq_of
        · by_contra hne
          apply h0
          intro i
          simp only [T0]
          split_ifs with hi
          · rw [hi, Finset.mem_singleton]; exact hlast
          · rcases hall i with e | e
            · exact absurd ⟨i, e⟩ hne
            · exact (hA0mem _).mpr e
        · intro i
          rcases hall i with e | e
          · exact e.le
          · exact Fin.le_def.mpr (by omega)
    · rw [if_neg hqb, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
      intro s hs
      simp only [caseSet, Finset.mem_filter, Finset.mem_univ, true_and] at hs
      obtain ⟨h1, _, h3⟩ := hs
      have e1 : s (Fin.last K) = q := h1
      rcases h3 (Fin.last K) with e | e
      · exact hqm (e1.symm.trans e)
      · rw [e1] at e; exact hqb e

/-- A sequence is covered by the table `bt`. -/
def Covered (bt : ℕ → ℕ → ℕ) (s : Fin (K + 1) → Fin V) : Prop :=
  0 < bt (query s) (seqMax s) ∧
    ∀ i, s i = seqMax s ∨ (s i : ℕ) < bt (query s) (seqMax s) - 1

instance (bt : ℕ → ℕ → ℕ) : DecidablePred (Covered (V := V) (K := K) bt) := fun s => by
  unfold Covered; infer_instance

/-- Number of covered sequences with query token `q` and maximum `m`, for the table value `c`. -/
def cnt (K q m c : ℕ) : ℕ :=
  if c = 0 then 0 else if q = m then c ^ K else if q < c - 1 then c ^ K - (c - 1) ^ K else 0

theorem card_covered (bt : ℕ → ℕ → ℕ) (hbt : ∀ q m : Fin V, bt q m ≤ m + 1) :
    (Finset.univ.filter (Covered (V := V) (K := K) bt)).card =
      ∑ q : Fin V, ∑ m : Fin V, cnt K q m (bt q m) := by
  classical
  rw [Finset.card_eq_sum_card_fiberwise (f := fun s => (query s, seqMax s)) (t := Finset.univ)
    (fun _ _ => Finset.mem_univ _), Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun m _ => ?_
  unfold cnt
  by_cases hc : bt q m = 0
  · rw [if_pos hc, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
    intro s hs
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Prod.mk.injEq, Covered] at hs
    obtain ⟨⟨h1, _⟩, hq, hm⟩ := hs
    rw [hq, hm, hc] at h1
    exact lt_irrefl _ h1
  · rw [if_neg hc]
    have hb : bt q m - 1 ≤ (m : ℕ) := by have := hbt q m; omega
    have key : (Finset.univ.filter (Covered (V := V) (K := K) bt)).filter
        (fun s => (query s, seqMax s) = (q, m)) = caseSet K q m (bt q m - 1) := by
      ext s
      simp only [caseSet, Covered, Finset.mem_filter, Finset.mem_univ, true_and, Prod.mk.injEq]
      constructor
      · rintro ⟨⟨_, h2⟩, hq, hm⟩
        refine ⟨hq, hm, fun i => ?_⟩
        have h' := h2 i
        rw [hq, hm] at h'
        exact h'
      · rintro ⟨hq, hm, h3⟩
        refine ⟨⟨?_, fun i => ?_⟩, hq, hm⟩
        · rw [hq, hm]; omega
        · rw [hq, hm]; exact h3 i
    rw [key, card_caseSet q m _ hb]
    have e : bt q m - 1 + 1 = bt q m := by omega
    rw [e]
    by_cases hqm : q = m
    · rw [if_pos hqm, if_pos (congrArg Fin.val hqm)]
    · rw [if_neg hqm, if_neg (fun h => hqm (Fin.ext h))]

/-- `f 0 + ... + f (n - 1)`, in a form the kernel evaluates. -/
def sumNat : ℕ → (ℕ → ℕ) → ℕ
  | 0, _ => 0
  | n + 1, f => sumNat n f + f n

theorem sumNat_eq (n : ℕ) (f : ℕ → ℕ) : sumNat n f = ∑ i : Fin n, f i := by
  rw [← Finset.sum_range]
  induction n with
  | zero => simp [sumNat]
  | succ n ih => rw [sumNat, Finset.sum_range_succ, ih]

end NNProofs.MaxOfK
