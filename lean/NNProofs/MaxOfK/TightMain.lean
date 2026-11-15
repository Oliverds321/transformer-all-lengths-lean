import Mathlib
import NNProofs.MaxOfK.PosFreeMain
import NNProofs.MaxOfK.CheckTight

/-!
# Exact thresholds: soundness

A row of a threshold table (`tightRow`) gives, for the query and maximum `m` and the
gap `g`, a number `N` such that

* `tight_correct`: the model answers `m` on every input, of any length, whose query
  and maximum are `m`, whose other tokens are at most `m - g`, and in which at most
  `N` positions other than the last hold a token different from `m`;
* `tight_fail`: for every `k > N` the model does not answer `m` on `k` copies of
  `m - g` followed by `m`.
-/

namespace NNProofs.MaxOfK

namespace IntPF

variable (M : IntPF)

theorem tightRow_spec (nt jt : ℕ → ℕ → ℕ) (m g : ℕ) (hrow : tightRow M nt jt m g = true)
    (hnt : nt m g ≠ 0) :
    1 ≤ g ∧ g ≤ m ∧ jt m g < M.V ∧ jt m g ≠ m ∧
      M.region m m (m - g + 1) (nt m g - 1) = true ∧
      M.wHi m m (jt m g) (m - g) ≤ 0 ∧
      ((nt m g : ℕ) : ℤ) * M.wHi m m (jt m g) (m - g) + M.wHi m m (jt m g) m ≤ 0 := by
  unfold tightRow at hrow
  have h1 : (nt m g == 0) = false := by simpa using hnt
  rw [h1, Bool.false_or] at hrow
  obtain ⟨a, hrow⟩ := Bool.and_eq_true_iff.mp hrow
  obtain ⟨b, hrow⟩ := Bool.and_eq_true_iff.mp hrow
  obtain ⟨c, hrow⟩ := Bool.and_eq_true_iff.mp hrow
  obtain ⟨d, hrow⟩ := Bool.and_eq_true_iff.mp hrow
  obtain ⟨e, hrow⟩ := Bool.and_eq_true_iff.mp hrow
  obtain ⟨f, k⟩ := Bool.and_eq_true_iff.mp hrow
  refine ⟨of_decide_eq_true a, of_decide_eq_true b, of_decide_eq_true c, ?_, e,
    of_decide_eq_true f, of_decide_eq_true k⟩
  simpa using d

/-- **Correct up to the threshold.** -/
theorem tight_correct (h : M.ExpOK) (nt jt : ℕ → ℕ → ℕ) (m g : ℕ)
    (hrow : tightRow M nt jt m g = true) (hnt : nt m g ≠ 0)
    (L : ℕ) (s : Fin (L + 1) → Fin M.V) (hq : (query s : ℕ) = m) (hmax : (seqMax s : ℕ) = m)
    (hgap : ∀ i, s i = seqMax s ∨ (s i : ℕ) + g ≤ m)
    (hN : (Finset.univ.filter fun i : Fin L => s (Fin.castSucc i) ≠ seqMax s).card ≤ nt m g - 1) :
    (M.tables L).Answers s (seqMax s) := by
  obtain ⟨h1, h2, -, -, hreg, -, -⟩ := M.tightRow_spec nt jt m g hrow hnt
  refine M.region_sound h (query s) (seqMax s) (m - g + 1) (nt m g - 1) ?_ L s rfl
    (exists_eq_seqMax s) ?_ hN
  · rw [hq, hmax]
    exact hreg
  · intro i
    rcases hgap i with e | e
    · left; exact e
    · right; omega

/-- **Wrong beyond the threshold.** -/
theorem tight_fail (h : M.ExpOK) (nt jt : ℕ → ℕ → ℕ) (m g : ℕ) (hm : m < M.V)
    (hrow : tightRow M nt jt m g = true) (hnt : nt m g ≠ 0) (k : ℕ) (hk : nt m g ≤ k) :
    ¬ (M.tables k).Answers (M.copies k ⟨m - g, Nat.lt_of_le_of_lt (Nat.sub_le m g) hm⟩ ⟨m, hm⟩)
      ⟨m, hm⟩ := by
  obtain ⟨-, -, hj, hjm, -, hA, hB⟩ := M.tightRow_spec nt jt m g hrow hnt
  refine M.fail_copies h ⟨m, hm⟩ ⟨m - g, Nat.lt_of_le_of_lt (Nat.sub_le m g) hm⟩ ⟨jt m g, hj⟩ ?_
    (nt m g) hA hB k hk
  intro e
  exact hjm (congrArg Fin.val e)

end IntPF

end NNProofs.MaxOfK
