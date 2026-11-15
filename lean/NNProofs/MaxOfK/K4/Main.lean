import Mathlib
import NNProofs.MaxOfK.Sound
import NNProofs.MaxOfK.ExpCheck
import NNProofs.MaxOfK.Count
import NNProofs.MaxOfK.K4.All
import NNProofs.MaxOfK.K4.WAll

/-!
# Max-of-4: a kernel-checked accuracy bound for a trained transformer

`M` is the quantised model exported from the checkpoint (vocabulary 64, four
positions, weights = integers / 2^10), in table form; `M.tables` is its
real-valued semantics with exact softmax (`NNProofs.MaxOfK.Model`).

`accuracy`: on at least 16,777,215 of the 64^4 = 16,777,216 input sequences the
model's logit of the maximum token is strictly larger than every other logit.
-/

namespace NNProofs.MaxOfK.K4

open NNProofs.MaxOfK

set_option maxRecDepth 1000000

/-- The exponential bounds stored in `M` are valid. -/
theorem expOK : M.ExpOK :=
  M.expOK_of_check 40 8 (by decide) (by decide) fun q hq => exp_all q hq

/-- Every sequence covered by a certified case is answered with its maximum. -/
theorem certified_correct (s : Fin (M.K + 1) → Fin M.V) (m : Fin M.V) (b : ℕ)
    (hb : bTab (query s) m = b + 1) (hmax : ∃ i, s i = m)
    (hothers : ∀ i, s i = m ∨ (s i : ℕ) < b) : M.tables.Answers s m :=
  M.case_sound expOK (query s) m b (rows_all _ _ b (query s).isLt m.isLt hb) s rfl hmax hothers

/-- The certificate table never allows tokens above the maximum. -/
def btOK : Bool := allTo 64 fun q => allTo 64 fun m => decide (bTab q m ≤ m + 1)

theorem btOK_true : btOK = true := by decide +kernel

theorem bTab_le (q m : Fin M.V) : bTab q m ≤ m + 1 := by
  have h1 := (allTo_iff _ _).mp btOK_true q q.isLt
  have h2 := (allTo_iff _ _).mp h1 m m.isLt
  exact of_decide_eq_true h2

/-- The number of covered sequences, as a kernel-evaluated sum. -/
def total : ℕ := sumNat 64 fun q => sumNat 64 fun m => cnt 3 q m (bTab q m)

theorem total_eq : total = 16777215 := by decide +kernel

theorem card_covered_eq :
    (Finset.univ.filter (Covered (V := M.V) (K := M.K) bTab)).card = 16777215 := by
  rw [card_covered bTab bTab_le, ← total_eq]
  show (∑ q : Fin 64, ∑ m : Fin 64, cnt 3 q m (bTab q m)) = total
  unfold total
  rw [sumNat_eq]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [sumNat_eq]

/-- **Accuracy of the trained Max-of-4 transformer.** At least 16,777,215 of the
16,777,216 input sequences are answered with their maximum token, with a strict
margin over every other token. -/
theorem accuracy :
    16777215 ≤ Set.ncard {s : Fin (M.K + 1) → Fin M.V | M.tables.Answers s (seqMax s)} := by
  have hsub : (↑(Finset.univ.filter (Covered (V := M.V) (K := M.K) bTab)) :
      Set (Fin (M.K + 1) → Fin M.V)) ⊆ {s | M.tables.Answers s (seqMax s)} := by
    intro s hs
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hs
    obtain ⟨h1, h2⟩ := hs
    exact certified_correct s (seqMax s) (bTab (query s) (seqMax s) - 1) (by omega)
      (exists_eq_seqMax s) h2
  calc 16777215 = (Finset.univ.filter (Covered (V := M.V) (K := M.K) bTab)).card :=
        card_covered_eq.symm
    _ = Set.ncard (↑(Finset.univ.filter (Covered (V := M.V) (K := M.K) bTab)) :
          Set (Fin (M.K + 1) → Fin M.V)) := (Set.ncard_coe_finset _).symm
    _ ≤ _ := Set.ncard_le_ncard hsub (Set.toFinite _)

/-- The quantised transformer itself: one layer, one head, attention only, with the
weight matrices `W / S` (`NNProofs.MaxOfK.Transformer`). -/
noncomputable def net : Transformer M.V M.K W.d := W.toTransformer M.V M.K

/-- Its logits are those of the table form `M.tables`. -/
theorem net_logit (s : Fin (M.K + 1) → Fin M.V) (j : Fin M.V) :
    net.logit s j = M.tables.logit s j := by
  rw [net.logit_eq, show net.toTables = M.tables from linked.tables_eq]

/-- **Main theorem (K = 4).** The quantised transformer, defined from its weight
matrices with real-valued softmax, puts the maximum token strictly above every
other token on at least 16,777,215 of the 64^4 = 16,777,216 input sequences. -/
theorem accuracy_transformer :
    16777215 ≤ Set.ncard {s : Fin (M.K + 1) → Fin M.V |
      ∀ j, j ≠ seqMax s → net.logit s j < net.logit s (seqMax s)} := by
  have e : {s : Fin (M.K + 1) → Fin M.V |
      ∀ j, j ≠ seqMax s → net.logit s j < net.logit s (seqMax s)} =
      {s | M.tables.Answers s (seqMax s)} := by
    ext s
    simp only [Set.mem_setOf_eq, Tables.Answers, net_logit]
  rw [e]
  exact accuracy

end NNProofs.MaxOfK.K4
