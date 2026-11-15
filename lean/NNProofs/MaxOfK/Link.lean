import Mathlib
import NNProofs.MaxOfK.Sound
import NNProofs.MaxOfK.Transformer
import NNProofs.MaxOfK.Weights

/-!
# From integer weights to the integer tables

`Linked W M ...` states that the tables of the integer model `M` are the matrix
products of the integer weights `W` (through intermediate products, each of which
is checked by the kernel). Under it, the table-form model `M.tables` is exactly
the table form of the real transformer with weights `W / S`, so every statement
about `M.tables` is a statement about that transformer.
-/

namespace NNProofs.MaxOfK

open Finset

theorem Tables.ext' {V K : ℕ} {T T' : Tables V K} (h1 : T.score = T'.score)
    (h2 : T.value = T'.value) (h3 : T.direct = T'.direct) : T = T' := by
  cases T; cases T'; simp_all

theorem cast_sumTo (n : ℕ) (f : ℕ → ℤ) :
    ((sumTo n f : ℤ) : ℝ) = ∑ a : Fin n, ((f a : ℤ) : ℝ) := by
  rw [sumTo_eq, Finset.sum_range]
  push_cast
  rfl

theorem chunk_all {c n : ℕ} {p : ℕ → Bool} (hc : 0 < c)
    (h : ∀ k, k < n → chunkOK c p k = true) : ∀ r, r < n * c → p r = true := by
  intro r hr
  have hk : r / c < n := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hr)
  have h1 := (allTo_iff _ _).mp (h (r / c) hk) (r % c) (Nat.mod_lt _ hc)
  have e : r / c * c + r % c = r := by rw [Nat.mul_comm]; exact Nat.div_add_mod r c
  rw [e] at h1
  exact h1

theorem rowMulOK_spec {d cols : ℕ} {A B C : ℕ → ℕ → ℤ} {r : ℕ} (h : rowMulOK d cols A B C r = true)
    (c : ℕ) (hc : c < cols) : C r c = sumTo d fun a => A r a * B a c :=
  of_decide_eq_true ((allTo_iff _ _).mp h c hc)

theorem rowMulScaleOK_spec {d cols : ℕ} {k : ℤ} {A B C : ℕ → ℕ → ℤ} {r : ℕ}
    (h : rowMulScaleOK d cols k A B C r = true) (c : ℕ) (hc : c < cols) :
    C r c = (sumTo d fun a => A r a * B a c) * k :=
  of_decide_eq_true ((allTo_iff _ _).mp h c hc)

/-- The real transformer with weights `W / S`. -/
noncomputable def IntWeights.toTransformer (W : IntWeights) (V K : ℕ) : Transformer V K W.d where
  E t a := (W.E t a : ℝ) / W.S
  P i a := (W.P i a : ℝ) / W.S
  WQ c a := (W.WQ c a : ℝ) / W.S
  WK c a := (W.WK c a : ℝ) / W.S
  WV c a := (W.WV c a : ℝ) / W.S
  WO c a := (W.WO c a : ℝ) / W.S
  WU a j := (W.WU a j : ℝ) / W.S

/-- The tables of `M` are the products of the weights `W`. -/
structure Linked (W : IntWeights) (M : IntModel) (XQ EK PK EV PV EVO PVO : ℕ → ℕ → ℤ) : Prop where
  S_pos : 0 < W.S
  D_eq : M.D = W.S ^ 4
  xq : ∀ q a, q < M.V → a < W.d → XQ q a = sumTo W.d fun c => (W.E q c + W.P M.K c) * W.WQ c a
  ek : ∀ t a, t < M.V → a < W.d → EK t a = sumTo W.d fun c => W.E t c * W.WK c a
  pk : ∀ i a, i ≤ M.K → a < W.d → PK i a = sumTo W.d fun c => W.P i c * W.WK c a
  eqke : ∀ q t, q < M.V → t < M.V → M.eqke q t = sumTo W.d fun a => XQ q a * EK t a
  eqkp : ∀ q i, q < M.V → i ≤ M.K → M.eqkp q i = sumTo W.d fun a => XQ q a * PK i a
  ev : ∀ t e, t < M.V → e < W.d → EV t e = sumTo W.d fun c => W.E t c * W.WV c e
  pv : ∀ i e, i ≤ M.K → e < W.d → PV i e = sumTo W.d fun c => W.P i c * W.WV c e
  evo : ∀ t a, t < M.V → a < W.d → EVO t a = sumTo W.d fun e => EV t e * W.WO e a
  pvo : ∀ i a, i ≤ M.K → a < W.d → PVO i a = sumTo W.d fun e => PV i e * W.WO e a
  evou : ∀ t j, t < M.V → j < M.V → M.evou t j = sumTo W.d fun a => EVO t a * W.WU a j
  pvou : ∀ i j, i ≤ M.K → j < M.V → M.pvou i j = sumTo W.d fun a => PVO i a * W.WU a j
  dir : ∀ q j, q < M.V → j < M.V →
    M.dir q j = (sumTo W.d fun a => (W.E q a + W.P M.K a) * W.WU a j) * ((W.S : ℤ) * (W.S : ℤ))

theorem sum_div_div {d : ℕ} (S : ℝ) (hS : S ≠ 0) (u v : Fin d → ℝ) :
    ∑ c, (u c / S) * (v c / S) = (∑ c, u c * v c) / S ^ 2 := by
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun c _ => ?_
  field_simp

namespace Linked

variable {W : IntWeights} {M : IntModel} {XQ EK PK EV PV EVO PVO : ℕ → ℕ → ℤ}

theorem tables_eq (h : Linked W M XQ EK PK EV PV EVO PVO) :
    (W.toTransformer M.V M.K).toTables = M.tables := by
  have hS : (W.S : ℝ) ≠ 0 := by exact_mod_cast h.S_pos.ne'
  have hD : (M.D : ℝ) = (W.S : ℝ) ^ 4 := by exact_mod_cast h.D_eq
  -- the three kinds of inner sums, for a token and for a position
  have hsumE : ∀ (Wm : ℕ → ℕ → ℤ) (t : ℕ) (a : ℕ),
      ∑ c : Fin W.d, ((W.E t c : ℝ) / W.S) * ((Wm c a : ℝ) / W.S) =
        ((sumTo W.d fun c => W.E t c * Wm c a : ℤ) : ℝ) / (W.S : ℝ) ^ 2 := by
    intro Wm t a
    rw [sum_div_div _ hS, cast_sumTo]
    push_cast
    rfl
  have hsumP : ∀ (Wm : ℕ → ℕ → ℤ) (i : ℕ) (a : ℕ),
      ∑ c : Fin W.d, ((W.P i c : ℝ) / W.S) * ((Wm c a : ℝ) / W.S) =
        ((sumTo W.d fun c => W.P i c * Wm c a : ℤ) : ℝ) / (W.S : ℝ) ^ 2 := by
    intro Wm i a
    rw [sum_div_div _ hS, cast_sumTo]
    push_cast
    rfl
  have hsumX : ∀ (Wm : ℕ → ℕ → ℤ) (q : ℕ) (a : ℕ),
      ∑ c : Fin W.d, ((W.E q c : ℝ) / W.S + (W.P M.K c : ℝ) / W.S) * ((Wm c a : ℝ) / W.S) =
        ((sumTo W.d fun c => (W.E q c + W.P M.K c) * Wm c a : ℤ) : ℝ) / (W.S : ℝ) ^ 2 := by
    intro Wm q a
    have e : ∀ c : Fin W.d, ((W.E q c : ℝ) / W.S + (W.P M.K c : ℝ) / W.S) * ((Wm c a : ℝ) / W.S) =
        (((W.E q c : ℝ) + (W.P M.K c : ℝ)) / W.S) * ((Wm c a : ℝ) / W.S) := by
      intro c; rw [add_div]
    rw [Finset.sum_congr rfl fun c _ => e c, sum_div_div _ hS, cast_sumTo]
    push_cast
    rfl
  have hsplit : ∀ (Wm : ℕ → ℕ → ℤ) (t i a : ℕ),
      ∑ c : Fin W.d, ((W.E t c : ℝ) / W.S + (W.P i c : ℝ) / W.S) * ((Wm c a : ℝ) / W.S) =
        ((sumTo W.d fun c => W.E t c * Wm c a : ℤ) : ℝ) / (W.S : ℝ) ^ 2 +
          ((sumTo W.d fun c => W.P i c * Wm c a : ℤ) : ℝ) / (W.S : ℝ) ^ 2 := by
    intro Wm t i a
    rw [← hsumE, ← hsumP, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun c _ => ?_
    ring
  apply Tables.ext'
  · -- scores
    funext q t i
    show Transformer.scoreOf _ _ _ = ((M.eqke q t + M.eqkp q i : ℤ) : ℝ) / M.D
    unfold Transformer.scoreOf Transformer.resid IntWeights.toTransformer
    simp only [Fin.val_last]
    have hterm : ∀ a : Fin W.d,
        (∑ c : Fin W.d, ((W.E q c : ℝ) / W.S + (W.P M.K c : ℝ) / W.S) * ((W.WQ c a : ℝ) / W.S)) *
          (∑ c : Fin W.d, ((W.E t c : ℝ) / W.S + (W.P i c : ℝ) / W.S) * ((W.WK c a : ℝ) / W.S)) =
        (((XQ q a * EK t a : ℤ) : ℝ) + ((XQ q a * PK i a : ℤ) : ℝ)) / (W.S : ℝ) ^ 4 := by
      intro a
      rw [hsumX, hsplit, ← h.xq q a q.isLt a.isLt, ← h.ek t a t.isLt a.isLt,
        ← h.pk i a (Nat.lt_succ_iff.mp i.isLt) a.isLt]
      push_cast
      field_simp
      try ring
    rw [Finset.sum_congr rfl fun a _ => hterm a, ← Finset.sum_div, Finset.sum_add_distrib,
      h.eqke q t q.isLt t.isLt, h.eqkp q i q.isLt (Nat.lt_succ_iff.mp i.isLt), hD,
      Int.cast_add, cast_sumTo, cast_sumTo]
  · -- values
    funext t i j
    show Transformer.ovLogit _ _ _ = ((M.evou t j + M.pvou i j : ℤ) : ℝ) / M.D
    unfold Transformer.ovLogit Transformer.resid IntWeights.toTransformer
    simp only
    have hinner : ∀ a : Fin W.d,
        (∑ e : Fin W.d, (∑ c : Fin W.d, ((W.E t c : ℝ) / W.S + (W.P i c : ℝ) / W.S) *
            ((W.WV c e : ℝ) / W.S)) * ((W.WO e a : ℝ) / W.S)) =
          (((EVO t a : ℤ) : ℝ) + ((PVO i a : ℤ) : ℝ)) / (W.S : ℝ) ^ 3 := by
      intro a
      have he : ∀ e : Fin W.d,
          (∑ c : Fin W.d, ((W.E t c : ℝ) / W.S + (W.P i c : ℝ) / W.S) * ((W.WV c e : ℝ) / W.S)) *
              ((W.WO e a : ℝ) / W.S) =
            (((EV t e * W.WO e a : ℤ) : ℝ) + ((PV i e * W.WO e a : ℤ) : ℝ)) / (W.S : ℝ) ^ 3 := by
        intro e
        rw [hsplit, ← h.ev t e t.isLt e.isLt, ← h.pv i e (Nat.lt_succ_iff.mp i.isLt) e.isLt]
        push_cast
        field_simp
        try ring
      rw [Finset.sum_congr rfl fun e _ => he e, ← Finset.sum_div, Finset.sum_add_distrib,
        h.evo t a t.isLt a.isLt, h.pvo i a (Nat.lt_succ_iff.mp i.isLt) a.isLt,
        cast_sumTo, cast_sumTo]
    have hterm : ∀ a : Fin W.d,
        (∑ e : Fin W.d, (∑ c : Fin W.d, ((W.E t c : ℝ) / W.S + (W.P i c : ℝ) / W.S) *
            ((W.WV c e : ℝ) / W.S)) * ((W.WO e a : ℝ) / W.S)) * ((W.WU a j : ℝ) / W.S) =
          (((EVO t a * W.WU a j : ℤ) : ℝ) + ((PVO i a * W.WU a j : ℤ) : ℝ)) / (W.S : ℝ) ^ 4 := by
      intro a
      rw [hinner]
      push_cast
      field_simp
      try ring
    rw [Finset.sum_congr rfl fun a _ => hterm a, ← Finset.sum_div, Finset.sum_add_distrib,
      h.evou t j t.isLt j.isLt, h.pvou i j (Nat.lt_succ_iff.mp i.isLt) j.isLt, hD,
      Int.cast_add, cast_sumTo, cast_sumTo]
  · -- direct path
    funext q j
    show ∑ a, Transformer.resid _ _ _ a * _ = ((M.dir q j : ℤ) : ℝ) / M.D
    unfold Transformer.resid IntWeights.toTransformer
    simp only [Fin.val_last]
    rw [hsumX, h.dir q j q.isLt j.isLt, hD]
    push_cast
    field_simp
    try ring

end Linked

end NNProofs.MaxOfK
