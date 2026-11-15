import Mathlib
import NNProofs.MaxOfK.PosFree
import NNProofs.MaxOfK.Count
import NNProofs.MaxOfK.Link

/-!
# Consequences for a model without positional embeddings

Generic statements that the generated files instantiate for a trained model:

* `covered_sound`, `accuracy_bound`: a table of region certificates for at most
  `N` other tokens gives an accuracy lower bound at every length up to `N + 1`;
* `fail_copies`: enough copies of one token below the maximum defeat it, at every
  larger length;
* `LinkedPF.tables_eq`: the integer tables are those of the transformer with
  weights `W / S` and zero positional embedding, at every length.
-/

namespace NNProofs.MaxOfK

open Finset

theorem sumTo_congr {n : ℕ} {f g : ℕ → ℤ} (h : ∀ i, i < n → f i = g i) : sumTo n f = sumTo n g := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [sumTo, sumTo, ih fun i hi => h i (Nat.lt_succ_of_lt hi), h n (Nat.lt_succ_self n)]

theorem sumTo_zero (n : ℕ) : sumTo n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumTo, ih]; rfl

namespace IntPF

variable (M : IntPF)

/-- A table of region certificates: `bt q m = b + 1` means that `region q m b N` holds. -/
def RegionTable (bt : ℕ → ℕ → ℕ) (N : ℕ) : Prop :=
  ∀ q m b, q < M.V → m < M.V → bt q m = b + 1 → M.region q m b N = true

theorem covered_sound (h : M.ExpOK) (bt : ℕ → ℕ → ℕ) (N : ℕ) (hreg : M.RegionTable bt N)
    (L : ℕ) (hL : L ≤ N) (s : Fin (L + 1) → Fin M.V) (hcov : Covered bt s) :
    (M.tables L).Answers s (seqMax s) := by
  obtain ⟨h1, h2⟩ := hcov
  have hb : bt (query s) (seqMax s) = (bt (query s) (seqMax s) - 1) + 1 := by omega
  refine M.region_sound h (query s) (seqMax s) _ N
    (hreg _ _ _ (query s).isLt (seqMax s).isLt hb) L s rfl (exists_eq_seqMax s) h2 ?_
  calc (Finset.univ.filter fun i : Fin L => s (Fin.castSucc i) ≠ seqMax s).card
      ≤ (Finset.univ : Finset (Fin L)).card := Finset.card_filter_le _ _
    _ = L := by rw [Finset.card_univ, Fintype.card_fin]
    _ ≤ N := hL

/-- **Accuracy at length `L + 1`** from a region table for at most `N ≥ L` other tokens. -/
theorem accuracy_bound (h : M.ExpOK) (bt : ℕ → ℕ → ℕ) (N : ℕ) (hreg : M.RegionTable bt N)
    (hbt : ∀ q m : Fin M.V, bt q m ≤ m + 1) (L : ℕ) (hL : L ≤ N) :
    (∑ q : Fin M.V, ∑ m : Fin M.V, cnt L q m (bt q m)) ≤
      Set.ncard {s : Fin (L + 1) → Fin M.V | (M.tables L).Answers s (seqMax s)} := by
  have hsub : (↑(Finset.univ.filter (Covered (V := M.V) (K := L) bt)) :
      Set (Fin (L + 1) → Fin M.V)) ⊆ {s | (M.tables L).Answers s (seqMax s)} := by
    intro s hs
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hs
    exact M.covered_sound h bt N hreg L hL s hs
  calc (∑ q : Fin M.V, ∑ m : Fin M.V, cnt L q m (bt q m))
      = (Finset.univ.filter (Covered (V := M.V) (K := L) bt)).card := (card_covered bt hbt).symm
    _ = Set.ncard (↑(Finset.univ.filter (Covered (V := M.V) (K := L) bt)) :
          Set (Fin (L + 1) → Fin M.V)) := (Set.ncard_coe_finset _).symm
    _ ≤ _ := Set.ncard_le_ncard hsub (Set.toFinite _)

theorem regionTable_of_rows (bt : ℕ → ℕ → ℕ) (N : ℕ)
    (h : ∀ q m, q < M.V → m < M.V → regRow M bt N q m = true) : M.RegionTable bt N := by
  intro q m b hq hm hb
  have h1 := h q m hq hm
  unfold regRow at h1
  rw [hb] at h1
  simpa using h1

/-- The input consisting of `n` copies of the token `a` followed by the query token `m`. -/
def copies (n : ℕ) (a m : Fin M.V) : Fin (n + 1) → Fin M.V := Fin.snoc (fun _ => a) m

/-- **Failure at every sufficient length.** If one occurrence of `a` contributes a
non-positive upper bound against `j`, and `n0` occurrences outweigh the query
token `m`, then for every `n ≥ n0` the model does not answer `m` on `n` copies of
`a` followed by `m`. -/
theorem fail_copies (h : M.ExpOK) (m a j : Fin M.V) (hj : j ≠ m) (n0 : ℕ)
    (hA : M.wHi m m j a ≤ 0) (hB : (n0 : ℤ) * M.wHi m m j a + M.wHi m m j m ≤ 0)
    (n : ℕ) (hn : n0 ≤ n) : ¬ (M.tables n).Answers (M.copies n a m) m := by
  apply M.not_answers_of_upper h n (M.copies n a m) m j hj
  have hq : query (M.copies n a m) = m := by
    unfold query copies
    exact Fin.snoc_last _ _
  rw [hq, Fin.sum_univ_castSucc]
  have h1 : ∀ i : Fin n, M.copies n a m (Fin.castSucc i) = a := by
    intro i
    unfold copies
    exact Fin.snoc_castSucc _ _ _
  have h2 : M.copies n a m (Fin.last n) = m := by
    unfold copies
    exact Fin.snoc_last _ _
  rw [Finset.sum_congr rfl fun i _ => by rw [h1 i], h2, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  have h3 : (n : ℤ) * M.wHi m m j a ≤ (n0 : ℤ) * M.wHi m m j a :=
    mul_le_mul_of_nonpos_right (by exact_mod_cast hn) hA
  omega

/-- A failure-table row gives failures at every sufficient length. -/
theorem fail_of_row (h : M.ExpOK) (nf : ℕ → ℕ) (m : ℕ) (hm : m < M.V) (hm0 : 0 < m)
    (hrow : failRow M nf m = true) (hnf : nf m ≠ 0) (n : ℕ) (hn : nf m - 1 ≤ n) :
    ¬ (M.tables n).Answers (M.copies n ⟨m - 1, by omega⟩ ⟨m, hm⟩) ⟨m, hm⟩ := by
  unfold failRow at hrow
  have h1 : (nf m == 0) = false := by simpa using hnf
  rw [h1, Bool.false_or, Bool.and_eq_true] at hrow
  obtain ⟨hA, hB⟩ := hrow
  refine M.fail_copies h ⟨m, hm⟩ ⟨m - 1, by omega⟩ ⟨m - 1, by omega⟩ ?_ (nf m - 1)
    (of_decide_eq_true hA) (of_decide_eq_true hB) n hn
  intro e
  have := congrArg Fin.val e
  simp at this
  omega

/-- The fixed-length integer model with zero positional tables. -/
def toIntModel (L : ℕ) : IntModel where
  V := M.V
  K := L
  D := M.D
  F := M.F
  eqke := M.eqke
  eqkp := fun _ _ => 0
  evou := M.evou
  pvou := fun _ _ => 0
  dir := M.dir
  cTok := M.cTok
  cPos := fun _ => 0
  tokLo := M.tokLo
  tokHi := M.tokHi
  posLo := fun _ _ => 2 ^ M.F
  posHi := fun _ _ => 2 ^ M.F

theorem toIntModel_tables (L : ℕ) : (M.toIntModel L).tables = M.tables L := by
  apply Tables.ext'
  · funext q t i
    show (((M.eqke q t + 0 : ℤ) : ℝ) / M.D) = ((M.eqke q t : ℤ) : ℝ) / M.D
    rw [add_zero]
  · funext t i j
    show (((M.evou t j + 0 : ℤ) : ℝ) / M.D) = ((M.evou t j : ℤ) : ℝ) / M.D
    rw [add_zero]
  · rfl

end IntPF

/-- The tables of the position-free model `M` are the products of the weights `W`,
whose positional embedding is zero. -/
structure LinkedPF (W : IntWeights) (M : IntPF) (XQ EK EV EVO : ℕ → ℕ → ℤ) : Prop where
  S_pos : 0 < W.S
  D_eq : M.D = W.S ^ 4
  P_zero : ∀ i a, W.P i a = 0
  xq : ∀ q a, q < M.V → a < W.d → XQ q a = sumTo W.d fun c => W.E q c * W.WQ c a
  ek : ∀ t a, t < M.V → a < W.d → EK t a = sumTo W.d fun c => W.E t c * W.WK c a
  eqke : ∀ q t, q < M.V → t < M.V → M.eqke q t = sumTo W.d fun a => XQ q a * EK t a
  ev : ∀ t e, t < M.V → e < W.d → EV t e = sumTo W.d fun c => W.E t c * W.WV c e
  evo : ∀ t a, t < M.V → a < W.d → EVO t a = sumTo W.d fun e => EV t e * W.WO e a
  evou : ∀ t j, t < M.V → j < M.V → M.evou t j = sumTo W.d fun a => EVO t a * W.WU a j
  dir : ∀ q j, q < M.V → j < M.V →
    M.dir q j = (sumTo W.d fun a => W.E q a * W.WU a j) * ((W.S : ℤ) * (W.S : ℤ))

namespace LinkedPF

variable {W : IntWeights} {M : IntPF} {XQ EK EV EVO : ℕ → ℕ → ℤ}

theorem linked (h : LinkedPF W M XQ EK EV EVO) (L : ℕ) :
    Linked W (M.toIntModel L) XQ EK (fun _ _ => 0) EV (fun _ _ => 0) EVO (fun _ _ => 0) where
  S_pos := h.S_pos
  D_eq := h.D_eq
  xq q a hq ha := by
    rw [h.xq q a hq ha]
    exact sumTo_congr fun c _ => by rw [h.P_zero, add_zero]
  ek := h.ek
  pk i a _ _ := by
    rw [sumTo_congr (g := fun _ => 0) fun c _ => by rw [h.P_zero, zero_mul], sumTo_zero]
  eqke := h.eqke
  eqkp q i _ _ := by
    show (0 : ℤ) = _
    rw [sumTo_congr (g := fun _ => 0) fun a _ => by rw [mul_zero], sumTo_zero]
  ev := h.ev
  pv i e _ _ := by
    rw [sumTo_congr (g := fun _ => 0) fun c _ => by rw [h.P_zero, zero_mul], sumTo_zero]
  evo := h.evo
  pvo i a _ _ := by
    rw [sumTo_congr (g := fun _ => 0) fun e _ => by rw [zero_mul], sumTo_zero]
  evou := h.evou
  pvou i j _ _ := by
    show (0 : ℤ) = _
    rw [sumTo_congr (g := fun _ => 0) fun a _ => by rw [zero_mul], sumTo_zero]
  dir q j hq hj := by
    show M.dir q j = _
    rw [h.dir q j hq hj]
    congr 1
    exact sumTo_congr fun a _ => by rw [h.P_zero, add_zero]

/-- At every length, the table form of the transformer with weights `W / S` is `M.tables`. -/
theorem tables_eq (h : LinkedPF W M XQ EK EV EVO) (L : ℕ) :
    (W.toTransformer M.V L).toTables = M.tables L := by
  rw [← M.toIntModel_tables L]
  exact (h.linked L).tables_eq

end LinkedPF

end NNProofs.MaxOfK
