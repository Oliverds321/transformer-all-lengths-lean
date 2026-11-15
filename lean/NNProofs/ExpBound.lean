import Mathlib

/-!
# Certified integer bounds for the exponential

`expBounds g n p y D` returns natural numbers `(lo, hi)` with

    lo ≤ 2 ^ g * exp (-y / D) ≤ hi

whenever `0 < D`, `0 < n` and `y ≤ D * 2 ^ p`. The method: a Taylor polynomial of
degree `n - 1` at `z = -y / (D * 2 ^ p)`, which lies in `[-1, 0]`, with the
remainder bound `Real.exp_bound`, evaluated in exact rational arithmetic and
rounded outwards to `g` fractional bits; then `p` squarings with outward
rounding. Everything is computable, so the kernel can evaluate it.
-/

namespace NNProofs.ExpBound

open Finset

/-- Taylor polynomial of `exp` with `n` terms. -/
def taylor : ℕ → ℚ → ℚ
  | 0, _ => 0
  | n + 1, z => taylor n z + z ^ n / (n.factorial : ℚ)

theorem taylor_cast (n : ℕ) (z : ℚ) :
    ((taylor n z : ℚ) : ℝ) = ∑ m ∈ range n, (z : ℝ) ^ m / (m.factorial : ℝ) := by
  induction n with
  | zero => simp [taylor]
  | succ n ih =>
    rw [taylor, Finset.sum_range_succ, ← ih]
    push_cast
    ring

/-- Remainder bound of `Real.exp_bound`. -/
def err (n : ℕ) (z : ℚ) : ℚ :=
  |z| ^ n * (((n.succ : ℕ) : ℚ) / ((n.factorial : ℚ) * (n : ℚ)))

theorem err_cast (n : ℕ) (z : ℚ) :
    ((err n z : ℚ) : ℝ) = |(z : ℝ)| ^ n * ((n.succ : ℝ) / ((n.factorial : ℝ) * (n : ℝ))) := by
  unfold err
  push_cast
  ring

theorem taylor_bounds {z : ℚ} (hz : |(z : ℝ)| ≤ 1) {n : ℕ} (hn : 0 < n) :
    ((taylor n z - err n z : ℚ) : ℝ) ≤ Real.exp z ∧ Real.exp z ≤ ((taylor n z + err n z : ℚ) : ℝ) := by
  have h := Real.exp_bound hz hn
  rw [abs_le] at h
  push_cast
  rw [taylor_cast, err_cast]
  constructor <;> linarith [h.1, h.2]

/-- Lower and upper fixed-point squaring with outward rounding. -/
def sqLo (g lo : ℕ) : ℕ := lo * lo / 2 ^ g
def sqHi (g hi : ℕ) : ℕ := (hi * hi + (2 ^ g - 1)) / 2 ^ g

theorem sqLo_le {g lo : ℕ} {e : ℝ} (he : 0 ≤ e) (h : (lo : ℝ) ≤ 2 ^ g * e) :
    ((sqLo g lo : ℕ) : ℝ) ≤ 2 ^ g * (e * e) := by
  have hpos : (0 : ℝ) < 2 ^ g := by positivity
  have h1 : sqLo g lo * 2 ^ g ≤ lo * lo := Nat.div_mul_le_self _ _
  have h2 : ((sqLo g lo : ℕ) : ℝ) * 2 ^ g ≤ (lo : ℝ) * lo := by exact_mod_cast h1
  have h3 : (lo : ℝ) * lo ≤ (2 ^ g * e) * (2 ^ g * e) :=
    mul_le_mul h h (Nat.cast_nonneg _) (by positivity)
  have h4 : ((sqLo g lo : ℕ) : ℝ) * 2 ^ g ≤ (2 ^ g * (e * e)) * 2 ^ g := by nlinarith
  exact le_of_mul_le_mul_right h4 hpos

theorem ceil_div_mul_ge (a d : ℕ) (hd : 0 < d) : a ≤ (a + (d - 1)) / d * d := by
  have h := Nat.lt_mul_div_succ (a + (d - 1)) hd
  rw [Nat.mul_succ] at h
  have h2 : d * ((a + (d - 1)) / d) = (a + (d - 1)) / d * d := Nat.mul_comm _ _
  omega

theorem le_sqHi {g hi : ℕ} {e : ℝ} (he : 0 ≤ e) (h : 2 ^ g * e ≤ (hi : ℝ)) :
    2 ^ g * (e * e) ≤ ((sqHi g hi : ℕ) : ℝ) := by
  have hpos : (0 : ℝ) < 2 ^ g := by positivity
  have hd : 0 < 2 ^ g := Nat.two_pow_pos g
  have h1 : hi * hi ≤ sqHi g hi * 2 ^ g := ceil_div_mul_ge _ _ hd
  have h2 : (hi : ℝ) * hi ≤ ((sqHi g hi : ℕ) : ℝ) * 2 ^ g := by exact_mod_cast h1
  have h0 : (0 : ℝ) ≤ 2 ^ g * e := by positivity
  have h3 : (2 ^ g * e) * (2 ^ g * e) ≤ (hi : ℝ) * hi := mul_le_mul h h h0 (Nat.cast_nonneg _)
  have h4 : (2 ^ g * (e * e)) * 2 ^ g ≤ ((sqHi g hi : ℕ) : ℝ) * 2 ^ g := by nlinarith
  exact le_of_mul_le_mul_right h4 hpos

/-- `k` squarings. -/
def iter (g : ℕ) : ℕ → ℕ × ℕ → ℕ × ℕ
  | 0, x => x
  | k + 1, x => iter g k (sqLo g x.1, sqHi g x.2)

theorem iter_bounds (g : ℕ) : ∀ (k lo hi : ℕ) (e : ℝ), 0 ≤ e → (lo : ℝ) ≤ 2 ^ g * e →
    2 ^ g * e ≤ (hi : ℝ) →
    (((iter g k (lo, hi)).1 : ℕ) : ℝ) ≤ 2 ^ g * e ^ 2 ^ k ∧
      2 ^ g * e ^ 2 ^ k ≤ (((iter g k (lo, hi)).2 : ℕ) : ℝ)
  | 0, lo, hi, e, _, h1, h2 => by simpa [iter] using And.intro h1 h2
  | k + 1, lo, hi, e, he, h1, h2 => by
    have ih := iter_bounds g k (sqLo g lo) (sqHi g hi) (e * e) (mul_nonneg he he)
      (sqLo_le he h1) (le_sqHi he h2)
    have hpow : (e * e) ^ 2 ^ k = e ^ 2 ^ (k + 1) := by
      rw [← pow_two, ← pow_mul, pow_succ, Nat.mul_comm]
    rw [hpow] at ih
    simpa [iter] using ih

theorem toNat_cast_le {a : ℤ} {x : ℝ} (h : (a : ℝ) ≤ x) (hx : 0 ≤ x) : ((a.toNat : ℕ) : ℝ) ≤ x := by
  rcases le_or_gt 0 a with ha | ha
  · have : ((a.toNat : ℕ) : ℤ) = a := Int.toNat_of_nonneg ha
    have h' : ((a.toNat : ℕ) : ℝ) = (a : ℝ) := by exact_mod_cast this
    rw [h']; exact h
  · have : a.toNat = 0 := Int.toNat_eq_zero.mpr ha.le
    rw [this]; simpa using hx

theorem le_toNat_cast {a : ℤ} {x : ℝ} (h : x ≤ (a : ℝ)) : x ≤ ((a.toNat : ℕ) : ℝ) := by
  have h1 : a ≤ ((a.toNat : ℕ) : ℤ) := Int.self_le_toNat a
  have h2 : (a : ℝ) ≤ ((a.toNat : ℕ) : ℝ) := by exact_mod_cast h1
  exact h.trans h2

/-- Bounds `(lo, hi)` for `2 ^ g * exp (-y / D)`. -/
def expBounds (g n p y D : ℕ) : ℕ × ℕ :=
  iter g p
    (Int.toNat ⌊(2 : ℚ) ^ g * (taylor n (-(y : ℚ) / ((D : ℚ) * 2 ^ p)) - err n (-(y : ℚ) / ((D : ℚ) * 2 ^ p)))⌋,
     Int.toNat ⌈(2 : ℚ) ^ g * (taylor n (-(y : ℚ) / ((D : ℚ) * 2 ^ p)) + err n (-(y : ℚ) / ((D : ℚ) * 2 ^ p)))⌉)

theorem expBounds_spec (g n p y D : ℕ) (hD : 0 < D) (hn : 0 < n) (hy : y ≤ D * 2 ^ p) :
    (((expBounds g n p y D).1 : ℕ) : ℝ) ≤ 2 ^ g * Real.exp (-(y : ℝ) / D) ∧
      2 ^ g * Real.exp (-(y : ℝ) / D) ≤ (((expBounds g n p y D).2 : ℕ) : ℝ) := by
  set z : ℚ := -(y : ℚ) / ((D : ℚ) * 2 ^ p) with hz
  have hDp : (0 : ℝ) < (D : ℝ) * 2 ^ p := by positivity
  have hzR : (z : ℝ) = -(y : ℝ) / ((D : ℝ) * 2 ^ p) := by rw [hz]; push_cast; ring
  have hyR : (y : ℝ) ≤ (D : ℝ) * 2 ^ p := by exact_mod_cast hy
  have habs : |(z : ℝ)| ≤ 1 := by
    rw [hzR, abs_div, abs_neg, abs_of_nonneg (Nat.cast_nonneg y), abs_of_pos hDp]
    exact (div_le_one hDp).mpr hyR
  obtain ⟨hlo, hhi⟩ := taylor_bounds habs hn
  have hg : (0 : ℝ) < 2 ^ g := by positivity
  have he : 0 ≤ Real.exp (z : ℝ) := (Real.exp_pos _).le
  -- the starting interval
  have h1 : (((Int.toNat ⌊(2 : ℚ) ^ g * (taylor n z - err n z)⌋ : ℕ)) : ℝ) ≤ 2 ^ g * Real.exp (z : ℝ) := by
    apply toNat_cast_le _ (by positivity)
    have hf : ((⌊(2 : ℚ) ^ g * (taylor n z - err n z)⌋ : ℤ) : ℚ) ≤ (2 : ℚ) ^ g * (taylor n z - err n z) :=
      Int.floor_le _
    have hf' : ((⌊(2 : ℚ) ^ g * (taylor n z - err n z)⌋ : ℤ) : ℝ) ≤
        (((2 : ℚ) ^ g * (taylor n z - err n z) : ℚ) : ℝ) := by exact_mod_cast hf
    refine hf'.trans ?_
    push_cast
    have := mul_le_mul_of_nonneg_left hlo hg.le
    push_cast at this
    exact this
  have h2 : 2 ^ g * Real.exp (z : ℝ) ≤ (((Int.toNat ⌈(2 : ℚ) ^ g * (taylor n z + err n z)⌉ : ℕ)) : ℝ) := by
    apply le_toNat_cast
    have hc : (2 : ℚ) ^ g * (taylor n z + err n z) ≤ ((⌈(2 : ℚ) ^ g * (taylor n z + err n z)⌉ : ℤ) : ℚ) :=
      Int.le_ceil _
    have hc' : (((2 : ℚ) ^ g * (taylor n z + err n z) : ℚ) : ℝ) ≤
        ((⌈(2 : ℚ) ^ g * (taylor n z + err n z)⌉ : ℤ) : ℝ) := by exact_mod_cast hc
    refine le_trans ?_ hc'
    push_cast
    have := mul_le_mul_of_nonneg_left hhi hg.le
    push_cast at this
    exact this
  have hit := iter_bounds g p _ _ _ he h1 h2
  have hexp : Real.exp (z : ℝ) ^ 2 ^ p = Real.exp (-(y : ℝ) / D) := by
    rw [← Real.exp_nat_mul]
    congr 1
    rw [hzR]
    push_cast
    field_simp
  rw [hexp] at hit
  exact hit

end NNProofs.ExpBound
