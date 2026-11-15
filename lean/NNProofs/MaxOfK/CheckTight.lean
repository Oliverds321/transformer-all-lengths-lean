import NNProofs.MaxOfK.CheckPF

/-!
# Exact thresholds for a model without positional embeddings

A threshold certificate pairs a region certificate with a failure certificate for
the same query and maximum `m` and the same gap `g`: the model is correct with at
most `N` other tokens, all at most `m - g`, and wrong on `N + 1` copies of `m - g`.
Pure integer computations.
-/

namespace NNProofs.MaxOfK

/-- One row of a threshold table, for the query and maximum `m` and the gap `g`.
`nt m g = 0`: no claim. `nt m g = N + 1`: the region certificate for other tokens up to
`m - g` holds with `N` of them, and `N + 1` copies of `m - g` outweigh the query token
`m` in favour of the answer `jt m g`. -/
def tightRow (M : IntPF) (nt jt : Nat → Nat → Nat) (m g : Nat) : Bool :=
  nt m g == 0 ||
    (decide (1 ≤ g) && (decide (g ≤ m) && (decide (jt m g < M.V) && (!(jt m g == m) &&
      (M.region m m (m - g + 1) (nt m g - 1) &&
        (decide (M.wHi m m (jt m g) (m - g) ≤ 0) &&
          decide (((nt m g : Nat) : Int) * M.wHi m m (jt m g) (m - g) +
            M.wHi m m (jt m g) m ≤ 0)))))))

end NNProofs.MaxOfK
