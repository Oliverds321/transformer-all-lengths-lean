/-!
# Kernel speed benchmark

How fast does the Lean kernel evaluate the kind of arithmetic the accuracy checker
needs: a large natural number (an exponential bound, about 512 bits) times a
48-bit value, followed by a running minimum and a comparison?

`run n s acc` performs `n` such steps on pseudo-random numbers produced by a
linear congruential generator, so that no large literals appear in the source.
The `match` on the comparison forces the kernel to evaluate each step before the
recursive call, which keeps the term from growing.
-/
namespace NNProofs.KernelBench

def step (s : Nat) : Nat :=
  (s * 6364136223846793005 + 1442695040888963407) % 2 ^ 512

def run : Nat → Nat → Nat → Nat
  | 0, _, acc => acc
  | n + 1, s, acc =>
    match Nat.ble (step s * (step s % 2 ^ 48)) acc with
    | true => run n (step s) (step s * (step s % 2 ^ 48))
    | false => run n (step s) acc

set_option maxRecDepth 100000 in
theorem bench_10k : run 10000 1 (2 ^ 600) < 2 ^ 600 := by decide +kernel

end NNProofs.KernelBench
