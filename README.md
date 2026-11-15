# Guaranteed behaviour of a small trained transformer at every input length: a machine-checked proof of concept

Lean 4 proofs, checked by the kernel with the standard axioms only, about what
small trained transformers compute. The networks are one-layer, one-head,
attention-only models trained on Max-of-K (output the largest input token;
vocabulary 64). The proofs work from the attention and value tables of the
network and never enumerate inputs.

The paper is in [`paper/manuscript.pdf`](paper/manuscript.pdf). This repository
(`transformer-all-lengths-lean`) is its supplement.

## What is proved

Everything is about the **quantised model**: each weight is an integer divided
by 2^10 (the query matrix is divided by sqrt(d_model) before rounding), with
exact real-valued softmax. The transformer is defined in Lean from its weight
matrices (`lean/NNProofs/MaxOfK/Transformer.lean`). "Answers m" means that the
logit of m is strictly above every other logit.

### Every input length (model without positional embeddings, trained at length 10)

Main file: [`lean/NNProofs/MaxOfK/PF10/Main.lean`](lean/NNProofs/MaxOfK/PF10/Main.lean).

| Theorem | Statement |
|---|---|
| `netAnswers_iff_counts` | On every input of every length, whether the model answers `m` is decided by linear inequalities in the token counts of the input. |
| `correct3`, `correct9`, `correct19`, `correct99`, `correct1000`, `correct1000000` | For every length: if the maximum is at least m0, every other token is at least g below it, and at most N positions other than the last hold another token, the model answers the maximum. (N, g, m0) = (3, 1, 15), (9, 3, 15), (19, 4, 16), (99, 6, 17), (1000, 8, 19), (1000000, 16, 23). |
| `failure`, `failCount`, `failMax`, `failZero`, `failRange`, `failure_21` | For 53 values of m (every m >= 15 among them), an explicit k0(m) <= 17: for every k >= k0(m), on k copies of m - 1 followed by m the model does not answer m. For example k0(21) = 5. |
| `threshold_correct`, `threshold_fail`, `threshold`, `tightCounts`, `threshold_21` | Exact thresholds. For 613 of the 640 pairs of a maximum m from 24 to 63 and a gap g from 1 to 16, an explicit N(m, g): the model answers m on every input, of any length, whose last token and maximum are m, with at most N other tokens, all at most m - g; and it does not answer m on more than N copies of m - g followed by m. For example, on k copies of 20 followed by 21 it answers 21 if and only if k <= 4. |
| `accuracy_length4`, `accuracy_length10`, `accuracy_length20`, `accuracy_length100` | Among all inputs of that length, at least a fraction 0.9972, 0.9705, 0.8841, 0.5203 is answered with its maximum (the theorems state the exact counts). |

The same statements, with their own constants, are proved for four more training
seeds in `PF10s1` to `PF10s4`:

| Seed | Accuracy bound at length 4 / 10 / 20 / 100 | N = 10^6: (g, m0) | Exact threshold: correct if and only if |
|---|---|---|---|
| 0 (`PF10`) | 0.9972 / 0.9705 / 0.8841 / 0.5203 | (16, 23) | k copies of 20 then 21: k <= 4 |
| 1 (`PF10s1`) | 0.9968 / 0.9638 / 0.9073 / 0.4902 | (24, 25) | k copies of 20 then 21: k <= 4 |
| 2 (`PF10s2`) | 0.9960 / 0.9682 / 0.9214 / 0.8299 | (20, 22) | k copies of 23 then 24: k <= 5 |
| 3 (`PF10s3`) | 0.9950 / 0.9660 / 0.9176 / 0.7324 | (20, 26) | k copies of 28 then 29: k <= 5 |
| 4 (`PF10s4`) | 0.9940 / 0.9680 / 0.9160 / 0.7804 | (20, 21) | k copies of 28 then 29: k <= 4 |

The threshold shown is for the maximum that fails with the fewest copies among
those covered by that seed's first correctness statement (m >= 15 to 18). In
every seed an exact threshold is proved for at least 39 of the 40 maxima from 24
to 63 at each gap from 1 to 6, and for at least 31 at each gap up to 16
(`tightCounts`).

### Fixed length (model with positional embeddings, trained at length 4)

`NNProofs.MaxOfK.K4.accuracy_transformer` in
[`lean/NNProofs/MaxOfK/K4/Main.lean`](lean/NNProofs/MaxOfK/K4/Main.lean): the
model answers the maximum on at least 16,777,215 of the 64^4 = 16,777,216
inputs.

### Not proved

The link between the floating-point checkpoints in `maxofk/checkpoints/` and the
integer weights in the Lean files. `maxofk/compare_quantised.py` compares the two
models on random inputs and on all inputs of length 4.

## Layout

| Path | Contents |
|---|---|
| `paper/` | manuscript (LaTeX, PDF), bibliography, figure and the script that computes it |
| `maxofk/` | model, training, certificate generators (`export_lean.py`, `export_pf.py`), experiments (`lengths.py`, `compare_quantised.py`), trained checkpoints |
| `lean/NNProofs/ExpBound.lean` | integer bounds for the exponential, proved from Mathlib's Taylor remainder bound |
| `lean/NNProofs/MaxOfK/Transformer.lean`, `Model.lean` | the transformer, its table form, and the separability theorem |
| `lean/NNProofs/MaxOfK/Check.lean`, `Sound.lean`, `Count.lean` | fixed length: integer checker, its soundness, the count of certified inputs |
| `lean/NNProofs/MaxOfK/CheckPF.lean`, `PosFree.lean`, `PosFreeMain.lean` | every length: certificates and their soundness |
| `lean/NNProofs/MaxOfK/CheckTight.lean`, `TightMain.lean` | exact thresholds: certificates and their soundness |
| `lean/NNProofs/MaxOfK/Weights.lean`, `Link.lean` | the integer tables are the matrix products of the integer weights |
| `lean/NNProofs/MaxOfK/K4/`, `PF10*/` | generated: packed weights and tables, kernel checks, main theorems |
| `lean/NNProofs/KernelBench.lean` | the kernel-speed measurement quoted in the paper |
| `lean/audit/` | `Axioms.lean` prints the axioms of every main theorem |

## Checking the proofs

Requirements: [elan](https://github.com/leanprover/elan) (Lean 4.35.0 is
selected by `lean/lean-toolchain`), about 10 GB of disk for Mathlib's prebuilt
cache, and a few hours of CPU time.

    cd lean
    lake exe cache get
    ./build_cases.sh K4
    ./build_seeds.sh PF10 PF10s1 PF10s2 PF10s3 PF10s4
    lake build
    lake env lean audit/Axioms.lean

The scripts build the kernel-check modules in bounded batches; run them before
the plain `lake build`, which would otherwise start as many processes as there
are cores. The case and region modules do not import Mathlib and are built
eight at a time (under 1 GB each). The exponential-bound modules import Mathlib
and are built four at a time: on Windows each of those processes commits about
6 GB of virtual memory, at times up to 9 GB, so allow about 40 GB of RAM plus
page file. The last command prints 129 lines; each must end in
`[propext, Classical.choice, Quot.sound]` or say that the theorem does not
depend on any axioms (28 facts about the failure and threshold tables). On a 24-core desktop the
kernel checks took 43 minutes for `K4` and 30 to 43 minutes for each `PF10*`,
under four hours in all.

The generated Lean files in this repository are the reference. To regenerate
them from the checkpoints:

    cd maxofk
    python export_lean.py checkpoints/maxofk_v64_k4_s0.pt K4
    python export_pf.py checkpoints/maxofk_v64_k10_s0_nopos.pt PF10
    python export_pf.py checkpoints/maxofk_v64_k10_s1_nopos.pt PF10s1     # and s2, s3, s4

Training (`python train.py --ctx 10 --steps 30000 --no-pos --seed 0`, under a
minute on a GPU) is not bit-reproducible across machines, which is why the
checkpoints are included. Python packages: `torch`, `numpy`, `matplotlib`.

## What has to be trusted

The Lean kernel; Mathlib's definitions of the real numbers and of the
exponential; and the definitions that the statements mention: the transformer
(`Transformer.lean`), its construction from integer weights (`IntWeights`,
`IntWeights.toTransformer` in `Weights.lean` and `Link.lean`), the stored
literals (the weights `W` in `*/WData.lean`; the model data `M` and the failure
table `nFail` in `*/Data.lean`; the threshold table `nTight` in `*/TightData.lean`) and the functions `unpack`, `unpackInt` and
`allTo` that read and test them (`Check.lean`), and the short definitions
`query` (`Model.lean`), `seqMax` (`Count.lean`), `IntPF.copies`
(`PosFreeMain.lean`), and `net`, `NetAnswers` and `others` (each
`PF10*/Main.lean`). The `Main.lean` files are generated by the certificate
generator, so an auditor should read their statements, not only check that they
compile. No `sorry`, no
`native_decide`. The Python programs
are not trusted for the theorems, which are about the integers written in the
Lean files.

## Licence and authorship

Code (Lean and Python) is under the Apache License 2.0 (`LICENSE`). The
manuscript and its figure are under CC BY 4.0 (`paper/LICENSE.md`).

Oliver D'Souza is the sole author of this work. The AI system Claude
(Anthropic) assisted with development, auditing, formalization, the
computational experiments and manuscript preparation; the author is responsible
for the mathematical claims.
