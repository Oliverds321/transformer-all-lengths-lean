#!/bin/bash
# Build the generated kernel-check modules of one model in bounded batches.
# usage: ./build_cases.sh K4 [cases-per-batch=8] [exp-per-batch=4]
# lake has no job limit, so the batches bound memory use: the Cases modules do not
# import Mathlib; the Exp modules do.
NAME=$1; CB=${2:-8}; EB=${3:-4}
cd "$(dirname "$0")"
start=$(date +%s)
for kind in Cases Exp; do
  B=$CB; [ $kind = Exp ] && B=$EB
  q=0
  while [ $q -lt 64 ]; do
    targets=""
    for i in $(seq $q $((q + B - 1))); do [ $i -lt 64 ] && targets="$targets NNProofs.MaxOfK.$NAME.$kind.Q$i"; done
    lake build $targets 2>&1 | grep -E "error|✖|Built NNProofs.MaxOfK.$NAME" | cut -c1-200
    q=$((q + B))
  done
  echo "$kind done at $(( $(date +%s) - start )) s"
done
