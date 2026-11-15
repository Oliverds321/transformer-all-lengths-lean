#!/bin/bash
# Build the generated kernel-check modules of a position-free model in bounded batches.
# usage: ./build_pf.sh PF10 [reg-per-batch=8] [exp-per-batch=4]
NAME=$1; RB=${2:-8}; EB=${3:-4}
cd "$(dirname "$0")"
start=$(date +%s)
for kind in Reg Exp; do
  B=$RB; [ $kind = Exp ] && B=$EB
  q=0
  while [ $q -lt 64 ]; do
    targets=""
    for i in $(seq $q $((q + B - 1))); do [ $i -lt 64 ] && targets="$targets NNProofs.MaxOfK.$NAME.$kind.Q$i"; done
    lake build $targets 2>&1 | grep -E "error|✖|Built NNProofs.MaxOfK.$NAME" | cut -c1-200
    q=$((q + B))
  done
  echo "$kind done at $(( $(date +%s) - start )) s"
done
lake build NNProofs.MaxOfK.$NAME.All 2>&1 | grep -E "error|✖|Built NNProofs.MaxOfK.$NAME.(All|WCheck)|completed" -A6 | cut -c1-300
echo "All done at $(( $(date +%s) - start )) s"
