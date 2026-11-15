#!/bin/bash
# Kernel-check several position-free models one after another.
cd "$(dirname "$0")"
for name in "$@"; do
  s=$(date +%s)
  ./build_pf.sh $name 8 4 > build_$name.log 2>&1
  lake build NNProofs.MaxOfK.$name.Main 2>&1 | grep -E "error|✖|completed" | cut -c1-200 >> build_$name.log
  echo "$name finished in $(( $(date +%s) - s )) s; errors: $(grep -c -E 'error|✖' build_$name.log)"
done
