#!/bin/bash
# Driver: runs lc-convergence for 9 (shape × material) combos (doublet not supported).
# Output: convergence_{shape}_{material}.hdf5 per CLAUDE.md §4.
set -u
cd /home/moteki/Julia/block-VIEM.jl

LOG=viem_results/paper/_run_all_lc_convergence.log
: > "$LOG"
SECONDS=0
for s in sphere oblate gre; do
  for m in n15 n20 Au; do
    echo "=== $(date '+%H:%M:%S')  LC  $s × $m  (elapsed $SECONDS s)" | tee -a "$LOG"
    julia --project=. -t auto viem_results/paper/run_lc_convergence.jl "$s" "$m" >>"$LOG" 2>&1
    echo "=== $(date '+%H:%M:%S')  LC  $s × $m  rc=$?  (elapsed $SECONDS s)" | tee -a "$LOG"
  done
done
echo "=== ALL LC CONVERGENCE DONE  total = $SECONDS s" | tee -a "$LOG"
