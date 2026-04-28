#!/bin/bash
# Driver: re-run gre lc convergence with reduced factors (1.5, 1.0, 0.7)
# to avoid the FFTW spawn segfault at factor <= 0.5.  Uses -t 8 for safety.
set -u
cd /home/moteki/Julia/block-VIEM.jl

LOG=viem_results/paper/_run_gre_lc_retry.log
: > "$LOG"
SECONDS=0
export LC_FACTORS="1.5,1.0,0.7"
for m in n15 n20 Au; do
  echo "=== $(date '+%H:%M:%S')  GRE LC RETRY  gre × $m  factors=$LC_FACTORS  (elapsed $SECONDS s)" | tee -a "$LOG"
  julia --project=. -t 8 viem_results/paper/run_lc_convergence.jl gre "$m" >>"$LOG" 2>&1
  echo "=== $(date '+%H:%M:%S')  GRE LC RETRY  gre × $m  rc=$?  (elapsed $SECONDS s)" | tee -a "$LOG"
done
echo "=== ALL GRE LC RETRY DONE  total = $SECONDS s" | tee -a "$LOG"
