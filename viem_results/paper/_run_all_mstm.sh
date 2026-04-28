#!/bin/bash
# Driver: runs MSTM reference for the 3 doublet HDF5s (separate env).
# Output: mstm_doublet_{mat}.hdf5 (CLAUDE.md §6)
set -u
cd /home/moteki/Julia/block-VIEM.jl

LOG=viem_results/paper/_run_all_mstm.log
: > "$LOG"
SECONDS=0
for mat in n15 n20 Au; do
  in=viem_results/paper/doublet_${mat}.hdf5
  out=viem_results/paper/mstm_doublet_${mat}.hdf5
  echo "=== $(date '+%H:%M:%S')  MSTM  $in → $out   (elapsed $SECONDS s)" | tee -a "$LOG"
  julia --project=/home/moteki/Julia/MSTMforCAS.jl \
        viem_results/paper/run_mstm_reference.jl "$in" "$out" >>"$LOG" 2>&1
  echo "=== $(date '+%H:%M:%S')  MSTM  $mat  rc=$?  (elapsed $SECONDS s)" | tee -a "$LOG"
done
echo "=== ALL MSTM DONE  total = $SECONDS s" | tee -a "$LOG"
