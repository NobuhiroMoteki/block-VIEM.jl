#!/bin/bash
# Driver: re-run the 4 SIGSEGV'd RHS-scaling HDF5s with -t 8 (FFTW thread fix).
set -u
cd /home/moteki/Julia/block-VIEM.jl

LOG=viem_results/paper/_run_rhs_retry.log
: > "$LOG"

H5S=(
  viem_results/paper/sphere_n20.hdf5
  viem_results/paper/doublet_n20.hdf5
  viem_results/paper/sphere_Au.hdf5
  viem_results/paper/doublet_Au.hdf5
)

SECONDS=0
for h5 in "${H5S[@]}"; do
  echo "====================================================================" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  RHS-RETRY (-t 8)  $h5  (elapsed $SECONDS s)" | tee -a "$LOG"
  echo "====================================================================" | tee -a "$LOG"
  julia --project=. -t 8 viem_results/paper/run_rhs_scaling.jl "$h5" >>"$LOG" 2>&1
  rc=$?
  echo "=== $(date '+%H:%M:%S')  RHS-RETRY  $h5  rc=$rc  (elapsed $SECONDS s)" | tee -a "$LOG"
done
echo "=== ALL RHS-RETRY DONE  total = $SECONDS s" | tee -a "$LOG"
