#!/bin/bash
# Driver: sequentially runs all 12 production VIEM sweeps.
# Order: n15 (light) → n20 (mid) → Au (heavy), sphere/doublet/oblate/gre per material.
# Log: viem_results/paper/_run_all_production.log (overwritten on each invocation)
set -u
cd /home/moteki/Julia/block-VIEM.jl

LOG=viem_results/paper/_run_all_production.log
: > "$LOG"

H5S=(
  viem_results/paper/sphere_n15.hdf5
  viem_results/paper/doublet_n15.hdf5
  viem_results/paper/oblate_n15.hdf5
  viem_results/paper/gre_n15.hdf5
  viem_results/paper/sphere_n20.hdf5
  viem_results/paper/doublet_n20.hdf5
  viem_results/paper/oblate_n20.hdf5
  viem_results/paper/gre_n20.hdf5
  viem_results/paper/sphere_Au.hdf5
  viem_results/paper/doublet_Au.hdf5
  viem_results/paper/oblate_Au.hdf5
  viem_results/paper/gre_Au.hdf5
)

SECONDS=0
for h5 in "${H5S[@]}"; do
  echo "====================================================================" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  START  $h5   (elapsed $SECONDS s)"  | tee -a "$LOG"
  echo "====================================================================" | tee -a "$LOG"
  julia --project=. -t auto viem_results/run_viem.jl "$h5" >>"$LOG" 2>&1
  rc=$?
  echo "=== $(date '+%H:%M:%S')  DONE   $h5   rc=$rc  (elapsed $SECONDS s)" | tee -a "$LOG"
  if [ $rc -ne 0 ]; then
    echo "=== NON-ZERO EXIT — continuing to next HDF5" | tee -a "$LOG"
  fi
done

echo "====================================================================" | tee -a "$LOG"
echo "=== ALL 12 SWEEPS DONE  total elapsed = $SECONDS s" | tee -a "$LOG"
echo "====================================================================" | tee -a "$LOG"
