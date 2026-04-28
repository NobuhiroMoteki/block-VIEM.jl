#!/bin/bash
# Driver: RHS-scaling diagnostic for all 12 production HDF5s.
# Writes /target/rhs_scaling/gmres/ into each HDF5 (does not touch
# production /target/simulated_data/).  Order: light → heavy.
set -u
cd /home/moteki/Julia/block-VIEM.jl

LOG=viem_results/paper/_run_all_rhs_scaling.log
: > "$LOG"

H5S=(
  viem_results/paper/sphere_n15.hdf5
  viem_results/paper/doublet_n15.hdf5
  viem_results/paper/oblate_n15.hdf5
  viem_results/paper/sphere_n20.hdf5
  viem_results/paper/doublet_n20.hdf5
  viem_results/paper/oblate_n20.hdf5
  viem_results/paper/sphere_Au.hdf5
  viem_results/paper/doublet_Au.hdf5
  viem_results/paper/oblate_Au.hdf5
  viem_results/paper/gre_n15.hdf5
  viem_results/paper/gre_n20.hdf5
  viem_results/paper/gre_Au.hdf5
)

SECONDS=0
for h5 in "${H5S[@]}"; do
  echo "====================================================================" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  RHS  $h5   (elapsed $SECONDS s)" | tee -a "$LOG"
  echo "====================================================================" | tee -a "$LOG"
  julia --project=. -t auto viem_results/paper/run_rhs_scaling.jl "$h5" >>"$LOG" 2>&1
  rc=$?
  echo "=== $(date '+%H:%M:%S')  RHS  $h5  rc=$rc  (elapsed $SECONDS s)" | tee -a "$LOG"
done
echo "=== ALL RHS DONE  total = $SECONDS s" | tee -a "$LOG"
