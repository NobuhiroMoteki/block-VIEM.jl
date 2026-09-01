#!/usr/bin/env bash
# Launch the hematite and goethite sweeps together, unattended.
#
# -t auto is NOT optional. Without it Julia runs one thread and the solve is about
# six times slower; that is how a first cost estimate came out at 22 h on this
# machine (measured 2026-09-01, single-threaded by mistake).
#
# The two species run in parallel on purpose: the setup phase (basis, AIM
# projection, mass matrix) is single-threaded while the solve is not, so one
# process's setup overlaps the other's solve instead of leaving cores idle.
#
# Each has its own checkpoint, so either can be killed and restarted without
# losing completed rows. Do NOT edit ironoxide_sweep_h5.jl while this is running.
set -u
cd "$(dirname "$0")/../.."

N_DVE=${N_DVE:-8}
N_AR=${N_AR:-7}
N_RI=${N_RI:-3}
OUT_DIR=${OUT_DIR:-benchmarks/cas_v2}

for sp in hematite goethite; do
    out="$OUT_DIR/spheroid_sweep_viem_${sp}_liquid.h5"
    log="$OUT_DIR/${sp}_sweep.log"
    echo "launching $sp -> $out (log: $log)"
    nohup julia --project=. -t auto benchmarks/cas_v2/ironoxide_sweep_h5.jl \
        --species "$sp" --n-dve "$N_DVE" --n-ar "$N_AR" --n-ri "$N_RI" \
        --output "$out" --checkpoint "$out.ckpt.jls" \
        > "$log" 2>&1 &
    echo "  pid $!"
done
wait
