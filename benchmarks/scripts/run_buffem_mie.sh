#!/bin/bash
# Run buff-scatter on a sphere mesh for the constant-epsilon Mie benchmark.
#
# Usage:
#   ./benchmarks/scripts/run_buffem_mie.sh MESH OMEGAFILE LABEL [EPS_REAL] [EPS_IMAG]
#
# Example:
#   ./benchmarks/scripts/run_buffem_mie.sh \
#       benchmarks/external/buff-em/examples/MieScattering/Sphere_677.vmsh \
#       benchmarks/runs/omega_mie.txt \
#       eps10p1i_Sphere_677  10  1
#
# Output: PFT file written to benchmarks/runs/<LABEL>.PFT
#
# IMPORTANT: buff-scatter hangs on terminal stdin; always run with </dev/null.
# First invocation computes the FIBBI cache (~10 min for Sphere_677).
# Subsequent runs on the same mesh reuse the cache and are much faster.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 MESH OMEGAFILE LABEL [EPS_REAL=10] [EPS_IMAG=1]" >&2
    exit 1
fi

MESH=$1
OMEGAFILE=$2
LABEL=$3
EPS_REAL=${4:-10}
EPS_IMAG=${5:-1}

# Resolve to absolute paths
MESH=$(readlink -f "$MESH")
OMEGAFILE=$(readlink -f "$OMEGAFILE")

# Repository root
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNDIR="$REPO/benchmarks/runs"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

# Activate the buff-em environment
# shellcheck source=../external/env.sh
source "$REPO/benchmarks/external/env.sh"

# Generate buff-em geometry file
GEOFILE="$LABEL.buffgeo"
cat > "$GEOFILE" <<EOF
MATERIAL ConstDielectric
  Eps(w) = ${EPS_REAL} + ${EPS_IMAG}i;
ENDMATERIAL

OBJECT TheSphere
  MESHFILE ${MESH}
  MATERIAL ConstDielectric
ENDOBJECT
EOF

PFTFILE="$LABEL.PFT"
LOGFILE="$LABEL.log"

echo "=== buff-scatter run: $LABEL ==="
echo "  mesh:      $MESH"
echo "  omegas:    $OMEGAFILE"
echo "  eps_p:     ${EPS_REAL} + ${EPS_IMAG}i"
echo "  geo file:  $RUNDIR/$GEOFILE"
echo "  PFT out:   $RUNDIR/$PFTFILE"
echo "  log:       $RUNDIR/$LOGFILE"
echo

# Run buff-scatter. stdin MUST be /dev/null otherwise it hangs reading socket.
buff-scatter \
    --geometry "$GEOFILE" \
    --OmegaFile "$OMEGAFILE" \
    --PFTFile "$PFTFILE" \
    --pwDirection 0 0 1 \
    --pwPolarization 1 0 0 \
    </dev/null 2>&1 | tee "$LOGFILE"

echo
echo "=== Results ($PFTFILE) ==="
cat "$PFTFILE"
