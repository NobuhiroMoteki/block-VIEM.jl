#!/bin/bash
# Environment setup for using buff-em and scuff-em binaries built into
# benchmarks/external/opt. Source this file before running the binaries:
#
#   source benchmarks/external/env.sh
#   buff-scatter --geometry foo.buffgeo ...
#
# Or use the helper scripts in benchmarks/scripts/ which source this
# automatically.

# Detect repository root (directory containing this script's parent)
_EXTDIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
export BUFFEM_PREFIX="${_EXTDIR}/opt"

# Prepend tool binaries to PATH
export PATH="${BUFFEM_PREFIX}/bin:${PATH}"

# Library path (handles RUNPATH mismatch after relocation).
# Guard against `set -u` by defaulting empty values.
export LD_LIBRARY_PATH="${BUFFEM_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# pkg-config and headers for rebuilds
export PKG_CONFIG_PATH="${BUFFEM_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CPPFLAGS="-I${BUFFEM_PREFIX}/include -I${BUFFEM_PREFIX}/include/scuff-em ${CPPFLAGS:-}"
export LDFLAGS="-L${BUFFEM_PREFIX}/lib ${LDFLAGS:-}"

unset _EXTDIR
