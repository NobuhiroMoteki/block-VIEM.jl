# External Benchmark Tools

This directory contains **buff-em** and **scuff-em** (by Homer Reid), built
from source for use as reference implementations when validating BlockVIEM.jl.

- `buff-em/`  — source tree of [buff-em](https://github.com/HomerReid/buff-em),
  a volume-integral-equation EM solver using SWG (RT0) basis functions on
  tetrahedral meshes.
- `scuff-em/` — source tree of [scuff-em](https://github.com/HomerReid/scuff-em),
  the surface-integral-equation companion. Provides `libscuff` which buff-em
  depends on.
- `opt/`      — install prefix containing built binaries, libraries, headers,
  and development tools (m4, autoconf, automake, libtool, OpenBLAS). Not
  checked into git (large: ~150 MB).

All three directories are `.gitignore`-d to keep the repository small.

## Why rebuild yourself

These tools were built from source because neither buff-em nor scuff-em are
packaged in apt/conda on this system, and sudo is unavailable. The build
creates a self-contained installation in `opt/` with no system-wide changes.

Total size: ~520 MB across source trees and install prefix.

## Activating the build environment

Source `env.sh` to get the binaries on `$PATH` and configure library paths:

```bash
source benchmarks/external/env.sh
buff-scatter --help      # works from any directory thereafter
```

The helper scripts in `benchmarks/scripts/` source this automatically.

## Running a Mie benchmark

```bash
# Omega file (size parameters for R=1 sphere in units of c/R)
cat > benchmarks/runs/omega_mie.txt <<EOF
0.01
0.1
0.316
0.631
EOF

# Run buff-scatter on the Sphere_677 mesh with eps = 10+1i
./benchmarks/scripts/run_buffem_mie.sh \
    benchmarks/external/buff-em/examples/MieScattering/Sphere_677.vmsh \
    benchmarks/runs/omega_mie.txt \
    eps10p1i_sphere677 10 1
```

Output is written to `benchmarks/runs/<LABEL>.PFT`, containing absorbed and
scattered **power** in watts (for a unit-amplitude plane wave). Convert to
cross sections via `sigma = 2 * Z_0 * P` with `Z_0 = 376.73031346177075` Ω.

## Known gotchas

1. **`buff-scatter` must be run with `</dev/null`** — if stdin is a terminal
   the process hangs forever waiting on a socket read.

2. **First run is slow** (~10 min for Sphere_677). buff-em computes a FIBBI
   (Frequency-Independent Basis-Basis Integrals) cache for the mesh on its
   first invocation. Subsequent runs on the same mesh reuse the cache.

3. **Binaries have stale RUNPATH**. The libtool build embeds the original
   install path (`/home/moteki/local/opt/lib`) as RUNPATH in the ELF headers.
   After relocation to `benchmarks/external/opt`, that path no longer exists,
   so `LD_LIBRARY_PATH` must be set via `env.sh` to find the libraries. No
   symlink workaround is needed as long as the helper scripts are used.

4. **SWG (RT0) is the only basis**. buff-em does not implement RT1 or higher-
   order H(div) elements. For RT1 validation, we need to compare against Mie
   theory directly (no code-to-code comparison available).

## Rebuilding from scratch

If `benchmarks/external/opt`, `buff-em/`, or `scuff-em/` are lost, follow
these steps to rebuild. Requires: `g++`, `gfortran`, `make`, `cmake`, `curl`,
`git`. No sudo needed.

```bash
cd benchmarks/external
export PREFIX=$PWD/opt
mkdir -p $PREFIX/bin $PREFIX/lib $PREFIX/include
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export CPPFLAGS="-I$PREFIX/include -I$PREFIX/include/scuff-em"
export LDFLAGS="-L$PREFIX/lib -lopenblas"
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig

# 1. OpenBLAS (provides LAPACK too)
git clone --depth 1 https://github.com/OpenMathLib/OpenBLAS.git /tmp/OpenBLAS
make -C /tmp/OpenBLAS -j$(nproc) NO_LAPACK=0 PREFIX=$PREFIX
make -C /tmp/OpenBLAS install PREFIX=$PREFIX

# 2. GNU build tools (m4 → autoconf → automake → libtool)
for pkg in \
    "m4/m4-1.4.19.tar.xz" \
    "autoconf/autoconf-2.71.tar.xz" \
    "automake/automake-1.16.5.tar.xz" \
    "libtool/libtool-2.4.7.tar.xz"; do
    name=$(basename $pkg .tar.xz)
    curl -sL https://ftp.gnu.org/gnu/$pkg | tar xJ -C /tmp
    (cd /tmp/$name && ./configure --prefix=$PREFIX -q && make -j$(nproc) -s && make install -s)
done

# 3. scuff-em
git clone --depth 1 https://github.com/HomerReid/scuff-em.git
cd scuff-em
sh autogen.sh --prefix=$PREFIX --without-hdf5 \
    --with-lapack="-L$PREFIX/lib -lopenblas" \
    CPPFLAGS="$CPPFLAGS" LDFLAGS="$LDFLAGS"
make -j$(nproc) && make install
cd ..

# 4. buff-em
git clone --depth 1 https://github.com/HomerReid/buff-em.git
cd buff-em
# buff-analyze has an API incompatibility with current scuff-em; skip it.
# Only libbuff and buff-scatter are needed for Mie benchmarking.
libtoolize --force --copy && aclocal -I m4 --force && autoconf --force
autoheader && automake --add-missing --force-missing --copy
./configure --prefix=$PREFIX \
    --with-lapack="-L$PREFIX/lib -lopenblas" \
    CPPFLAGS="$CPPFLAGS" LDFLAGS="$LDFLAGS"
make -C src/libs/libbuff -j$(nproc)
make -C src/applications/buff-scatter -j$(nproc)
# Install manually (make install fails in buff-analyze)
cp src/libs/libbuff/.libs/libbuff.so* $PREFIX/lib/
cp src/libs/libbuff/.libs/libbuff.a $PREFIX/lib/
cp src/libs/libbuff/.libs/libbuff.la $PREFIX/lib/
cp src/applications/buff-scatter/.libs/buff-scatter $PREFIX/bin/
```

Total build time: ~20–30 minutes on a modern multi-core machine.
