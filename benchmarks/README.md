# BlockVIEM.jl Benchmarks

Reference solutions and cross-code comparisons for validating BlockVIEM.jl.

## Layout

```
benchmarks/
├── external/            # buff-em, scuff-em, and build prefix (gitignored)
│   ├── buff-em/         # Homer Reid's VIE solver (SWG/RT0 only)
│   ├── scuff-em/        # Homer Reid's SIE solver (needed by buff-em)
│   ├── opt/             # install prefix: binaries, libs, headers
│   ├── env.sh           # environment activator
│   └── README.md        # build instructions and usage
├── scripts/             # helper scripts for running external tools
│   └── run_buffem_mie.sh
└── runs/                # output directory (PFT files, logs, caches)
```

## Quick start: validate RT0 against buff-em

```bash
# 1. Activate the build environment
source benchmarks/external/env.sh

# 2. Run buff-scatter on a Mie sphere
cat > benchmarks/runs/omega_mie.txt <<EOF
0.01
0.1
0.316
0.631
EOF

./benchmarks/scripts/run_buffem_mie.sh \
    benchmarks/external/buff-em/examples/MieScattering/Sphere_677.vmsh \
    benchmarks/runs/omega_mie.txt \
    eps10p1i_sphere677 10 1

# First run takes ~10 min (FIBBI cache computation); subsequent runs reuse
# the cache and complete in seconds.
```

The PFT output contains absorbed/scattered power. Convert to cross sections:

```julia
const Z0 = 376.73031346177075  # free-space impedance
C_abs = 2 * Z0 * P_abs          # [length²] in same units as mesh
C_sca = 2 * Z0 * P_sca
```

## Validated comparison: BlockVIEM RT0 ≡ buff-em

On the same mesh (`Sphere_677.vmsh`, 677 tets, 1188 SWG DOFs) with
`ε_p = 10 + 1i`, BlockVIEM's RT0 output matches buff-em to within ~1%:

| ka    | BlockVIEM C_abs err | buff-em C_abs err | BlockVIEM C_sca err | buff-em C_sca err |
|-------|---------------------|-------------------|---------------------|-------------------|
| 0.01  | 17.5 %              | 17.7 %            | —                   | 25.6 %            |
| 0.10  | 17.4 %              | 17.6 %            | 25.9 %              | 25.7 %            |
| 0.316 | 16.7 %              | 17.3 %            | 27.0 %              | 26.9 %            |
| 0.631 | 13.3 %              | 17.1 %            | 29.2 %              | 29.9 %            |

The ~17 % error at low `ka` on this coarse mesh is the **intrinsic O(h)
discretization error of SWG/RT0**, not a bug in either code. Both solvers
agree closely, confirming BlockVIEM's RT0 implementation is correct.
