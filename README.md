# block-VIEM.jl

[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FNobuhiroMoteki%2Fblock-VIEM.jl%2Fmain%2FProject.toml&query=%24.version&prefix=v&label=version&color=blue&cacheSeconds=300)](https://github.com/NobuhiroMoteki/block-VIEM.jl/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia ≥ 1.10](https://img.shields.io/badge/julia-%E2%89%A5%201.10-blueviolet.svg)](https://julialang.org/)

A Julia implementation of the **Volume Integral Equation Method (VIEM)** for
electromagnetic scattering by arbitrarily shaped, high-contrast dielectric
particles (e.g., iron-oxide aggregates, rough gold nanoparticles).

block-VIEM.jl is designed as a **higher-accuracy successor** to
[`block-DDA_Py`](https://github.com/NobuhiroMoteki/block-DDA_Py),
targeting scattering problems where DDA does not converge or provides
insufficient accuracy (high refractive-index contrast, plasmonic
particles, extreme aspect ratios). The two codes share the same physical
input conventions, HDF5 output schema, and CAS-v2 observable definitions,
so downstream analysis pipelines work with either without modification.

## Features

1. **Higher accuracy per DOF than DDA.** Tetrahedral discretization with
   SWG (half-SWG) basis functions and Duffy-transform singular integration
   achieves 10-12x better accuracy per degree of freedom than DDA on
   equivalent problems.
2. **AIM/FFT-accelerated block-Krylov solver.** The Adaptive Integral
   Method (AIM) computes matrix-vector products in O(N log N) via FFT.
   Block GMRES (default since v0.7.1) / Block BiCGSTAB solve all
   particle orientations simultaneously. AIM grid pitch is auto-detected
   from the mesh.
3. **Multi-orientation batch solve.** Supports deterministic uniform Euler
   angle grids on SO(3) (`N_alpha` x `N_beta` x `N_gamma`), identical to
   block-DDA_Py. **Spheroid mode** (`ab_ratio=1`, `beta=0`) solves only
   `N_beta` orientations and fills the full grid analytically via the
   alpha-expansion `S_theta(alpha) = A + B*exp(+2j*alpha)`.
4. **Birefringent (anisotropic) materials.** Diagonal permittivity tensor
   `eps_p = diag(eps_x, eps_y, eps_z)` is supported throughout the solver,
   post-processing, and sweep scripts.
5. **Gaussian Random Ellipsoid (GRE) and sphere-aggregate shape models.**
   Parametric irregular shapes (`GREParams`) and discrete-monomer
   clusters (`SphereAggregate`, with built-in chain / planar / FCC /
   BCC / HCP generators and PTSA HDF5 loader) are both meshed via
   Gmsh OpenCASCADE with adaptive mesh size and a volume-preserving
   final rescale (block-DDA_Py-compatible). See
   [docs/descriptions_particle_shape_model.md](docs/descriptions_particle_shape_model.md)
   for the full specification of parameter ranges, the `neck_ratio`
   convention, the volume-preserving rescale, and discretisation
   examples.
6. **CAS-v2 observables.** Computes the polarized complex forward-scattering
   amplitudes `S(0)_theta`, `S(0)_phi` (PCAS) and the complex
   backward-scattering amplitude `S(180)` (OCBS), plus `C_ext`, `C_abs`,
   `C_sca`.
7. **Production sweep with resume.** `run_viem.jl` performs multi-dimensional
   parameter sweeps (wavelength x refractive index x shape x orientation),
   writes results to HDF5 in the block-DDA_Py-compatible schema, and
   resumes from partially completed files. On solver failure, fills NaN and
   continues (matching block-DDA_Py).
8. **Mie reference.** Each sweep condition is compared to the Mie solution
   for the volume-equivalent sphere with axis-averaged refractive index.

### Validation

block-VIEM.jl is validated against four independent references:
the **Mie series** (sphere, dielectric and plasmonic Au), an
analytical **α-symmetry identity** (oblate spheroid), the
multi-sphere T-matrix package **MSTMforCAS.jl** (touching doublet,
dielectric and plasmonic Au), and **block-DDA_Py** (spheroid, GRE
β > 0, anisotropic ε_p). Sub-1 % accuracy on all five CAS-v2
observables is reached at modest mesh sizes for the Mie reference;
agreement against the independent codes is at the few-percent
level expected from the discretisation budget of either side. All
numerical conditions, per-observable tables, mesh-refinement rates,
and AIM-operator memory measurements are collected in
**[docs/benchmark_results.md](docs/benchmark_results.md)**.

## When to use block-VIEM.jl vs block-DDA_Py

block-DDA_Py and block-VIEM.jl solve the same physics and produce
the same observables in the same HDF5 format. The choice depends on
refractive-index contrast and geometry.

- **Low contrast (|m_p| < 2)**: prefer **block-DDA_Py**. DDA's exact
  cubic-lattice FFT-MVP gives effectively-zero setup; sub-2 %
  accuracy at ~ 300 dipoles in < 0.1 s. VIEM matches that accuracy
  but needs ~ 2 000 half-SWG DOFs and ~ 10 s of setup.
- **High contrast (|m_p| > 2 – 3, plasmonic metals, extreme aspect
  ratios)**: prefer **block-VIEM.jl**. DDA converges as
  $\mathcal{O}(N^{-1/2})$ here and may stall at 4 – 7 % error;
  VIEM's $H(\mathrm{div})$-conforming half-SWG basis on a
  tetrahedral mesh converges as $\mathcal{O}(h^2)$ and reaches
  sub-1 % at $N \sim 2 \times 10^3$ DOFs even on plasmonic Au.

The per-DOF efficiency gap (~ 10 – 12 × in VIEM's favour at high
contrast), the four mechanisms behind it (conforming geometry,
$H(\mathrm{div})$-conforming basis, linear vector basis, and no
polarisability tuning), and the asymptotic cost model are documented
with full numerical tables in
**[docs/benchmark_results.md §9](docs/benchmark_results.md#9-per-dof-accuracy-vs-block-dda_py)**.

### Summary decision rule

```text
if |m_p / m_m| < 2 and DDA converges:
    use block-DDA_Py         # 100-1000x faster setup
else:
    use block-VIEM.jl        # converges where DDA cannot
```

## Quick start

```julia
using BlockVIEM, StaticArrays
using BlockVIEM: Vec3

# 1. Load a tetrahedral mesh produced by Gmsh (.msh format)
mesh  = read_msh("sphere.msh")
basis = build_swg_basis(mesh; include_boundary_faces = true)   # half-SWG

# 2. Multi-orientation solve (block-DDA_Py-compatible API)
#    Pass (wl_0, m_m, m_p) exactly as in block-DDA_Py.
euler_list = [(0.0, 0.0, 0.0), (0.3, 0.7, 0.5), (0.0, π/2, 0.0)]

results = solve_cas_v2_orientations(
    basis, euler_list;
    wl_0 = 0.638,                # vacuum wavelength [μm]
    m_m  = 1.0,                  # background refractive index
    m_p  = 1.5 + 0.01im,        # particle refractive index (Im > 0 = absorbing)
)
# results[i].S_fw_mean, results[i].S_bk, results[i].S_fw_theta, ...
```

For **anisotropic** particles, pass `m_p` as a 3-element vector
`[m_x, m_y, m_z]` (principal-axis refractive indices in the particle
frame), matching block-DDA_Py's `m_p_xyz`:

```julia
results = solve_cas_v2_orientations(
    basis, euler_list;
    wl_0 = 0.638, m_m = 1.0,
    m_p  = [1.6 + 0im, 1.5 + 0im, 1.4 + 0im],   # birefringent
)
```

Alternatively, pass the raw VIEM form `(k0, eps_p, eps_bg)` where
`k0` is the background-medium wavenumber (`2π·m_m/λ₀`, **not** the
vacuum wavenumber) and `eps_p`, `eps_bg` are absolute permittivities.
`eps_p` can be a scalar or a 3-vector `[ε_x, ε_y, ε_z]`.
See [`solve_cas_v2_orientations`](src/postprocess.jl) docstring.

### Solver selection

The default solver (since v0.7.1) is **AIM + Block GMRES**
(`method = :aim_gmres`), which uses FFT-accelerated matrix-vector
products (O(N log N) per iteration) and solves all orientations
simultaneously via a block-Krylov iteration.  GMRES is chosen over
BiCGSTAB as the default because it is monotone-convergent and robust
against near-linearly-dependent RHS columns (which can break BiCGSTAB
at large block sizes, observed at L ≥ 64 in paper-production pilots).
The AIM grid pitch is auto-detected from the mesh
(`0.5 * mean_edge_length`) when not supplied explicitly.

```julia
# Default: AIM + Block GMRES (pitch auto-detected)
results = solve_cas_v2_orientations(
    basis, euler_list;
    wl_0 = 0.638, m_m = 1.0, m_p = 1.5 + 0.01im,
)

# Override solver parameters
results = solve_cas_v2_orientations(
    basis, euler_list;
    wl_0 = 0.638, m_m = 1.0, m_p = 1.5 + 0.01im,
    method  = :aim_bicgstab,  # :aim_gmres (default), :aim_bicgstab, :dense
    pitch   = 0.01,           # AIM grid pitch [μm] (auto if omitted)
    padding = 4,
    tol     = 1e-8,
    maxiter = 400,
)
```

Available methods:

- **`:aim_gmres`** (default, v0.7.1+) — unrestarted Block GMRES (Simoncini-Szyld 1996).
  Monotone; robust against near-linearly-dependent RHS columns.
- **`:aim_bicgstab`** — Block BiCGSTAB (Tadano-Sakurai-Kuramashi 2009).
  Typically fewer iterations than GMRES but can break down on degenerate RHS blocks.
- **`:dense`** — assemble Z and LU-factorize. Only for small problems (N < 10^3).

Both block solvers are also exposed directly as
`block_bicgstab(A, B; tol, maxiter)` and `block_gmres(A, B; tol, maxiter)`
for any linear operator `A` supporting `A * X::AbstractMatrix`.

The `return_D=true` keyword causes `solve_cas_v2_orientations` to return
`(results, D_block)` instead of just `results`, giving access to the raw
D-field expansion coefficients for computing cross sections or other
post-processing.

### Parallel execution

Launch Julia with `-t N` (or set `JULIA_NUM_THREADS=N`) and all of
the following run in parallel automatically:

- **Setup** — `assemble_precorrection` (the dominant setup cost,
  ≳90 % of `build_aim_operator` wall time at N ≳ 10³) and
  `build_aim_projection` partition their outer loops into one
  `Threads.@spawn` task per chunk. Per-task COO buffers are merged
  into the final sparse matrices at the end. Near-field Z-matrix and
  half-SWG assembly are similarly threaded (`Threads.@threads`).
- **Block MVP** — `aim_mvp(op, X)` dispatches the `L` right-hand-side
  columns across Julia threads; each column's FFT convolution runs
  concurrently.
- **FFT convolutions** — FFTW's own thread pool is initialised at
  module load to `Threads.nthreads()`. Override with
  `BlockVIEM.set_fft_threads(n)`; for block MVPs with `L` RHSs a good
  balance is `set_fft_threads(max(1, cld(Threads.nthreads(), L)))`.
- **Far-field radiation** and **multi-orientation RHS construction**
  run one `Threads.@spawn` task per direction / orientation.

Measured on a 20-core Intel Xeon against Sphere_1675 (N = 3041
half-SWG DoFs), 4 threads vs. serial: `assemble_precorrection`
2.5 ×, block MVP (`L` = 4) 2.2 ×, block MVP (`L` = 8) 2.6 ×.

**Reuse across parameter sweeps.** `build_aim_operator` accepts
pre-computed `projection` and `mass` keyword arguments, which are
`k₀` / `ε_p`-independent. For wavelength or material sweeps, build
them once and pass them back in — only the Green FFT and
precorrection are rebuilt on each new `(k₀, ε_p)`:

```julia
grid = aim_grid(basis.mesh; pitch = pitch, padding = 4)
proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
mass = assemble_mass_matrix(basis)

for λ in wavelengths
    op = build_aim_operator(basis; k0 = 2π / λ, eps_p = eps_p,
                             projection = proj, mass = mass)
    # ... solve and post-process
end
```

### Particle shape models

The two shape families supported by block-VIEM.jl — Gaussian Random
Ellipsoid (GRE) and sphere aggregates — share the same Gmsh-based
discretisation pipeline and a common volume-preserving rescale.
Parameter ranges, the `neck_ratio` neck-width convention, the
volume-preserving rescale (v0.7.7+), discretisation examples, and
recommended usage patterns are documented in
**[docs/descriptions_particle_shape_model.md](docs/descriptions_particle_shape_model.md)**.

Wireframe galleries of representative discretised targets are
generated by

```bash
julia --project=viz viz/visualize_gre.jl         # GRE gallery
julia --project=viz viz/visualize_aggregate.jl   # sphere-aggregate gallery
```

writing PNGs to `viz/figs/`.

### Parameter-sweep HDF5 I/O

For multi-dimensional parameter sweeps (`wl_0 × m_p × r_v_base ×
bc_ratio × ab_ratio × gre_beta × orientation`), block-VIEM.jl provides
scripts that create, run, and inspect HDF5 files in the same schema as
block-DDA_Py (`dda_results/create_h5py.ipynb` / `run_dda.py` /
`check_h5py.ipynb`):

```bash
# 1. Edit sweep parameters at the top of create_h5.jl, then run:
julia --project=. viem_results/create_h5.jl

# 2. Run the production sweep (fills the HDF5 with VIEM results + Mie reference):
julia --project=. -t auto viem_results/run_viem.jl

# 3. Inspect completion status and results:
julia --project=. viem_results/check_h5.jl
```

`run_viem.jl` is the Julia equivalent of block-DDA_Py's `run_dda.py`:

- Uses **AIM + Block GMRES** by default (since v0.7.1; O(N log N) per iteration)
- Automatically detects **spheroid mode** (`ab_ratio == 1` and
  `gre_beta == 0`): solves only `N_beta` orientations at α = 0, then
  fills the full `(N_alpha × N_beta × N_gamma)` grid analytically via
  `S_θ(α) = A + B·exp(+2jα)`
- **Reuses projection + mass matrices** across the inner
  `(wl_0, m_p)` loop. Each `shape_idx` builds one worst-case mesh
  (using the shortest `wl_0` and largest `|m_p|` anywhere in the
  sweep) and caches its `AIMProjection` / mass matrix; every inner
  iteration then rebuilds only the Green FFT and the precorrection.
  Set `REUSE_PROJECTION_PER_SHAPE = false` at the top of the script
  to restore per-(wl_0, m_p) adaptive mesh sizing instead.
- **Resume**: skips already-computed conditions (checks `S_fw_PCAS_mie`)
- **Failure handling**: on solver error, fills NaN and continues to the
  next condition (matching block-DDA_Py); only Ctrl+C stops the sweep
- Computes Mie reference (volume-equivalent sphere) for each condition
- Accepts an optional filename argument:
  `julia --project=. viem_results/run_viem.jl path/to/file.hdf5`

The HDF5 layout (`/target/simulated_data/...`) with datasets
`S_fw_PCAS_theta`, `S_fw_PCAS_phi`, `S_bk_OCBS`, `C_ext`, `C_abs`,
`Euler_angles`, and Mie reference values is byte-compatible with
block-DDA_Py, so downstream consumers (e.g. `build_spheroid_lut.py`)
can read either DDA or VIEM output without modification.

## Paper-production workflow (v0.7.0)

[viem_results/paper/](viem_results/paper/) contains a self-contained
scaffold for running publication-quality sweeps against a fixed
calculation matrix (4 shapes × 3 materials × up to 4 sizes, fixed
wavelength λ₀ = 0.638 μm, vacuum background). See
[CLAUDE.md](CLAUDE.md) for the full specification.

### Shape × material templates

Twelve short generators write block-DDA_Py-compatible sweep HDF5s via
a shared schema helper [viem_results/paper/_common.jl](viem_results/paper/_common.jl):

```bash
# sphere / oblate / gre / doublet × n15 / n20 / Au
julia --project=. viem_results/paper/sphere_n20.jl
julia --project=. viem_results/paper/doublet_Au.jl
# ... etc.
```

Materials (`n_p @ λ=0.638 μm`) are hard-coded as `N_LOW = 1.5+0.01i`,
`N_20 = 2.0+0.0i` (paper "high" since v0.7.5; previously `N_HIGH = 3.17+0.16i`
kept for reference), `N_AU = 0.17525+3.4830i` (Johnson & Christy 1972).
The `doublet` shape is a two-sphere cluster with monomer radius
`R = a_eq / 2^(1/3)`, axis-direction surface gap `g = 0.1 R`, axis aligned
with particle z so that `run_viem.jl`'s spheroid α-expansion applies
directly (cylindrical symmetry).

### Pre-run cost estimator

[viem_results/estimate_cost.jl](viem_results/estimate_cost.jl) reads the
same HDF5 and prints estimated `N_DOF`, peak RSS, setup + solve times
per shape slot. It flags slots exceeding 24 h of wall time or 90 % of
`MemAvailable`, to help decide whether to downsize a sweep before
launching `run_viem.jl`:

```bash
julia --project=. -t auto viem_results/estimate_cost.jl \
    viem_results/paper/sphere_n20.hdf5
```

Calibrated against empirical phase-A + pilot-run data; tune per-DOF
constants via `T_SETUP_MS_DOF`, `T_ITER_MS_DOF`, `RSS_KB_PER_DOF`,
`N_ITER_EST` environment variables.

### Block-Krylov RHS-scaling diagnostic

[viem_results/paper/run_rhs_scaling.jl](viem_results/paper/run_rhs_scaling.jl)
measures, per shape slot, `block_gmres` at `L ∈ {1, 2, 4, 8, 16, 32, 64, 128}`
RHS using the shared worst-case mesh + projection + mass (BiCGSTAB
dropped from the paper scope at v0.7.6 to focus on shape × material ×
N_DOF scaling):

```bash
julia --project=. -t auto viem_results/paper/run_rhs_scaling.jl \
    viem_results/paper/gre_n20.hdf5
```

Results are written to `/target/rhs_scaling/gmres/` with datasets
`iters`, `converged`, `t_total_s`, `t_end2end_per_orient_s`,
`peak_rss_bytes`, `residual_history` (last shape `(nL, N_rv, N_bc, N_ab,
N_bt, MAXITER)`).  The diagnostic relies on
`solve_cas_v2_orientations(return_solve_info=true)` and the
`BlockSolveResult.residual_history` field.

### MSTM exact reference for doublet

[viem_results/paper/run_mstm_reference.jl](viem_results/paper/run_mstm_reference.jl)
runs in the `MSTMforCAS.jl` environment and consumes a doublet sweep
HDF5 to produce a separate `mstm_<basename>.hdf5` indexed by
`(a_eq, β)` with the numerically exact CAS-v2 observables at
`truncation_order = 15` (converged for both dielectric and plasmonic
monomers, see `benchmarks/cas_v2/doublet_mstm/`):

```bash
julia --project=/path/to/MSTMforCAS.jl \
    viem_results/paper/run_mstm_reference.jl \
    viem_results/paper/doublet_n20.hdf5
# → viem_results/paper/mstm_doublet_n20.hdf5
```

### lc-convergence study

[viem_results/paper/run_lc_convergence.jl](viem_results/paper/run_lc_convergence.jl)
sweeps five mesh-size factors (1.5, 1.0, 0.7, 0.5, 0.35 × `adaptive_lc`)
for one `(shape, material)` at `a_eq = 0.1 μm`, single orientation, and
writes `convergence_<shape>_<material>.hdf5`:

```bash
julia --project=. -t auto viem_results/paper/run_lc_convergence.jl \
    sphere n20
```

### End-to-end pipeline

```text
create_h5 (template)  →  estimate_cost  →  run_viem  →  check_h5
                                       ↘  run_rhs_scaling
                                       ↘  run_mstm_reference (doublet only)
                                       ↘  run_lc_convergence (per (shape, material))
```

## Benchmark: 2-sphere doublet CAS-v2 vs MSTM

Cross-validation against [MSTMforCAS.jl](https://github.com/MOTEKI-LAB/MSTMforCAS.jl)
on a touching doublet of equal spheres (radius 0.030 μm, gap
0.003 μm, λ₀ = 0.638 μm) at three orientations and two materials
(polystyrene-like dielectric and Johnson & Christy 1972 gold).
Numerically exact MSTM reference at VSWF truncation $N = 15$.
For the dielectric doublet $|S_\text{fw,mean}|$ matches MSTM to
~ 1.6 %; for the plasmonic Au doublet the error grows from 1.3 % at
β = 0 to 4.0 % at β = π/2 with strong polarisation anisotropy
(Im $S_{\text{fw},\theta}$ reaching 10.5 % at β = π/2). Mesh
refinement to $\ell_c = R/8$ confirms the theoretical SWG
convergence rate $p \approx 2$ on every observable.

**Note on MSTMforCAS truncation order.** The MSTMforCAS
auto-truncation returns $N_\text{trunc} = 3$ at $x \approx 0.3$,
which is converged for the dielectric case but under-truncates the
plasmonic Au case (roughly 5 % shift in $|S_\text{fw,mean}|$
between $N = 3$ and the converged reference).  We therefore force
$N_\text{trunc} = 15$ in `run_mstm.jl` for both materials, reaching
rtol $\le 10^{-6}$ — see `check_trunc_convergence.jl`.

Full numerical conditions, per-orientation Re/Im-resolved tables,
the $\ell_c$-refinement convergence-rate fit, and the Phase A
AIM-memory scaling are documented in
**[docs/benchmark_results.md §7–§8](docs/benchmark_results.md#7-two-sphere-doublet-vs-mstm)**.

```bash
# Reproduce the full benchmark
julia --project=. benchmarks/cas_v2/doublet_mstm/run_viem.jl
julia --project=/path/to/MSTMforCAS.jl benchmarks/cas_v2/doublet_mstm/run_mstm.jl
julia --project=. benchmarks/cas_v2/doublet_mstm/compare.jl
```

## Conventions

- **Time convention:** physics, `exp(−iωt)`, matching block-DDA_Py and
  the Mie reference. Outgoing scalar Helmholtz Green's function is
  `exp(+ik₀R)/(4πR)`; the far-field integrand carries `exp(−ik₀ r̂·r')`.
- **Units:** SI / Lorentz–Heaviside (the scalar Green's function has the
  `1/(4π)` factor). `compute_cas_observables` divides `F` by `4π` to
  match the block-DDA_Py / Mie scattering amplitude definition.
- **Complex refractive index:** absorbing materials have `Im(m_p) > 0`
  (and therefore `Im(ε_p) > 0`). The `(wl_0, m_m, m_p)` physical
  input API matches block-DDA_Py exactly — the same values can be
  passed to both codes.
- **Euler angles:** ZYZ intrinsic, same as
  `scipy.spatial.transform.Rotation` and block-DDA_Py.
  `R(α,β,γ) = Rz(α)·Ry(β)·Rz(γ)` maps particle-frame to lab-frame;
  lab-to-particle uses `R^T`.

## Design references

- `docs/theory_note.tex` — full formulation reference: EFVIE-D, SWG /
  half-SWG, Duffy transform, AIM, block-Krylov, CAS-v2 observables,
  Mie validation (dielectric + plasmonic Au), spheroid benchmarks
- `docs/descriptions_particle_shape_model.md` — GRE and sphere-aggregate
  shape models, discretisation, `neck_ratio` convention, and
  volume-preserving rescale
- `docs/benchmark_results.md` — all validation and cross-comparison
  benchmarks (Mie, MSTM, block-DDA_Py) with full numerical conditions
  and per-observable error tables
- `docs/io_spec.md` — block-DDA_Py compatible I/O specification
- `.claude/reference/` — primary literature (SWG 1984, Volakis-Sertel,
  Sheng-Song, Mousavi-Sukumar 2010)

## Installation

For a from-scratch setup on **Windows + WSL2 (Ubuntu)** or **native
Ubuntu** — including Julia 1.12.5 via `juliaup`, system-package
prerequisites for `Gmsh.jl`, the `viz/` figure-generation
sub-environment, and the test-suite smoke test — see the dedicated
**[Installation Guide](Installation_guide.md)**.

Quick development setup once the prerequisites are in place
(Julia ≥ 1.10, Gmsh's runtime libraries, a clone of this repository):

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
julia> Pkg.test()
```

## License

MIT (see `LICENSE`).

## Author

Nobuhiro Moteki ([@NobuhiroMoteki](https://github.com/NobuhiroMoteki),
`nobuhiro.moteki@gmail.com`)
