# block-VIEM.jl

[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FNobuhiroMoteki%2Fblock-VIEM.jl%2Fmain%2FProject.toml&query=%24.version&prefix=v&label=version&color=blue&cacheSeconds=300)](https://github.com/NobuhiroMoteki/block-VIEM.jl/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia ≥ 1.10](https://img.shields.io/badge/julia-%E2%89%A5%201.10-blueviolet.svg)](https://julialang.org/)

A Julia implementation of the **Volume Integral Equation Method (VIEM)** for
electromagnetic scattering by arbitrarily shaped, high-contrast dielectric
particles (e.g., iron-oxide aggregates, rough gold nanoparticles).

**Changes and corrections:** see [CHANGELOG.md](CHANGELOG.md). Its first section lists the corrections that change numerical results, with the affected versions.

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
   backward-scattering amplitude `S(180)` = (−*S*<sub>11</sub> + *S*<sub>22</sub>)/2
   (OCBS, corrected definition from v0.8.0, see Conventions), plus `C_ext`,
   `C_abs`, `C_sca`.
7. **Production sweep with resume.** `run_viem.jl` performs multi-dimensional
   parameter sweeps (wavelength x refractive index x shape x orientation),
   writes results to HDF5 in the block-DDA_Py-compatible schema, and
   resumes from partially completed files. On solver failure, fills NaN and
   continues (matching block-DDA_Py).
8. **Mie reference.** Each sweep condition is compared to the Mie solution
   for the volume-equivalent sphere with axis-averaged refractive index.
9. **Refractive-index sweep with one Krylov process (v0.8.0).** For a
   homogeneous isotropic particle all refractive indices on one mesh are
   solved by a single shifted block-GMRES process. The AIM operator is
   built once per mesh. Measured gain over one solve per index: about 3,
   10 and 29–30 for 3, 10 and 30 indices. Available in the sweep drivers
   as `--shifted-ri`.

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

For an anisotropic particle the mass term of the impedance matrix is
`Σ_α (1/ε_α) M_α`. Up to v0.8.0 the AIM path (`:aim_gmres`,
`:aim_bicgstab`) used the mean of `1/ε_α` instead, which changes the
observables by 5–25 % for index contrasts between the axes of 0.1–0.5.
Anisotropic results of those versions are correct only with
`method = :dense`. The isotropic case was not affected.

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

### Refractive-index sweep with one Krylov process (v0.8.0)

For a homogeneous isotropic particle the systems of different refractive
indices on one mesh are shifts of one operator,
`Z(m_p) = (kappa/eps_bg) (tau M - K)` with `tau = 1/((m_p/m_m)^2 - 1)`.
`solve_cas_v2_orientations_shifted` solves all of them, for all
orientations, with a single shifted block-GMRES process:

```julia
m_p_list = [2.7 + 0.1im, 2.9 + 0.1im, 3.1 + 0.1im]
results, info = solve_cas_v2_orientations_shifted(
    basis, euler_list;
    wl_0 = 0.638, m_m = 1.0, m_p_list = m_p_list,
    tol = 1e-5, maxiter = 200,
)
# results[s, i]: CASv2Result of m_p_list[s] and euler_list[i]
# info.iterations, info.converged, info.residual_norm: one entry per refractive index
```

- The AIM operator is built once per mesh instead of once per refractive
  index. The block Krylov space is shared, so the number of block
  products is that of the slowest index. Measured gain over one solve
  per index on the same mesh: 3.0–3.2 for 3 indices, 9.9–10.4 for 10
  and 29–30 for 30 (25,680 to 124,049 unknowns,
  [docs/benchmark_results.md §11](docs/benchmark_results.md#11-shifted-block-gmres-for-refractive-index-sweeps)).
- The convergence criterion is the one of `solve_cas_v2_orientations`,
  `‖B − Z D‖_F / ‖B‖_F < tol`, for every refractive index. The results
  agree with independent solves on the same mesh to the solver tolerance.
- The mesh is shared. Build it for the largest `|m_p|` of the list.
- Refractive indices of similar contrast should be solved together,
  because the slowest one sets the iteration count.
- Isotropic indices only. A 3-component `m_p` raises an `ArgumentError`.

The sweep drivers `benchmarks/cas_v2/ironoxide_sweep_h5.jl`,
`benchmarks/cas_v2/spheroid_sweep_h5.jl` and `viem_results/run_viem.jl`
use this path with `--shifted-ri`. Their default mode is unchanged. The solver itself is
exposed as `block_gmres_shifted(C, B, taus; tol, maxiter)` for any
operator `C` supporting `C * X::AbstractMatrix`. The formulation is in
`docs/theory_note.pdf` (Sec. 11, "Shifted Block GMRES for
Refractive-Index Sweeps").

**Mass-matrix preconditioning for ordinary solves.**
`solve_cas_v2_orientations(...; precond = :mass_right)` runs the block
solver on `Z M⁻¹` (sparse Cholesky of the mass matrix). The convergence
criterion is unchanged. In the tests of the theory note the block-GMRES
iteration count dropped to 0.23–0.59 of the unpreconditioned one. The
default is `precond = :none`.

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
- **Multi-orientation RHS construction** runs one `Threads.@spawn`
  task per orientation. The **far-field integration** for `C_sca` runs
  the scattering directions under `Threads.@threads`.

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
- **`--shifted-ri`** (v0.8.0): solves all isotropic entries of
  `m_p_xyz_list` of a (shape, wavelength) slot with one shifted
  block-GMRES process, as `run_dda.py --shifted-ri` of block-DDA_Py does.
  The mesh is the per-shape mesh of the default mode, so the results
  agree with the default mode to the solver tolerance. Anisotropic
  entries are solved one at a time. In `/target/cost/` the time fields of
  the entries solved together hold the batch time divided by the number
  of entries. A file keeps the mode it was started with (attribute
  `shifted_ri` of `/target`).

The HDF5 layout (`/target/simulated_data/...`) with datasets
`S_fw_PCAS_theta`, `S_fw_PCAS_phi`, `S_bk_OCBS`, `C_ext`, `C_abs`,
`Euler_angles`, and Mie reference values is the one of block-DDA_Py's
`run_dda.py`, so the same analysis scripts read either DDA or VIEM
output. The spheroid look-up-table format read by
`build_spheroid_lut.py` (root grids and one `wl_<value>/` group per
wavelength) is a different file. It is written by
`write_spheroid_sweep_h5`, which the drivers
`benchmarks/cas_v2/spheroid_sweep_h5.jl` and
`benchmarks/cas_v2/ironoxide_sweep_h5.jl` use.

**Doublet geometry.** For `shape_kind = "doublet"` the runner meshes two
spheres of radius `R = a_eq / 2^(1/3)` with a centre distance of `2.1 R`
(surface-to-surface gap `0.1 R`), the geometry of the MSTM reference
files `mstm_doublet_*.hdf5`. From v0.8.2 `mesh_sphere_aggregate` scales
each sphere to its nominal volume about its own centre, so the centre
distance does not depend on the mesh size.

Earlier versions scaled the whole mesh uniformly. Up to v0.8.0 the
runner also used the default 2 % inflation of the spheres, and the centre
distance was `2.06 R` to `2.08 R` (gap `0.0595 R` between the faceted
surfaces). With v0.8.1 it was `2.10 R` to `2.13 R`.

The files `viem_results/paper/doublet_*.hdf5` were regenerated with
v0.8.2 on 2026-11-19 (`supermicro-647`). They carry the attribute
`doublet_gap_over_R = 0.1` on `/target` and the corrected `S_bk`
definition. The files of the v0.7.8 production run are in the git history
(commit `416cece`). The cost fields of the regenerated files come from a
different machine than those of the other paper files and are not
comparable with them.

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

Results of the re-run with v0.8.2 (2026-11-19), mesh size
$\ell_c = R/5$ (11 501 DOFs), AIM block GMRES with right
preconditioning by the mass matrix, every solve converged to
$10^{-7}$:

| Material | $\|S_\text{fw,mean}\|$ error (β = 0, π/4, π/2) | Largest component error |
| -------- | ------------------------------------------- | ----------------------- |
| dielectric, $m_p = 1.60 + 0.01i$ | 0.05 %, 0.09 %, 0.13 % | 0.48 % (Im $S_{\text{fw},\theta}$, β = π/2) |
| Au, $m_p = 0.17525 + 3.4830i$ | 0.004 %, 0.69 %, 1.09 % | 2.34 % (Im $S_{\text{fw},\theta}$, β = π/2) |

The Au error is largest when the field is polarised along the doublet
axis. With $\ell_c = R/6$ to $R/8$ every component of the Au doublet
is within 0.41 % of MSTM at all three orientations. The errors at these
levels do not decrease monotonically, so no convergence rate is quoted.

The earlier results (April 2026, git history, commit `d1482fa`) had
errors up to 10.5 % for Au. They were computed before v0.7.7 on meshes
that were not scaled to the nominal volume. The volume of the faceted
spheres was 1.4 % too small at $\ell_c = R/5$. Solving that mesh with
the present solver reproduces the earlier errors, so the improvement
comes from the volume rescale. Block BiCGSTAB, the solver of the
earlier runs, does not converge for the Au doublet (residual
$8 \times 10^{-3}$ after 400 iterations), which changes the amplitudes
by up to 0.5 %.

**Note on MSTMforCAS truncation order.** The MSTMforCAS
auto-truncation returns $N_\text{trunc} = 3$ at $x \approx 0.3$,
which is converged for the dielectric case but under-truncates the
plasmonic Au case (roughly 5 % shift in $|S_\text{fw,mean}|$
between $N = 3$ and the converged reference).  We therefore force
$N_\text{trunc} = 15$ in `run_mstm.jl` for both materials, reaching
rtol $\le 10^{-6}$ — see `check_trunc_convergence.jl`.

Full numerical conditions, per-orientation Re/Im-resolved tables,
the $\ell_c$-refinement study, and the Phase A AIM-memory scaling are
documented in
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
- **Backward amplitude:** `S_bk` (`S_bk_OCBS` in the HDF5 files, *S*<sub>bak</sub>
  of CAS-v2) is `(−S11 + S22)/2` at the backward direction. This is the
  corrected Eq. (28) of Moteki and Adachi (2024), given in the erratum to
  that paper, and the definition of block-DDA_Py from v0.8.2. Up to
  v0.7.9 block-VIEM.jl followed the printed equation, which has `1/√2`
  in place of `1/2`. Values of `S_bk` from v0.7.9 or earlier are `√2`
  times the present ones, with the same phase. The forward amplitudes and
  the cross sections are not affected. All files under
  `viem_results/paper/` hold the corrected form. `doublet_*.hdf5` were
  regenerated with v0.8.2. In the other files (VIEM, Mie, MSTM and
  T-matrix values) the stored backward amplitudes were multiplied by
  `1/√2` on 2026-11-19, without recomputation, and the attributes
  `S_definition` or `S_bk_definition` and `S_bk_conversion_note` say so.
  The files as produced by v0.7.8 are in the git history (commit
  `416cece`).
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
- `CHANGELOG.md` — changes by version, and the corrections that change
  numerical results
- Primary literature: Schaubert, Wilton and Glisson (1984); Volakis and
  Sertel, *Integral Equation Methods for Electromagnetics*; Sheng and
  Song, *Essentials of Computational Electromagnetics*; Mousavi and
  Sukumar (2010). Full references are in `docs/theory_note.pdf`.

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
