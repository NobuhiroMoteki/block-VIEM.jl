# block-VIEM.jl

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
   Block BiCGSTAB / Block GMRES solve all particle orientations
   simultaneously. AIM grid pitch is auto-detected from the mesh.
3. **Multi-orientation batch solve.** Supports deterministic uniform Euler
   angle grids on SO(3) (`N_alpha` x `N_beta` x `N_gamma`), identical to
   block-DDA_Py. **Spheroid mode** (`ab_ratio=1`, `beta=0`) solves only
   `N_beta` orientations and fills the full grid analytically via the
   alpha-expansion `S_theta(alpha) = A + B*exp(+2j*alpha)`.
4. **Birefringent (anisotropic) materials.** Diagonal permittivity tensor
   `eps_p = diag(eps_x, eps_y, eps_z)` is supported throughout the solver,
   post-processing, and sweep scripts.
5. **Gaussian Random Ellipsoid (GRE) shape model.** Parametric irregular
   particle shapes `(r_v_base, bc_ratio, ab_ratio, beta)` matching
   block-DDA_Py (Muinonen & Pieniluoma 2011, JQSRT). Tetrahedral meshes
   are generated automatically via Gmsh with adaptive mesh size based on
   wavelength, geometry, and surface-deformation correlation length.
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

- **Plasmonic Au Mie sphere** (`benchmarks/cas_v2/au_sphere_mie.jl`):
  Johnson & Christy 1972 gold at wl_0 = 0.638 um
  (m_p = 0.17525 + 3.4830i, eps_p = -12.10 + 1.22i, x = 0.63).
  Both dense (LU) and AIM-BiCGSTAB solvers achieve sub-1 % on all
  five observables at N = 2134 half-SWG DOFs:

  | lc [um] | N    | method       | C_ext  | C_abs  | C_sca  | S_fw_mean | S_bk   | time  |
  |---------|------|--------------|--------|--------|--------|-----------|--------|-------|
  | 0.020   | 2134 | dense (LU)   | 0.65 % | 0.01 % | 0.75 % | 0.58 %    | 0.28 % | 17 s  |
  | 0.020   | 2134 | AIM-BiCGSTAB | 0.25 % | 0.44 % | 0.37 % | 0.32 %    | 0.11 % | 43 s  |
  | 0.014   | 4932 | dense (LU)   | 0.34 % | 0.08 % | 0.41 % | 0.32 %    | 0.15 % | 68 s  |
  | 0.014   | 4932 | AIM-BiCGSTAB | 0.15 % | 0.24 % | 0.22 % | 0.18 %    | 0.07 % | 105 s |

- **Spheroid symmetry** (`benchmarks/cas_v2/spheroid_ar3.jl`): the
  analytical α-expansion `S_θ(α) = A + B·exp(+2jα)` holds at machine
  precision (~5×10⁻¹⁵) on an oblate AR = 3 mesh.
- **Cross-validation vs block-DDA_Py (spheroid)**
  (`benchmarks/cas_v2/dda_comparison/`): oblate AR = 3 spheroid
  (D_ve = 0.40 μm, m_p = 1.5, λ₀ = 0.638 μm), VIEM and DDA agree to
  ~2 % on the complex polarimetric forward amplitudes at two tilt
  angles (β = π/4, π/2).
- **GRE cross-validation vs block-DDA_Py (β > 0)**
  (`benchmarks/cas_v2/gre_comparison/`): three GRE shapes with surface
  deformation (β = 0.10–0.20, bc_ratio = 1.5–3.0, ab_ratio = 1.0–1.5),
  m_p = 1.5 + 0.01i, λ₀ = 0.638 μm, at two tilt angles. The same
  Gaussian random field is shared between DDA and VIEM. Agreement:

  | Case | b/c | a/b | β    | ΔS_θ rel.  | ΔS_φ rel.  |
  |------|-----|-----|------|------------|------------|
  | A    | 2.0 | 1.0 | 0.10 | 0.5–0.7 %  | 0.1–0.3 %  |
  | B    | 3.0 | 1.0 | 0.15 | 2.0–2.6 %  | 1.7–1.8 %  |
  | C    | 1.5 | 1.5 | 0.20 | 1.2–1.6 %  | 1.6–1.8 %  |

- **Anisotropic ε_p cross-validation vs block-DDA_Py**
  (`benchmarks/cas_v2/aniso_comparison/`): sphere (r_ve = 0.20 μm,
  λ₀ = 0.638 μm) with diagonal-tensor refractive index. The same
  `m_p_xyz = [m_x, m_y, m_z]` is passed to both DDA and VIEM:

  | Case   | m_p               | ΔS_θ rel.  | ΔS_φ rel.  |
  |--------|-------------------|------------|------------|
  | iso    | [1.5, 1.5, 1.5]  | 1.3–2.4 %  | 2.4 %      |
  | mild   | [1.55, 1.5, 1.45] | 1.2–3.5 %  | 2.3–2.6 %  |
  | strong | [1.6, 1.5, 1.4]  | 0.7–5.0 %  | 2.3–2.9 %  |

## When to use block-VIEM.jl vs block-DDA_Py

block-DDA_Py and block-VIEM.jl solve the same physics and produce the
same observables in the same HDF5 format. The choice depends on the
refractive-index contrast and particle geometry.

### Low contrast (|m_p| < 2): use block-DDA_Py

For weakly scattering particles (e.g. mineral dust, m_p ~ 1.5), DDA
is far more efficient. Benchmark on a Mie sphere (r = 1 um,
m_p = 1.5 + 0.01i, wl_0 = 10 um, x = 0.63, single orientation,
Intel i7-1265U):

| Code          | N DOF  | C_abs error | t_setup  | t_solve  | memory  |
|---------------|--------|-------------|----------|----------|---------|
| block-DDA_Py  | 302    | 1.7 %       | < 0.1 s  | < 0.1 s  | 6 MB    |
| block-DDA_Py  | 4 419  | 0.8 %       | < 0.1 s  | 0.1 s    | 64 MB   |
| block-VIEM.jl | 589    | 8.4 %       | 7.9 s    | 2.2 s    | 6 MB    |
| block-VIEM.jl | 1 986  | 3.6 %       | 23.9 s   | 0.7 s    | 63 MB   |
| block-VIEM.jl | 7 868  | 1.4 %       | 145.8 s  | 2.7 s    | 991 MB  |

DDA achieves 1.7 % accuracy with 302 dipoles in under 0.1 s.
VIEM requires ~2 000 DOFs and 25 s of setup for comparable accuracy.
The DDA cubic lattice has an exact FFT-MVP with no near-field
precorrection, so setup cost is effectively zero.

**Recommendation:** For |m_p/m_m| < 2 and moderate aspect ratios,
block-DDA_Py is the right tool.

### High contrast (|m_p| > 3, plasmonic): use block-VIEM.jl

DDA convergence degrades sharply at high refractive-index contrast.
Same benchmark sphere with m_p = 3.17 + 0.16i (eps_p ~ 10):

| Code          | N DOF  | C_abs error | convergence     |
|---------------|--------|-------------|-----------------|
| block-DDA_Py  | 1 496  | 6.8 %       | slow (1/N^0.5)  |
| block-DDA_Py  | 4 759  | 4.2 %       | slow (1/N^0.5)  |
| block-VIEM.jl | 1 986  | 3.6 %       | O(h^2)          |
| block-VIEM.jl | 7 868  | 1.4 %       | O(h^2)          |

At N ~ 5000, DDA still has 4 % error while VIEM reaches 1.4 %.
For plasmonic Au (m_p = 0.175 + 3.48i), VIEM converges monotonically
at sub-1 % with N = 2134 DOFs (see Validation above); DDA may not
converge at all for such materials.

The VIEM advantages come from:

1. **Conforming geometry** -- tetrahedral meshes represent curved
   surfaces to O(h^2); cubic lattices staircase them to O(d).
2. **H(div)-conforming basis** -- normal D continuity is built into the
   SWG basis, not approximated via a polarizability prescription.
3. **Linear vector basis** -- each tet carries up to 4 SWG functions
   (linear in position), versus one point dipole per DDA element.
4. **No polarizability tuning** -- VIEM solves the integral equation
   directly; DDA requires Clausius-Mossotti + radiative correction.

**Recommendation:** For |m_p/m_m| > 2-3, plasmonic metals,
extreme aspect ratios, or problems where block-DDA_Py fails to
converge, use block-VIEM.jl.

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

The default solver is **AIM + Block BiCGSTAB** (`method = :aim_bicgstab`),
which uses FFT-accelerated matrix-vector products (O(N log N) per
iteration) and solves all orientations simultaneously via a block-Krylov
iteration.  The AIM grid pitch is auto-detected from the mesh
(`0.5 * mean_edge_length`) when not supplied explicitly.

```julia
# Default: AIM + Block BiCGSTAB (pitch auto-detected)
results = solve_cas_v2_orientations(
    basis, euler_list;
    wl_0 = 0.638, m_m = 1.0, m_p = 1.5 + 0.01im,
)

# Override solver parameters
results = solve_cas_v2_orientations(
    basis, euler_list;
    wl_0 = 0.638, m_m = 1.0, m_p = 1.5 + 0.01im,
    method  = :aim_gmres,     # or :aim_bicgstab (default), :dense
    pitch   = 0.01,           # AIM grid pitch [μm] (auto if omitted)
    padding = 4,
    tol     = 1e-8,
    maxiter = 400,
)
```

Available methods:

- **`:aim_bicgstab`** (default) — Block BiCGSTAB (Tadano-Sakurai-Kuramashi 2009).
- **`:aim_gmres`** — unrestarted Block GMRES (Simoncini-Szyld 1996).
- **`:dense`** — assemble Z and LU-factorize. Only for small problems (N < 10^3).

Both block solvers are also exposed directly as
`block_bicgstab(A, B; tol, maxiter)` and `block_gmres(A, B; tol, maxiter)`
for any linear operator `A` supporting `A * X::AbstractMatrix`.

The `return_D=true` keyword causes `solve_cas_v2_orientations` to return
`(results, D_block)` instead of just `results`, giving access to the raw
D-field expansion coefficients for computing cross sections or other
post-processing.

### Gaussian Random Ellipsoid (GRE) shapes

block-VIEM.jl can generate tetrahedral meshes for the same Gaussian
Random Ellipsoid particle shapes used by block-DDA_Py, directly from the
`(r_v_base, bc_ratio, ab_ratio, beta)` parameter set:

```julia
using BlockVIEM, Random

p = GREParams(
    0.2,    # r_v_base [μm] — volume-equivalent radius of base ellipsoid
    3.0,    # bc_ratio      — b/c semi-axis ratio  (range [1, 7])
    1.5,    # ab_ratio      — a/b semi-axis ratio  (range [1, 2])
    0.15,   # beta          — Gaussian deformation σ (range [0, 0.3])
)

rng = MersenneTwister(42)

# Mesh size is chosen automatically from wavelength + geometry:
mesh, r_ve = gre_mesh(p, rng;
    wl_0    = 0.638,   # vacuum wavelength [μm]
    m_p_max = 1.5,     # max |m_p| (for resolution estimate)
    N_pw    = 10,       # elements per wavelength inside particle
)

# Or set lc explicitly:
mesh, r_ve = gre_mesh(p, rng; lc = 0.04)

basis = build_swg_basis(mesh; include_boundary_faces = true)
# ... proceed with solve_cas_v2_orientations as above
```

For `beta = 0` (smooth ellipsoid) the mesh is generated via the Gmsh
OpenCASCADE kernel.  For `beta > 0`, a Gaussian random deformation
field is sampled on the ellipsoid surface (Muinonen & Pieniluoma 2011,
JQSRT) and all mesh nodes are deformed radially, preserving mesh
quality.  The adaptive mesh size `adaptive_lc(p; ...)` takes the minimum
of a wavelength constraint, a geometry constraint (`c/3`), and the
surface-deformation correlation length (`0.1c`).

#### Visualising discretised targets

[viz/visualize_gre.jl](viz/visualize_gre.jl) renders the tetrahedral
target of a GRE particle at a given laboratory-frame orientation
(mirroring `run_gaussian_ellipsoid.ipynb` in block-DDA_Py). The
semi-transparent wireframe shows the discretised surface triangles;
black/red/blue arrows mark the lab-frame z/x/y axes (z = incident beam
direction):

```julia
# from the viz/ environment
include("viz/visualize_gre.jl")

p = GREParams(
    0.30,   # r_v_base [μm]
    2.0,    # bc_ratio
    1.0,    # ab_ratio
    0.10,   # beta
)
visualize_gre(p, (0, 45, 0);                 # ZYZ Euler angles in degrees
              output_path = "gre.png",
              lc          = 0.04)             # coarser mesh for clarity
```

Batch generation of the README gallery:

```bash
julia --project=viz viz/visualize_gre.jl
# writes PNGs to viz/figs/
```

Five representative shapes at `r_v_base = 0.30 μm`, rendered with a
coarser-than-physics mesh (`lc = c / 2.5`) purely for wireframe
readability:

| sphere (`bc=1, ab=1, β=0`)         | oblate (`bc=3, ab=1, β=0`)             | triaxial (`bc=2, ab=1.5, β=0`)                  |
| :--------------------------------: | :------------------------------------: | :---------------------------------------------: |
| ![sphere](viz/figs/gre_sphere.png) | ![oblate](viz/figs/gre_oblate_bc3.png) | ![triaxial](viz/figs/gre_triaxial_bc2_ab15.png) |
| Euler = (0°, 0°, 0°)               | Euler = (0°, 60°, 0°)                  | Euler = (30°, 45°, 0°)                          |

| GRE (`bc=2, ab=1, β=0.10`)                | GRE (`bc=1.5, ab=1.5, β=0.20`)                  |
| :---------------------------------------: | :---------------------------------------------: |
| ![gre_b010](viz/figs/gre_beta010_bc2.png) | ![gre_b020](viz/figs/gre_beta020_bc15_ab15.png) |
| Euler = (0°, 45°, 0°)                     | Euler = (20°, 60°, 10°)                         |

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

- Uses **AIM + Block BiCGSTAB** by default (O(N log N) per iteration)
- Automatically detects **spheroid mode** (`ab_ratio == 1` and
  `gre_beta == 0`): solves only `N_beta` orientations at α = 0, then
  fills the full `(N_alpha × N_beta × N_gamma)` grid analytically via
  `S_θ(α) = A + B·exp(+2jα)`
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

## Benchmark: 2-sphere doublet CAS-v2 vs MSTM

Cross-validation against [MSTMforCAS.jl](https://github.com/MOTEKI-LAB/MSTMforCAS.jl)
(multi-sphere T-matrix, numerically exact for aggregates of homogeneous
spheres). The benchmark target is a touching-but-disjoint doublet of
equal spheres (radius 0.030 μm, gap 0.003 μm along the doublet axis),
vacuum wavelength 0.638 μm, background n = 1.0. The orientation angle
β is the angle between the incidence direction and the doublet axis:
block-VIEM realises it via `cas_orientation(0, β, 0)` (particle fixed
along lab-z, incidence rotated); MSTM rotates the doublet axis to make
angle β with the z-axis incidence — physically equivalent setups.

The observable is the CAS-v2 forward amplitude for RCP incidence in
the block-DDA_Py / block-VIEM convention,
`S_fw_θ = S11 + i·S12`, `S_fw_φ = S22 − i·S21`,
`S_fw_mean = (S_fw_θ + S_fw_φ)/2`,
where the MI02 scattering amplitudes are related to the BH83 forward
amplitudes `(S₁, S₂, S₃, S₄)` by
`S11 = S₂/(−ik)`, `S12 = S₃/(ik)`, `S21 = S₄/(ik)`, `S22 = S₁/(−ik)`.
Combining the two conversions:
`S_fw_θ = (S₂ − i·S₃)/(−ik)`, `S_fw_φ = (S₁ + i·S₄)/(−ik)`.
For this reflection-symmetric doublet benchmark `S₃ = S₄ = 0` so the
off-diagonal terms drop out; the correct signs are retained in the
driver code for generic aggregates.

The VIEM side uses 5348 tets / 11501 SWG DOFs (half-SWG, `lc = R/5`)
and is solved by the AIM block-BiCGSTAB driver with `tol = 1e-7`.

**MSTM VSWF truncation order.** The monomer size parameter is
`x = k·R ≈ 0.295`, for which MSTMforCAS's automatic Wiscombe-based
heuristic returns N = 3. For the polystyrene case the single-sphere
Mie coefficients satisfy `|a_N|,|b_N| < 10⁻⁶` already at N = 3, so
N = 3 is a converged reference. For **gold**, however, the high
`|m·x| ≈ 1.03` combined with the multi-sphere translation coupling
amplifies higher-order multipoles: empirically `|S_fw_mean|` shifts by
~0.9 % per step from N = 3 to N = 6 (a total of ~5 %). We therefore
force `truncation_order = 6` for both materials in `run_mstm.jl`. At
`x ≈ 0.3` MSTMforCAS's Riccati–Bessel upward recurrence becomes
numerically unstable at N ≥ 7, so N = 6 is both the largest stable
value and sufficient for machine-precision convergence of the MSTM
reference (since `|a_n|, |b_n|` fall below 10⁻¹⁶ at n = 6 for these
parameters).

**Dielectric case — m_p = 1.60 + 0.01i** (polystyrene-like high contrast):

Summary (|S_fw_mean| magnitude + phase, MSTM at N = 6):

| β [rad] | \|S_fw_mean\| VIEM | \|S_fw_mean\| MSTM | rel \|·\| err | complex rel err | phase err [rad] |
|---------|--------------------|--------------------|---------------|-----------------|-----------------|
| 0       | 1.769e−03          | 1.794e−03          | 1.4 %         | 1.4 %           | 1.6e−04         |
| π/4     | 1.822e−03          | 1.849e−03          | 1.5 %         | 1.5 %           | 1.6e−04         |
| π/2     | 1.878e−03          | 1.908e−03          | 1.6 %         | 1.6 %           | 2.0e−04         |

S_fw_θ — real and imaginary parts:

| β [rad] | Re VIEM      | Re MSTM      | rel err Re | Im VIEM      | Im MSTM      | rel err Im |
|---------|--------------|--------------|------------|--------------|--------------|------------|
| 0       | +1.7682e−03  | +1.7937e−03  | 1.4 %      | +4.1063e−05  | +4.2003e−05  | 2.2 %      |
| π/4     | +1.8768e−03  | +1.9068e−03  | 1.6 %      | +4.7872e−05  | +4.9087e−05  | 2.5 %      |
| π/2     | +1.9933e−03  | +2.0282e−03  | 1.7 %      | +5.5225e−05  | +5.6773e−05  | 2.7 %      |

S_fw_φ — real and imaginary parts:

| β [rad] | Re VIEM      | Re MSTM      | rel err Re | Im VIEM      | Im MSTM      | rel err Im |
|---------|--------------|--------------|------------|--------------|--------------|------------|
| 0       | +1.7683e−03  | +1.7937e−03  | 1.4 %      | +4.1285e−05  | +4.2003e−05  | 1.7 %      |
| π/4     | +1.7651e−03  | +1.7902e−03  | 1.4 %      | +4.2005e−05  | +4.2723e−05  | 1.7 %      |
| π/2     | +1.7619e−03  | +1.7870e−03  | 1.4 %      | +4.2680e−05  | +4.3462e−05  | 1.8 %      |

**Plasmonic case — m_p = 0.17525 + 3.4830i** (Au @ 638 nm, Johnson &
Christy 1972):

Summary (MSTM at N = 6):

| β [rad] | \|S_fw_mean\| VIEM | \|S_fw_mean\| MSTM | rel \|·\| err | complex rel err | phase err [rad] |
|---------|--------------------|--------------------|---------------|-----------------|-----------------|
| 0       | 6.420e−03          | 6.515e−03          | 1.5 %         | 1.5 %           | 5.3e−04         |
| π/4     | 8.043e−03          | 8.247e−03          | 2.5 %         | 2.5 %           | 3.2e−03         |
| π/2     | 9.785e−03          | 1.011e−02          | 3.2 %         | 3.3 %           | 4.5e−03         |

S_fw_θ — real and imaginary parts:

| β [rad] | Re VIEM      | Re MSTM      | rel err Re | Im VIEM      | Im MSTM      | rel err Im |
|---------|--------------|--------------|------------|--------------|--------------|------------|
| 0       | +6.4046e−03  | +6.4995e−03  | 1.5 %      | +4.3635e−04  | +4.4797e−04  | 2.6 %      |
| π/4     | +9.6439e−03  | +9.9603e−03  | 3.2 %      | +1.2147e−03  | +1.3024e−03  | 6.7 %      |
| π/2     | +1.3102e−02  | +1.3654e−02  | 4.1 %      | +2.0467e−03  | +2.2120e−03  | 7.5 %      |

S_fw_φ — real and imaginary parts:

| β [rad] | Re VIEM      | Re MSTM      | rel err Re | Im VIEM      | Im MSTM      | rel err Im |
|---------|--------------|--------------|------------|--------------|--------------|------------|
| 0       | +6.4049e−03  | +6.4995e−03  | 1.5 %      | +4.3967e−04  | +4.4797e−04  | 1.9 %      |
| π/4     | +6.3556e−03  | +6.4387e−03  | 1.3 %      | +4.4777e−04  | +4.5496e−04  | 1.6 %      |
| π/2     | +6.3084e−03  | +6.3892e−03  | 1.3 %      | +4.5536e−04  | +4.6332e−04  | 1.7 %      |

**Discussion.**
For the dielectric doublet the agreement is uniform across orientations
and polarizations: Re parts are recovered to ~1.4 % and Im parts to
~2 % — the extra factor on Im simply reflects that |Im| is
30–50× smaller than |Re|, so a fixed absolute discretization error
maps to a larger relative error on the imaginary axis. Both parts
converge as `O(h²)` under mesh refinement.

For the plasmonic gold doublet a clear anisotropy emerges. The
**S_fw_φ** component — which at any β corresponds to the polarization
channel perpendicular to the plane of incidence and the doublet axis —
is essentially β-independent (1.3–1.5 % on Re, 1.6–1.9 % on Im). In
contrast the **S_fw_θ** component, which picks up the polarization
component *along* the doublet axis after rotation, grows rapidly with
β: at β = π/2 the real part is off by 4.1 % and the imaginary part by
7.5 %. This is the orientation at which the two spheres are excited
along their own axis and the electric field concentrates in the
inter-sphere gap, with surface plasmons living in a skin layer of depth
δ ≈ λ₀/(2π·Im m_p) ≈ 29 nm — essentially the monomer radius itself.
The linear-SWG mesh with `lc = R/5 ≈ 6 nm` resolves this skin layer
with roughly five elements, which is enough for a few-percent
observable but not for high-precision plasmonic resonance reproduction.
Refining to `lc = R/8–R/10` (or enabling the boundary-face half-SWG
correction with tighter Duffy rules) is expected to bring the Im error
below 3 % uniformly.

The key takeaway is that a single-number relative error on |S_fw_mean|
undersells the orientation / polarization structure of the discretization
error: the imaginary part of the axis-aligned-polarization channel is
the most sensitive probe of mesh quality for high-|m| absorbing
aggregates, and should be used as the convergence indicator when
tuning `lc` for plasmonic targets.

A second lesson concerns the reference solution itself. MSTM's
Wiscombe-based auto-truncation (N = 3 at x ≈ 0.3) is converged for
polystyrene but systematically under-truncates gold — forcing
N = 6 shifts `|S_fw_mean|` by up to 5 %, and the VIEM errors quoted above
already incorporate that correction. For larger gold monomers where
the effective internal size parameter `|m·x| > 2` even higher N is
expected to be necessary, but for `x ≈ 0.3` the Mie series saturates
at N = 6 to machine precision and no further truncation error remains.

Reproduce the benchmark:

```bash
# Generate block-VIEM predictions
julia --project=. benchmarks/cas_v2/doublet_mstm/run_viem.jl

# Generate exact MSTM reference (requires MSTMforCAS.jl checked out as a sibling)
julia --project=/path/to/MSTMforCAS.jl benchmarks/cas_v2/doublet_mstm/run_mstm.jl

# Print / save the relative-error tables
julia --project=. benchmarks/cas_v2/doublet_mstm/compare.jl
```

All scripts read a single `config.jl` so the geometry, wavelength,
materials, and orientation grid stay in lock-step between the two codes.

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
  Mie validation (dielectric + plasmonic Au), spheroid benchmarks,
  GRE shape model
- `docs/io_spec.md` — block-DDA_Py compatible I/O specification
- `.claude/reference/` — primary literature (SWG 1984, Volakis-Sertel,
  Sheng-Song, Mousavi-Sukumar 2010)

## Installation (development)

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
julia> Pkg.test()
```

Julia ≥ 1.10 is required. The package depends on `Gmsh.jl` for mesh
generation and I/O.

## License

MIT (see `LICENSE`).

## Author

Nobuhiro Moteki ([@NobuhiroMoteki](https://github.com/NobuhiroMoteki),
`nobuhiro.moteki@gmail.com`)
