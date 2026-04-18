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

  | lc [um] | N    | method       | C_ext  | C_abs  | C_sca  | S_fw   | S_bk   | time  |
  |---------|------|--------------|--------|--------|--------|--------|--------|-------|
  | 0.020   | 2134 | dense (LU)   | 0.65 % | 0.01 % | 0.75 % | 0.58 % | 0.28 % | 17 s  |
  | 0.020   | 2134 | AIM-BiCGSTAB | 0.25 % | 0.44 % | 0.37 % | 0.32 % | 0.11 % | 43 s  |
  | 0.014   | 4932 | dense (LU)   | 0.34 % | 0.08 % | 0.41 % | 0.32 % | 0.15 % | 68 s  |
  | 0.014   | 4932 | AIM-BiCGSTAB | 0.15 % | 0.24 % | 0.22 % | 0.18 % | 0.07 % | 105 s |

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
# results[i].S_fw, results[i].S_bk, results[i].S_fw_theta, ...
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
