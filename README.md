# BlockVIEM.jl

A Julia implementation of the **Volume Integral Equation Method (VIEM)** for
electromagnetic scattering by arbitrarily shaped, high-contrast dielectric
particles (e.g., iron-oxide aggregates, rough gold nanoparticles).

The package targets the same observables as
[`block-DDA_Py`](https://github.com/NobuhiroMoteki/block-DDA_Py)
(CAS-v2 complex scattering amplitudes) with higher accuracy per
degree-of-freedom via tetrahedral discretization (SWG basis),
Duffy-transform singular integration, and AIM/FFT-accelerated matrix-vector
products.

## Status

**v0.2.0** — 1763 tests pass. All numerical kernels are in place.

| Phase | Module                                          | Status |
|-------|-------------------------------------------------|--------|
| 1     | Mesh & SWG / half-SWG / RT1 bases               | done   |
| 2     | Duffy-transform singular integration            | done   |
| 3     | AIM (FFT-MVP) with precorrection                | done   |
| 4     | Direct and iterative (Krylov) solvers            | done   |
| 5.1   | CAS-v2 forward / backward scattering observables | done   |
| 5.2   | Multi-orientation batch solve (shared LU)        | done   |
| 5.3   | Oblate spheroid benchmark vs block-DDA_Py        | done   |
| 5.4   | HDF5 spheroid-sweep output (block-DDA_Py schema) | done   |
| 5.5   | AIM + Block-Krylov multi-orientation solve        | done   |

### Validation

- **Dielectric Mie sphere** (`test/test_mie_validation.jl`,
  `test/test_cas_v2.jl`): `C_ext`, `C_abs`, `C_sca`, `S_fw`, `S_bk`
  match Mie reference to ~0.4 % at `lc = 0.30` (N ≈ 400 DOFs) and
  converge with mesh refinement. Weakly absorbing particle
  (m_p = 1.5 + 0.01i).
- **Plasmonic Au Mie sphere** (`benchmarks/cas_v2/au_sphere_mie.jl`):
  Johnson–Christy gold (m_p = 0.18 + 3.07i, ε_p = −9.39 + 1.10i) at
  x = 0.63. All five observables (C_ext, C_abs, C_sca, S_fw, S_bk)
  converge monotonically and are sub-1 % at N = 2134 half-SWG DOFs:

  | lc [μm] | N    | C_ext  | C_abs  | C_sca  | S_fw   | S_bk   |
  |---------|------|--------|--------|--------|--------|--------|
  | 0.020   | 2134 | 0.62 % | 0.44 % | 0.66 % | 0.49 % | 0.26 % |
  | 0.014   | 4932 | 0.32 % | 0.19 % | 0.35 % | 0.26 % | 0.14 % |

- **Half-SWG accuracy** (`benchmarks/rt0/v8b_half_swg_convergence.jl`):
  sphere C_abs error ≈ 0.21 % at `lc = 0.50` (589 DOFs), outperforming
  DDA by 10–12× per DOF.
- **Spheroid symmetry** (`benchmarks/cas_v2/spheroid_ar3.jl`): the
  analytical α-expansion `S_θ(α) = A + B·exp(+2jα)` holds at machine
  precision (~5×10⁻¹⁵) on an oblate AR = 3 mesh.
- **Cross-validation vs block-DDA_Py**
  (`benchmarks/cas_v2/dda_comparison/`): oblate AR = 3 spheroid
  (D_ve = 0.40 μm, m_p = 1.5, λ₀ = 0.638 μm), VIEM and DDA agree to
  ~2 % on the complex polarimetric forward amplitudes at two tilt
  angles (β = π/4, π/2).

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

Alternatively, pass the raw VIEM form `(k0, eps_p, eps_bg)` where
`k0` is the background-medium wavenumber (`2π·m_m/λ₀`, **not** the
vacuum wavenumber) and `eps_p`, `eps_bg` are absolute permittivities.
See [`solve_cas_v2_orientations`](src/postprocess.jl) docstring.

### AIM + Block Krylov (large problems)

For larger problems where dense `Z` no longer fits, use the AIM-
accelerated block-Krylov path:

```julia
results = solve_cas_v2_orientations(
    basis, euler_list;
    wl_0 = 0.638, m_m = 1.0, m_p = 1.5 + 0.01im,
    method = :aim_bicgstab,    # or :aim_gmres  (or :dense — default)
    pitch   = 0.5 * mean_edge_length,
    padding = 4,
    tol     = 1e-8,
    maxiter = 400,
)
```

- **`:aim_bicgstab`** — Block BiCGSTAB (Tadano–Sakurai–Kuramashi 2009).
- **`:aim_gmres`** — unrestarted Block GMRES (Simoncini–Szyld 1996).

Both are also exposed directly as `block_bicgstab(A, B; tol, maxiter)`
and `block_gmres(A, B; tol, maxiter)` for any linear operator `A`
supporting `A * X::AbstractMatrix`.

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
