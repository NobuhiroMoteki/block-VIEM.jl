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

**v0.2.0 — Phase 5.5 (AIM-accelerated block-Krylov multi-orientation
solve) complete.** 1746 tests pass. All numerical kernels are in place;
multi-orientation solves and oblate spheroid benchmarks match
block-DDA_Py and the Mie reference, and large problems can now be
solved for many incident directions at once via Block BiCGSTAB or
Block GMRES on top of the AIM FFT MVP.

| Phase | Module                                          | Status |
|-------|-------------------------------------------------|--------|
| 1     | Mesh & SWG / half-SWG / RT1 bases               | done   |
| 2     | Duffy-transform singular integration            | done   |
| 3     | AIM (FFT-MVP) with precorrection                 | done   |
| 4     | Direct and iterative (Krylov) solvers            | done   |
| 5.1   | CAS-v2 forward / backward scattering observables | done   |
| 5.2   | Multi-orientation batch solve (shared LU)        | done   |
| 5.3   | Oblate spheroid benchmark vs block-DDA_Py        | done   |
| 5.4   | HDF5 spheroid-sweep output (block-DDA_Py schema) | done   |
| 5.5   | **AIM + Block-Krylov multi-orientation solve**   | done   |

### Validation

- **Mie sphere** (`test/test_mie_validation.jl`, `test/test_cas_v2.jl`):
  `C_abs`, `C_sca`, `C_ext`, and both Re and Im of `S_fw` match Mie reference
  to ~0.4 % at `lc = 0.30` (N ≈ 400 DOFs) and converge with mesh refinement.
- **Half-SWG accuracy** (`benchmarks/rt0/v8b_half_swg_convergence.jl`):
  sphere `C_abs` error ≈ 0.21 % at `lc = 0.50` (589 DOFs), outperforming
  DDA by 10-12× per DOF.
- **Spheroid symmetry** (`benchmarks/cas_v2/spheroid_ar3.jl`): the
  analytical α-expansion `S_θ(α) = A + B·exp(+2jα)` used by block-DDA_Py
  holds at machine precision (~5×10⁻¹⁵) on an oblate AR = 3 mesh.
- **Direct cross-validation vs block-DDA_Py**
  (`benchmarks/cas_v2/dda_comparison/`): on an oblate AR = 3 spheroid
  (D_ve = 0.40 μm, m_p = 1.5, λ₀ = 0.638 μm) at two tilt angles, VIEM
  (lc = 0.035 μm, 8746 half-SWG DoFs) and DDA (dpl = 17, 2236 dipoles)
  agree to ~2 % on the complex polarimetric forward amplitudes:

  | β    | \|ΔS_θ\|/\|S_θ\| | \|ΔS_φ\|/\|S_φ\| | \|ΔC_ext\|/C_ext |
  |------|------------------|------------------|-----------------|
  | π/4  | 1.67 %           | 2.46 %           | 2.96 %          |
  | π/2  | 2.08 %           | 2.75 %           | 3.34 %          |

  Both codes are discretisation-limited at these mesh densities (the
  VIEM mesh r_ve is itself 0.41 % short of the target), so a few-percent
  disagreement between the two independent codes is the expected level.

## Quick start

```julia
using BlockVIEM, StaticArrays
using BlockVIEM: Vec3

# 1. Load a tetrahedral mesh produced by Gmsh (.msh format)
mesh  = read_msh("sphere.msh")
basis = build_swg_basis(mesh; include_boundary_faces = true)   # half-SWG

# 2. Assemble impedance matrix and solve for D-field coefficients
k0     = 2π / 0.638                        # wavenumber in background [1/μm]
eps_p  = (1.5 + 0.01im)^2                   # particle permittivity
k_hat  = Vec3(0, 0, 1)                       # incident propagation direction
E0     = SVector{3,ComplexF64}(1, 0, 0)     # x-polarized

Z = assemble_impedance_matrix(basis; k0, eps_p, eps_bg = 1,
                              duffy_rule = duffy_reference_rule(7),
                              symmetrize = true)
b = project_plane_wave(basis; k_hat, E0, k_bg = ComplexF64(k0))
D = Z \ b

# 3. Cross sections and CAS-v2 observables
scat = compute_scattering(basis, D; k_hat, E0, k0, eps_p)
# scat.C_ext, scat.C_abs, scat.C_sca, scat.S_fw_s, ...

ori  = cas_orientation(0.0, 0.0, 0.0)        # ZYZ Euler angles
cas  = compute_cas_observables(basis, D; orientation = ori,
                               k0, eps_p, eps_bg = 1)
# cas.S_fw_theta, cas.S_fw_phi, cas.S_fw, cas.S_bk, ...
```

For multi-orientation sweeps (assemble Z once, reuse LU), see
[`solve_cas_v2_orientations`](src/postprocess.jl) and
[`benchmarks/cas_v2/spheroid_ar3.jl`](benchmarks/cas_v2/spheroid_ar3.jl).

### Multi-orientation via AIM + Block Krylov (large problems)

For larger problems where dense `Z` no longer fits, use the AIM-
accelerated block-Krylov path. All orientations are solved at once
against a single pre-built AIM operator, sharing FFT plans and the
sparse precorrection matrix across right-hand sides.

```julia
euler_list = [(0.0, 0.0, 0.0), (0.3, 0.7, 0.5), (0.0, π/2, 0.0), ...]

results = solve_cas_v2_orientations(
    basis, euler_list;
    k0, eps_p, eps_bg = 1,
    method = :aim_bicgstab,    # or :aim_gmres  (or :dense — default)
    pitch   = 0.5 * mean_edge_length,
    padding = 4,
    tol     = 1e-8,
    maxiter = 400,
)
```

The two block solvers follow
- **`:aim_bicgstab`** — Block BiCGSTAB, Tadano–Sakurai–Kuramashi 2009
  (port of block-DDA_Py's `bl_bicgstab_jacobi_mvp_fft`).
- **`:aim_gmres`**    — unrestarted Block GMRES, Simoncini–Szyld 1996.
  Slower per iteration (full block Krylov basis retained) but robust
  against BiCGSTAB breakdowns on rank-deficient RHS blocks.

Both are also exposed directly as `block_bicgstab(A, B; tol, maxiter)`
and `block_gmres(A, B; tol, maxiter)` for any linear operator `A`
supporting `A * X::AbstractMatrix`.

## Conventions

- **Time convention:** physics, `exp(−iωt)`, matching block-DDA_Py and
  the Mie reference. Outgoing scalar Helmholtz Green's function is
  `exp(+ik₀R)/(4πR)`; the far-field integrand carries `exp(−ik₀ r̂·r')`.
- **Units:** SI / Lorentz–Heaviside (the scalar Green's function has the
  `1/(4π)` factor). `compute_cas_observables` therefore divides `F` by
  `4π` to match the block-DDA_Py / Mie definition of the scattering
  amplitude.
- **Complex permittivity:** absorbing materials have `Im(εₚ) > 0`.
- **Euler angles:** ZYZ intrinsic, same as `scipy.spatial.transform.Rotation`
  and block-DDA_Py.

## Design references

- `.claude/technical_note.md` — EFVIE / SWG formulation, Duffy transform,
  half-SWG surface correction terms (K^B + K^C + K^D)
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
