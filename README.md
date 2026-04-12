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

**v0.1.1 — Phase 5.1–5.3 (CAS-v2 observables) complete.** 1663 tests pass.
All numerical kernels are in place; multi-orientation solves and oblate
spheroid benchmarks match block-DDA_Py and the Mie reference.

| Phase | Module                                         | Status |
|-------|------------------------------------------------|--------|
| 1     | Mesh & SWG / half-SWG / RT1 bases              | done   |
| 2     | Duffy-transform singular integration           | done   |
| 3     | AIM (FFT-MVP) with precorrection                | done   |
| 4     | Direct and iterative (Krylov) solvers           | done   |
| 5.1   | CAS-v2 forward / backward scattering observables| done   |
| 5.2   | Multi-orientation batch solve (shared LU)       | done   |
| 5.3   | Oblate spheroid benchmark vs block-DDA_Py       | done   |
| 5.4   | HDF5 output format parity with block-DDA_Py    | not started |

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
