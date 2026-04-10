# BlockVIEM.jl I/O Specification (block-DDA_Py compatibility)

This document mirrors the input/output conventions of
[`block-DDA_Py`](https://github.com/NobuhiroMoteki/block-DDA_Py) so that
BlockVIEM.jl can be used as a drop-in alternative for CAS-v2 simulation
campaigns. References below cite the upstream README and `docs/theory_note.pdf`
of `block-DDA_Py` (commit pinned in `~/Python_in_WSL/block-DDA_Py`).

The unit system is SI throughout, with the convention that all length-like
inputs share a common length unit chosen by the user (e.g., μm). The harmonic
time convention is `exp(+jωt)` so that an outgoing wave is `exp(-jk₀R)/(4πR)`.

---

## 1. Inputs

### 1.1 Electromagnetic parameters

| Symbol (DDA) | Symbol (VIEM) | Type | Description |
|---|---|---|---|
| `wl_0` | `λ₀` | `Float64` | Vacuum wavelength |
| `m_m` | `n_m` | `Float64` | Medium refractive index (real, non-absorbing) |
| `m_p_x`, `m_p_y`, `m_p_z` | `m_p::SVector{3,ComplexF64}` | complex | Anisotropic particle complex refractive index along principal axes |

Background permittivity: `ε₀_bg = (n_m)² · ε₀_vac`. The contrast ratio used in
VIEM is `κ(r) = (ε(r) - ε₀_bg)/ε(r)` (see `technical_note.md` §2).

### 1.2 Geometry (replaces GRE shape model)

`block-DDA_Py` parameterizes geometry via the Gaussian Random Ellipsoid (GRE)
shape model (`r_v_base`, `bc_ratio`, `ab_ratio`, `gre_beta`). BlockVIEM.jl
instead consumes a tetrahedral mesh:

| Field | Type | Description |
|---|---|---|
| `mesh_file` | `String` | Path to a Gmsh `.msh` file (format 4.x recommended) |
| `physical_tag → m_p` | `Dict{Int,SVector{3,ComplexF64}}` | Per-region refractive index assignment |

For benchmarking against the DDA outputs, BlockVIEM.jl provides
`mesh/gre_to_msh.jl` (Phase 1) to convert a GRE surface to a tetrahedral mesh.

### 1.3 Discretization (Phase 2-3)

| Field | Type | Description |
|---|---|---|
| `target_h` | `Float64` | Target tetrahedron edge length, typically `λ₀/(max\|m_p\|·N_pw)` with `N_pw ≈ 10` |
| `aim_grid_pitch` | `Float64` | Pitch of the AIM auxiliary Cartesian grid |
| `aim_near_radius` | `Float64` | Threshold radius (in grid pitches) below which interactions use direct Duffy integration |

### 1.4 Orientation grid

Identical to `block-DDA_Py` (deterministic uniform grid on SO(3) with
equal-area `cos(β)` sampling):

| Field | Type | Description |
|---|---|---|
| `N_alpha_ori` | `Int` | Divisions in α (azimuth-1) |
| `N_beta_ori` | `Int` | Divisions in β (`cos β` equally spaced in `(-1, 1)`) |
| `N_gamma_ori` | `Int` | Divisions in γ (azimuth-2) |

Total orientations: `L = N_alpha_ori · N_beta_ori · N_gamma_ori`.
For axisymmetric targets, a "spheroid mode" analogue may be added in Phase 5.

---

## 2. Primary outputs (CAS-v2 observables)

CAS-v2 (Moteki & Adachi 2024, Optics Express 32, 36500) measures the polarized
complex forward-scattering amplitudes. BlockVIEM.jl reproduces them per
orientation:

| Field | Shape | Description |
|---|---|---|
| `S_fw_s` | `Complex{Float64}[L]` | Forward scattering amplitude, s-polarization |
| `S_fw_p` | `Complex{Float64}[L]` | Forward scattering amplitude, p-polarization |
| `S_bak`  | `Complex{Float64}[L]` | Backward scattering amplitude (scalar) |

Both real and imaginary parts are stored separately on disk to match the DDA
HDF5 layout (`S_fw_s_re`, `S_fw_s_im`, ...).

### Forward / backward direction conventions
- Forward: `k̂_sca = k̂_inc`
- Backward: `k̂_sca = -k̂_inc`
- s-polarization: incident `Ê` perpendicular to scattering plane
- p-polarization: incident `Ê` in scattering plane

These match the CAS-v2 definitions in
`~/Python_in_WSL/block-DDA_Py/docs/theory_note.pdf` (Sec. 7).

---

## 3. Secondary outputs (cross sections, validation)

| Field | Type | Description |
|---|---|---|
| `C_ext` | `Float64[L]` | Extinction cross section (optical theorem from `S_fw_s`, `S_fw_p`) |
| `C_abs` | `Float64[L]` | Absorption cross section (volumetric Joule loss from interior `D` field) |
| `C_sca` | `Float64[L]` | `C_ext - C_abs` |
| `converged` | `Bool[L]` | Block-Krylov convergence flag per orientation |

Phase function and Mueller matrix are **not** primary outputs (per project
spec) but may be derived externally from the complex amplitudes.

---

## 4. HDF5 layout (sweep mode)

To allow direct cross-validation with `block-DDA_Py` sweeps, BlockVIEM.jl
writes the same root-level grid axes and per-wavelength group structure.
For the spheroid sweep file
(`dda_results/dda_results_spheroid_sweep.h5`):

```
/
├── D_ve_grid              (N_Dve,)             float64
├── RI_real_grid           (N_RI,)              float64
├── log_AR_grid            (N_AR,)              float64
├── cos_theta_o_half_grid  (N_u_half,)          float64
├── phi_o_grid             (N_ph,)              float64
└── wl_<wavelength>/
    ├── S_fw_theta_re      (N_Dve,N_RI,N_AR,N_u_half,N_ph)  float64
    ├── S_fw_theta_im      ...
    ├── S_fw_phi_re        ...
    ├── S_fw_phi_im        ...
    └── converged          (N_Dve,N_RI,N_AR)               bool
```

For the general sweep file (`dda_results/dda_results.h5`), the dataset shape
is `(N_pairs, N_mp, N_rv, N_bc, N_ab, N_bt, N_ori)` per the upstream README.
BlockVIEM.jl writes the same shape but with `(N_rv, N_bc, N_ab, N_bt)`
replaced by mesh-file references when GRE conversion is not used.

---

## 5. Open questions

1. **Anisotropic permittivity per tetrahedron**: VIEM with diagonal-tensor
   `ε(r) = diag(εx, εy, εz)` requires extending the SWG formulation;
   the technical note currently assumes scalar `ε`. Anisotropy support is
   deferred to Phase 5+.
2. **HDF5 backend choice**: `HDF5.jl` vs `JLD2.jl` — to be decided when
   PostProcess is implemented. The bytewise layout above assumes `HDF5.jl`.
3. **Volume-preserving rescaling** (DDA `dpl`/`r_v_base` mechanism): VIEM
   uses an explicit mesh, so rescaling must be applied at mesh-generation
   time rather than at solver entry. A helper for "rescale `.msh` to a
   target volume-equivalent radius" should live in Phase 1.
