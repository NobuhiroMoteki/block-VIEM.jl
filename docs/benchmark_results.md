# Benchmark Results — block-VIEM.jl

This document is the authoritative record of all validation and
cross-comparison benchmarks for block-VIEM.jl. It collects in one
place:

- Numerical conditions (geometry, material, mesh, solver settings).
- Reference data (Mie series, MSTM T-matrix, block-DDA_Py).
- Tabulated relative errors for every observable reported in the
  paper-production HDF5 schema (`C_ext`, `C_abs`, `C_sca`,
  `S_fw_θ`, `S_fw_φ`, `S_bk`).
- Mesh-refinement (`ℓ_c` convergence) studies and the corresponding
  fitted convergence rates.
- Memory and wall-time scaling for the AIM operator and solver.

Per-section quick navigation:

| § | Benchmark | Reference |
| --- | --- | --- |
| [§0](#0-conventions-used-in-error-reporting) | Error-reporting conventions | — |
| [§1](#1-mie-sphere-dielectric) | Mie sphere, dielectric | Mie series |
| [§2](#2-mie-sphere-plasmonic-gold) | Mie sphere, plasmonic Au | Mie series |
| [§3](#3-oblate-spheroid-α-symmetry-self-check) | Oblate spheroid α-symmetry | analytical |
| [§4](#4-oblate-spheroid-cross-validation-vs-block-dda_py) | Oblate spheroid vs DDA | block-DDA_Py |
| [§5](#5-gre-cross-validation-vs-block-dda_py-β--0) | GRE (β > 0) vs DDA | block-DDA_Py |
| [§6](#6-anisotropic-ε_p-cross-validation-vs-block-dda_py) | Anisotropic ε_p vs DDA | block-DDA_Py |
| [§7](#7-two-sphere-doublet-vs-mstm) | 2-sphere doublet vs MSTM | MSTMforCAS.jl |
| [§8](#8-doublet-mesh-refinement-and-aim-memory) | Doublet ℓ_c refinement + memory | self |
| [§9](#9-per-dof-accuracy-vs-block-dda_py) | Per-DoF accuracy vs DDA | analytical (Mie) |
| [§10](#10-paper-production-ℓ_c-convergence-protocol) | Paper-production ℓ_c protocol | self |
| [§11](#11-shifted-block-gmres-for-refractive-index-sweeps) | Shifted block GMRES for refractive-index sweeps | existing solver |

Each subsection's first line gives the driver script; reproduction
is `julia --project=. <script>` unless noted otherwise.

---


## 0. Conventions used in error reporting

Throughout this document, relative errors are defined pointwise:

$$
\text{rel. err.}^{\operatorname{Re}} =
\frac{|\operatorname{Re} S^\text{VIEM} - \operatorname{Re} S^\text{ref}|}{|\operatorname{Re} S^\text{ref}|}, \quad
\text{rel. err.}^{\operatorname{Im}} =
\frac{|\operatorname{Im} S^\text{VIEM} - \operatorname{Im} S^\text{ref}|}{|\operatorname{Im} S^\text{ref}|},
$$

and "complex relative error" is

$$
\text{complex rel. err.} =
\frac{|S^\text{VIEM} - S^\text{ref}|}{|S^\text{ref}|}.
$$

Cross-section errors are
$|\Delta C| / C^\text{ref}$. "Phase err." is
$\arg(S^\text{VIEM} / S^\text{ref})$.

**Backward amplitude.** Since v0.8.0 (2026-11-19) `S_bk` is
$(-S_{11} + S_{22})/2$ at the backward direction, the corrected Eq. (28)
of Moteki and Adachi (2024). The benchmarks of §1–§10 were run with the
printed equation, which has $1/\sqrt{2}$ in place of $1/2$. The
exception is the doublet benchmark of §7 and §8, which was re-run with
v0.8.2 on 2026-11-19. The VIEM
value and its reference changed by the same factor, so the relative and
phase errors of `S_bk` reported below are unchanged. The HDF5 files
under `viem_results/paper/` were converted to the corrected form on
2026-11-19. The result files under `benchmarks/` no longer contain
backward amplitudes in the printed form: the doublet benchmark files
hold the corrected form and the two `S_bk` entries of
`benchmarks/cas_v2/dda_comparison/viem_result.json` were removed.

---

## 1. Mie sphere, dielectric

Driver: `test/mie_reference.jl` (used inside the Phase-5 validation
tests, not a separate benchmark script).

**Conditions.** Volume-equivalent sphere,
$\lambda_0 = 4\,\text{μm}$, $m_m = 1$,
$m_p = 1.5 + 0.01\,i$, $r = 0.5\,\text{μm}$.
Mesh sweep $\ell_c \in \{0.30, 0.18, 0.12\}$ μm.

**Result.** After the 2026-04-13 physics-convention switch, both
real and imaginary parts of the PCAS forward amplitude
$S_\text{fw}^\text{PCAS}$ agree with the Mie reference to
$\sim 0.4\,\%$ at the coarsest mesh and $\sim 0.1\,\%$ at
$\ell_c = 0.12$. Used as a smoke test of the SWG / half-SWG basis
in the dielectric regime; no numerical tables are kept here because
the test_*.jl assertions enforce the tolerance directly.

---

## 2. Mie sphere, plasmonic gold

Driver: [benchmarks/cas_v2/au_sphere_mie.jl](../benchmarks/cas_v2/au_sphere_mie.jl).

**Conditions.**
- Material: Johnson & Christy 1972 gold,
  $m_\text{Au} = 0.17525 + 3.4830\,i$,
  $\varepsilon_\text{Au} = -12.10 + 1.22\,i$ at
  $\lambda_0 = 0.638\,\text{μm}$ (via
  [refractiveindex.info](https://refractiveindex.info/?shelf=main&book=Au&page=Johnson)).
- Geometry: sphere, size parameter $x = k_0 r = 0.63$,
  $r \approx 0.0640\,\text{μm}$.
- Background: vacuum, $m_m = 1$.
- Basis: half-SWG (boundary faces included).
- Quadrature: `duffy_reference_rule(5)`.
- Two mesh sizes, two solvers: dense LU and AIM-BiCGSTAB.
- Single-threaded wall times (the v0.6.0 threaded setup gives a
  2–3 × speed-up at 4 + Julia threads).

**Mie reference for the target sphere.**

$$
Q_\text{ext} = 1.386, \quad Q_\text{abs} = 0.192, \quad Q_\text{sca} = 1.195,
$$

$$
S_\text{fw,mean}^\text{PCAS} = +0.03877 + 0.01397\,i,
\qquad
S_\text{bk}^\text{OCBS} = +0.04204 + 0.01348\,i.
$$

The value of $S_\text{bk}^\text{OCBS}$ follows the corrected definition
(§0). With the printed equation used when this benchmark was run it was
$+0.05946 + 0.01907\,i$.

Errors below are relative to Mie evaluated at the actual mesh
volume-equivalent radius (so residual mesh-volume bias does not
contaminate the discretisation error).

| ℓ_c [μm] | N    | Method       | C_ext  | C_abs  | C_sca  | S_fw,mean | S_bk   | wall  |
| -------- | ---- | ------------ | ------ | ------ | ------ | --------- | ------ | ----- |
| 0.020    | 2134 | dense (LU)   | 0.65 % | 0.005 % | 0.75 % | 0.58 %    | 0.28 % | 17 s  |
| 0.020    | 2134 | AIM-BiCGSTAB | 0.25 % | 0.44 % | 0.37 % | 0.32 %    | 0.11 % | 43 s  |
| 0.014    | 4932 | dense (LU)   | 0.34 % | 0.08 % | 0.41 % | 0.32 %    | 0.15 % | 68 s  |
| 0.014    | 4932 | AIM-BiCGSTAB | 0.15 % | 0.24 % | 0.22 % | 0.18 %    | 0.07 % | 105 s |

**Findings.**

1. All five observables converge **monotonically** between the two
   meshes. Doubling DOF count $2134 \to 4932$ (factor $\sim 2.3$)
   cuts every error by a factor $\sim 2$ — consistent with $h^2$
   per-DOF scaling.
2. Even at the coarse mesh ($N = 2134$, only ~ 4 elements per
   radius) every observable is below 1 % relative error, well
   inside the CAS-v2 LUT-generation budget of ~ 2 % tolerated by
   the downstream Bayesian retrieval.
3. Polarimetric amplitudes $S_\text{fw,mean}$ and $S_\text{bk}$ are
   more accurate than the optical-theorem cross sections at the
   same mesh, because cross-section evaluation involves an
   additional far-field quadrature on the unit sphere whose error
   adds to the SWG discretisation error. The amplitudes inherit
   only the SWG error.
4. Monotone sub-1 % accuracy in the plasmonic regime — where the
   divergence-free part of $\boldsymbol{D}$ is comparable to the divergent
   (charge) part — confirms that the half-SWG surface correction
   $K^{B} + K^{C} + K^{D}$ correctly enforces the surface
   polarisation-charge boundary condition. The interior-only SWG
   basis (no boundary half-functions) produces 10 – 30 % errors
   here.

---

## 3. Oblate spheroid α-symmetry self-check

Driver: [benchmarks/cas_v2/spheroid_ar3.jl](../benchmarks/cas_v2/spheroid_ar3.jl).

**Setup.** Oblate spheroid `bc_ratio = b/c = 3`, single mesh,
β = π/4, α swept over `{0, π/8, π/4, 3π/8, π/2}`. Each α is solved
independently and compared against the analytical α-expansion

$$
S_\theta(\alpha, \beta, 0) = A(\beta) + B(\beta)\,e^{+2 i \alpha},
\qquad
S_\phi(\alpha, \beta, 0) = A(\beta) - B(\beta)\,e^{+2 i \alpha},
$$

with $A, B$ from solves at $\alpha = 0$. The `+2iα` sign matches
block-DDA_Py after the physics-convention switch.

**Result.** Maximum deviation $\sim 5 \times 10^{-15}$ — machine
precision for double-precision arithmetic. This is an internal
consistency check: it confirms the rotational machinery, but does
not validate the absolute amplitudes (that is done by §4 below).

---

## 4. Oblate spheroid cross-validation vs block-DDA_Py

Drivers:
[benchmarks/cas_v2/dda_comparison/run_dda_spheroid_single.py](../benchmarks/cas_v2/dda_comparison/run_dda_spheroid_single.py)
(DDA) and the matching VIEM driver in the same directory. JSON is
the exchange format.

**Conditions.**
- Particle: oblate spheroid, $D_{ve} = 0.40\,\text{μm}$,
  `bc_ratio = 3`, $m_p = 1.5$ (non-absorbing).
- Background: $m_m = 1$, $\lambda_0 = 0.638\,\text{μm}$
  ($k_0 r_{ve} \approx 1.97$).
- Orientations: $(\alpha, \beta, \gamma) = (0, \pi/4, 0)$ and
  $(0, \pi/2, 0)$.
- Discretisations: DDA $N_\text{dip} = 2236$ with dpl = 17;
  VIEM half-SWG, $\ell_c = 0.035\,\text{μm}$,
  $N_\text{dof} = 8746$.

**Result.**

| Observable | block-DDA_Py | block-VIEM.jl | rel. err. |
| --- | --- | --- | --- |
| **β = π/4** | | | |
| $S_{\text{fw},\theta}$ | $+0.23514 + 0.09368\,i$ | $+0.23143 + 0.09164\,i$ | 1.67 % |
| $S_{\text{fw},\phi}$ | $+0.23884 + 0.18561\,i$ | $+0.23480 + 0.17937\,i$ | 2.46 % |
| $C_\text{ext}$ | $0.17819\,\text{μm²}$ | $0.17291\,\text{μm²}$ | 2.96 % |
| **β = π/2** | | | |
| $S_{\text{fw},\theta}$ | $+0.23703 + 0.08742\,i$ | $+0.23221 + 0.08536\,i$ | 2.08 % |
| $S_{\text{fw},\phi}$ | $+0.29578 + 0.21695\,i$ | $+0.28980 + 0.20883\,i$ | 2.75 % |
| $C_\text{ext}$ | $0.19419\,\text{μm²}$ | $0.18769\,\text{μm²}$ | 3.34 % |

**Findings.**

- VIEM and DDA agree to ~ 2–3 % on all complex polarimetric
  amplitudes at both tilts, on both real and imaginary parts
  independently — not just on magnitudes.
- The VIEM mesh recovered $r_{ve}^\text{mesh} = 0.19918\,\text{μm}$
  (0.41 % volumetric error, before the v0.7.7 rescale; current
  builds hit $r_{ve} = 0.20000$ to machine precision); DDA
  introduces a comparable stair-casing error. The ~ 2 % code-to-code
  gap is the discretisation-limited expectation at these mesh
  densities, not a methodology disagreement.

---

## 5. GRE cross-validation vs block-DDA_Py (β > 0)

Drivers: [benchmarks/cas_v2/gre_comparison/run_dda_gre.py](../benchmarks/cas_v2/gre_comparison/run_dda_gre.py)
(DDA) and matching VIEM driver. The same Gaussian random
deformation field $s(\theta, \phi)$ is generated by the Python
script and consumed by the Julia side via
[gre_mesh_with_field](../src/gre_mesh.jl#L571) so both codes solve
the identical shape.

**Common parameters.** $\lambda_0 = 0.638\,\text{μm}$,
$n_m = 1.0$, $m_p = 1.5 + 0.01\,i$,
$r_\text{v,base} = 0.20\,\text{μm}$. DDA uses dpl = 17
($N_\text{DDA} \approx 2300$); VIEM uses
$\ell_c = 0.035\,\text{μm}$
($N_\text{VIEM} \approx 8600$ half-SWG DOFs). Two tilt angles
$\beta_\text{ori} = \pi/4, \pi/2$.

| Case | b/c | a/b | β    | $\|\Delta S_\theta\|/\|S_\theta\|$ | $\|\Delta S_\phi\|/\|S_\phi\|$ |
| ---- | --- | --- | ---- | ---------------------------------- | ------------------------------ |
| A    | 2.0 | 1.0 | 0.10 | 0.5 – 0.7 % | 0.1 – 0.3 % |
| B    | 3.0 | 1.0 | 0.15 | 2.0 – 2.6 % | 1.7 – 1.8 % |
| C    | 1.5 | 1.5 | 0.20 | 1.2 – 1.6 % | 1.6 – 1.8 % |

Larger errors at higher β and b/c reflect both the coarser
per-feature DDA-lattice resolution on thin / strongly deformed
shapes and the inherent discretisation mismatch between
cubic-lattice DDA and tetrahedral VIEM.

---

## 6. Anisotropic ε_p cross-validation vs block-DDA_Py

Drivers under [benchmarks/cas_v2/aniso_comparison/](../benchmarks/cas_v2/aniso_comparison/).

The VIEM side of this benchmark uses `method = :dense`. Up to v0.8.0 the
AIM path applied the mass term of an anisotropic particle with the mean
of $1/\epsilon_\alpha$ instead of the component weights, and its
anisotropic results were off by 5–25 % (measured on the coarse test
sphere for index differences of 0.1–0.5 between the axes). The AIM path
now matches the dense matrix to 0.2 %, as for isotropic particles
(`test/test_anisotropic.jl`).

**Setup.** Sphere, $r_{ve} = 0.20\,\text{μm}$,
$\lambda_0 = 0.638\,\text{μm}$, with diagonal-tensor
refractive index $\boldsymbol{m}_p = (m_x, m_y, m_z)$ passed identically to
both DDA and VIEM. Two tilt angles
$\beta_\text{ori} = \pi/4, \pi/2$. VIEM mesh:
$\ell_c = 0.035\,\text{μm}$, $N_\text{VIEM} \approx 8000$
half-SWG DOFs.

| Case   | $m_p$               | $\|\Delta S_\theta\|/\|S_\theta\|$ | $\|\Delta S_\phi\|/\|S_\phi\|$ |
| ------ | ------------------- | ---------------------------------- | ------------------------------ |
| iso    | $[1.5, 1.5, 1.5]$   | 1.3 – 2.4 % | 2.4 % |
| mild   | $[1.55, 1.5, 1.45]$ | 1.2 – 3.5 % | 2.3 – 2.6 % |
| strong | $[1.6, 1.5, 1.4]$   | 0.7 – 5.0 % | 2.3 – 2.9 % |

Agreement is at the same few-percent level as the isotropic case,
confirming that the diagonal-tensor extension of the EFVIE-D
formulation (introduced in v0.4.0) reproduces block-DDA_Py's
`m_p_xyz` channel-wise weighting throughout the assembly,
near-field, AIM, and far-field paths.

---

## 7. Two-sphere doublet vs MSTM

> **Re-run with v0.8.2 (2026-11-19).** The results of §7 and §8 were
> computed again with v0.8.2. The earlier results (git history, commit
> `d1482fa`) dated from April 2026 (v0.4.2 to v0.5.0). Two things differ:
>
> - **Mesh volume.** In April 2026 the mesh was used as Gmsh returned
>   it. Its nodes lie on the nominal spheres, so the centre distance
>   ($2.1 R$) and the gap ($0.1 R$) were nominal, but the volume of the
>   faceted spheres was below the nominal volume: by 1.40 %, 1.00 %,
>   0.72 % and 0.53 % at $\ell_c = R/5, R/6, R/7, R/8$. From v0.7.7 the
>   mesh is scaled to the nominal volume, and from v0.8.2 each sphere is
>   scaled about its own centre, so the centre distance stays $2.1 R$.
> - **Solver.** The earlier runs used AIM block BiCGSTAB. The present
>   runs use AIM block GMRES with right preconditioning by the mass
>   matrix (§7.2). Every solve reported here reached the tolerance
>   $10^{-7}$.
>
> The errors against MSTM are several times smaller than before. The
> difference comes from the mesh volume and not from the solver. This
> was checked at $\ell_c = R/5$ by solving the mesh without the volume
> rescale with the present solver (2026-11-20, $\beta = \pi/2$):
>
> | Mesh | Solver | PS $\|S_\text{fw,mean}\|$ | Au $\|S_\text{fw,mean}\|$ | Au $\operatorname{Im} S_{\text{fw},\theta}$ |
> | ---- | ------ | ----- | ----- | ----- |
> | no volume rescale | block BiCGSTAB (results of April 2026) | 1.6 % | 4.0 % | 10.5 % |
> | no volume rescale | block GMRES, converged | 1.62 % | 4.36 % | 10.62 % |
> | each sphere scaled to its nominal volume (v0.8.2) | block GMRES, converged | 0.13 % | 1.09 % | 2.34 % |

Drivers under [benchmarks/cas_v2/doublet_mstm/](../benchmarks/cas_v2/doublet_mstm/):

```bash
# 1. block-VIEM predictions
julia --project=. benchmarks/cas_v2/doublet_mstm/run_viem.jl

# 2. exact MSTM reference (run inside the MSTMforCAS.jl env)
julia --project=/path/to/MSTMforCAS.jl \
      benchmarks/cas_v2/doublet_mstm/run_mstm.jl

# 3. relative-error tables
julia --project=. benchmarks/cas_v2/doublet_mstm/compare.jl
```

Both scripts read a shared `config.jl` so the two codes solve the
exact same physical problem.

### 7.1 Geometry, materials, and observable

- Two equal-radius spheres of radius $R = 0.030\,\text{μm}$
  with axial gap $0.003\,\text{μm}$ (centre-to-centre
  separation $d = 0.063\,\text{μm}$).
- Doublet axis along the particle-frame $z$. The spheres are
  topologically disjoint so MSTM treats each monomer as an
  isolated VSWF expansion.
- Vacuum background, $\lambda_0 = 0.638\,\text{μm}$.
- Three orientations $\beta \in \{0, \pi/4, \pi/2\}$, where
  $\beta$ is the angle between $\hat{\boldsymbol{u}}_\text{inc}$ and the
  doublet axis. VIEM realises $\beta$ via Euler $(0, \beta, 0)$;
  MSTM rotates the doublet about $\hat{\boldsymbol{e}}_y$ by $\beta$.
- Two materials: high-contrast dielectric (polystyrene-like)
  $m_p = 1.60 + 0.01\,i$ and plasmonic
  $m_p = 0.17525 + 3.4830\,i$ (Au @ 638 nm, Johnson & Christy 1972).

The CAS-v2 forward amplitudes are defined in the
block-DDA_Py / block-VIEM convention (RCP incidence) as

$$
S_{\text{fw},\theta} = S_{11} + i\,S_{12}, \quad
S_{\text{fw},\phi}   = S_{22} - i\,S_{21}, \quad
S_{\text{fw,mean}}    = \tfrac{1}{2}(S_{\text{fw},\theta} + S_{\text{fw},\phi}),
$$

with $S_{ij}$ the MI02 forward-direction amplitudes. The MSTM side
returns the BH83 tuple $(S_1, S_2, S_3, S_4)$ and the conversion is

$$
S_{11} = \frac{S_2}{-i k_m}, \quad
S_{12} = \frac{S_3}{i k_m},  \quad
S_{21} = \frac{S_4}{i k_m},  \quad
S_{22} = \frac{S_1}{-i k_m},
$$

so that $S_{\text{fw},\theta} = (S_2 - i S_3)/(-i k_m)$ and
$S_{\text{fw},\phi} = (S_1 + i S_4)/(-i k_m)$ with
$k_m = 2\pi m_m / \lambda_0$. The reflection-symmetric doublet
satisfies $S_3 = S_4 = 0$, so the off-diagonal terms drop out for
this benchmark; the correct signs are retained in the driver code
for generic aggregates.

### 7.2 Numerical setup

- VIEM: $\ell_c = R/5 = 0.006\,\text{μm}$, 5348 tets,
  $N = 11501$ half-SWG DOFs. AIM block GMRES (`:aim_gmres`) with right
  preconditioning by the mass matrix (`precond = :mass_right`),
  tolerance $10^{-7}$, at most 400 iterations, pitch
  $0.5\bar{h} = 0.0038\,\text{μm}$. The three orientations are solved
  as one block. The solve converged in 10 iterations for the dielectric
  doublet and in 167 iterations for the Au doublet.
- Choice of solver (Au doublet, $\ell_c = R/5$, tolerance $10^{-7}$):

  | Solver | Iterations | Final residual | Largest deviation of an amplitude component from the converged solution |
  | ------ | ---------- | -------------- | ----------------------------------------------------------------------- |
  | block BiCGSTAB, at most 400 iterations | 400 (not converged) | 8.0e−3 | 4.8e−3 |
  | block GMRES, no preconditioner, at most 400 iterations | 400 (not converged) | 9.2e−5 | 2.7e−5 |
  | block GMRES, no preconditioner, at most 1200 iterations | 621 | < 1e−7 | reference |
  | block GMRES, mass preconditioner | 167 | 9.3e−8 | < 1e−7 |

  The preconditioner changes the iteration count and not the solution.
  The production runner `viem_results/run_viem.jl` uses block GMRES
  without a preconditioner and at most 200 iterations.
- MSTM: VSWF truncation order **forced to $N_\text{trunc} = 15$**
  for both materials.  The MSTMforCAS auto-truncation returns
  $N = 3$ at $x \approx 0.30$, which is converged for the
  dielectric case but **under-truncates** the plasmonic Au case
  by $\approx 5\,\%$ on $|S_\text{fw,mean}|$ — the inter-sphere
  coupling amplified by $|m\!\cdot\!x|$ drives higher-order
  multipoles even though the single-sphere Mie expansion is long
  since converged.  $N = 15$ reaches rtol $\le 10^{-6}$ on both
  materials at $x \approx 0.3$ (geometric convergence in $N$,
  per-order ratio $\approx 0.3$).  CBICG tolerance $10^{-8}$.
  Any new comparison point should re-confirm convergence with
  `benchmarks/cas_v2/doublet_mstm/check_trunc_convergence.jl`.

### 7.3 Dielectric doublet — $m_p = 1.60 + 0.01\,i$

#### Magnitude summary

| β [rad] | $\|S_\text{fw,mean}\|$ VIEM | $\|S_\text{fw,mean}\|$ MSTM | rel. $\|\cdot\|$ err. | complex rel. err. | phase err. [rad] |
| ------- | ------------------ | ------------------ | --------------------- | ----------------- | ---------------- |
| 0       | 1.7930e−03         | 1.7939e−03         | 0.05 %                | 0.05 %            | 3.8e−06          |
| π/4     | 1.8473e−03         | 1.8489e−03         | 0.09 %                | 0.09 %            | 3.5e−06          |
| π/2     | 1.9056e−03         | 1.9081e−03         | 0.13 %                | 0.13 %            | 2.3e−05          |

#### $S_{\text{fw},\theta}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +1.7925e−03 | +1.7934e−03 | 0.05 %     | +4.1903e−05 | +4.1991e−05 | 0.21 %     |
| π/4     | +1.9041e−03 | +1.9066e−03 | 0.13 %     | +4.8930e−05 | +4.9082e−05 | 0.31 %     |
| π/2     | +2.0239e−03 | +2.0279e−03 | 0.20 %     | +5.6501e−05 | +5.6771e−05 | 0.48 %     |

#### $S_{\text{fw},\phi}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +1.7926e−03 | +1.7934e−03 | 0.05 %     | +4.2025e−05 | +4.1991e−05 | 0.08 %     |
| π/4     | +1.7893e−03 | +1.7902e−03 | 0.05 %     | +4.2780e−05 | +4.2724e−05 | 0.13 %     |
| π/2     | +1.7860e−03 | +1.7870e−03 | 0.06 %     | +4.3512e−05 | +4.3463e−05 | 0.11 %     |

All components agree with MSTM within 0.5 %: the real parts within
0.05–0.20 % and the imaginary parts within 0.08–0.48 %. The errors of
the earlier run (April 2026, mesh without the volume rescale) were
1.4–1.7 % on the real parts and 1.7–2.7 % on the imaginary parts. The
error of the real parts was close to the volume deficit of that mesh
(1.40 %).

### 7.4 Plasmonic Au doublet — $m_p = 0.17525 + 3.4830\,i$

#### Magnitude summary

| β [rad] | $\|S_\text{fw,mean}\|$ VIEM | $\|S_\text{fw,mean}\|$ MSTM | rel. $\|\cdot\|$ err. | complex rel. err. | phase err. [rad] |
| ------- | ------------------ | ------------------ | --------------------- | ----------------- | ---------------- |
| 0       | 6.5042e−03         | 6.5039e−03         | 0.004 %               | 0.005 %           | 3.4e−05          |
| π/4     | 8.2293e−03         | 8.2865e−03         | 0.69 %                | 0.70 %            | 8.9e−04          |
| π/2     | 1.0085e−02         | 1.0197e−02         | 1.09 %                | 1.10 %            | 1.1e−03          |

#### $S_{\text{fw},\theta}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +6.4886e−03 | +6.4886e−03 | 0.001 %    | +4.4603e−04 | +4.4705e−04 | 0.23 %     |
| π/4     | +9.9280e−03 | +1.0037e−02 | 1.08 %     | +1.3105e−03 | +1.3381e−03 | 2.07 %     |
| π/2     | +1.3603e−02 | +1.3818e−02 | 1.55 %     | +2.2322e−03 | +2.2857e−03 | 2.34 %     |

#### $S_{\text{fw},\phi}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +6.4890e−03 | +6.4886e−03 | 0.007 %    | +4.4766e−04 | +4.4705e−04 | 0.14 %     |
| π/4     | +6.4356e−03 | +6.4392e−03 | 0.06 %     | +4.5567e−04 | +4.5502e−04 | 0.14 %     |
| π/2     | +6.3862e−03 | +6.3893e−03 | 0.05 %     | +4.6397e−04 | +4.6332e−04 | 0.14 %     |

#### Discussion

The error of the Au doublet depends on the orientation and on the
polarisation.

- The **$\phi$ channel** agrees with MSTM within 0.06 % on the real
  part and 0.14 % on the imaginary part at every $\beta$.
  $S_{\text{fw},\phi}$ probes the polarisation transverse to the doublet
  axis, for which the coupling of the two spheres through the gap is
  weak.
- The **$\theta$ channel** has the largest errors: 1.6 % on the real
  part and 2.3 % on the imaginary part at $\beta = \pi/2$. In this
  configuration the field is polarised along the doublet axis and
  concentrates in the gap. At $\beta = 0$ both channels are transverse
  to the axis and the errors stay below 0.25 %.

The relative error of $|S_\text{fw,mean}|$ (1.1 % at $\beta = \pi/2$)
is about half of the error of $\operatorname{Im} S_{\text{fw},\theta}$.
The component view is therefore the better indicator when $\ell_c$ is
chosen for plasmonic targets. One refinement step to $\ell_c = R/6$
brings every component within 0.4 % (§8).

The earlier run (April 2026, mesh without the volume rescale, block
BiCGSTAB) had errors of 4.0 % on $|S_\text{fw,mean}|$ and 10.5 % on
$\operatorname{Im} S_{\text{fw},\theta}$ at $\beta = \pi/2$. These
errors are reproduced with the present solver on the mesh without the
rescale (4.36 % and 10.62 %, see the note at the top of §7). They
exceed the volume deficit of that mesh (1.40 %) several times. A likely
reason is that the faces of the unscaled mesh lie inside the nominal
spheres, so the two surfaces are on average farther apart than the
nominal gap, and the $\theta$ channel is sensitive to the gap. This
interpretation was not tested separately.

---

## 8. Doublet mesh refinement and AIM memory

Driver:
[benchmarks/cas_v2/doublet_mstm/phase_a_memory_study.jl](../benchmarks/cas_v2/doublet_mstm/phase_a_memory_study.jl)
+ `compute_refinement_errors.py`. Au doublet of §7. Same
geometry, four refinement levels $\ell_c = R/k$ for
$k = 5, 6, 7, 8$ (the script default is $k = 5, 6, 7$; $k = 8$ is set
with `BM_LC_RATIOS`), all compared against the converged MSTM
$N = 15$ reference. Solver as in §7.2 (block GMRES with the mass
preconditioner). Measured on 2026-11-19 with v0.8.2 on an AMD Ryzen 5
220 laptop (16 GB RAM, 12 GB available to WSL2, Julia 1.12.5) with one
Julia thread. More Julia threads shorten the setup and the block MVP.

### 8.1 Per-component error vs ℓ_c at β = π/2

Relative error against MSTM, Au doublet, $\beta = \pi/2$. Every solve
reached the tolerance $10^{-7}$ (167, 166, 218 and 234 iterations for
$k = 5, 6, 7, 8$).

| ℓ_c / R | N DOF  | $\|S_\text{fw,mean}\|$ | $\operatorname{Re} S_{\text{fw},\theta}$ | $\operatorname{Im} S_{\text{fw},\theta}$ | $\operatorname{Re} S_{\text{fw},\phi}$ | $\operatorname{Im} S_{\text{fw},\phi}$ |
| ------- | ------ | ----- | ----- | ------ | ----- | ----- |
| 1/5     | 11 501 | 1.09 %| 1.55 %| 2.34 % | 0.05 %| 0.14 %|
| 1/6     | 18 917 | 0.25 %| 0.37 %| 0.03 % | 0.03 %| 0.17 %|
| 1/7     | 29 616 | 0.12 %| 0.16 %| 0.14 % | 0.04 %| 0.21 %|
| 1/8     | 45 697 | 0.24 %| 0.35 %| 0.41 % | 0.003 %| 0.05 %|

Largest error over the five components at the other orientations:
0.23 %, 0.18 %, 0.10 %, 0.05 % at $\beta = 0$ and 2.07 %, 0.25 %,
0.10 %, 0.36 % at $\beta = \pi/4$ for $k = 5, 6, 7, 8$.

From $\ell_c = R/6$ on, every component at every orientation is within
0.41 % of MSTM. The step from $R/5$ to $R/6$ reduces the
$\theta$-channel error by a factor of 4 or more. Below that the errors
do not decrease monotonically. The error at $R/8$ is larger than at
$R/7$ for the $\theta$ channel, and the sign of the error of
$\operatorname{Im} S_{\text{fw},\theta}$ changes between levels
(−2.34 %, +0.03 %, +0.14 %, −0.41 %).

The earlier run (April 2026, mesh without the volume rescale) gave
4.4 %, 2.7 %, 1.9 % and 1.6 % on $|S_\text{fw,mean}|$ for the same four
levels.

### 8.2 Fitted convergence rate $p$ in $\text{err} \propto (\ell_c/R)^p$

4-point log-log least-squares fit of the $\beta = \pi/2$ errors of
§8.1:

| Component | $p$ |
| --------- | --- |
| $\|S_\text{fw,mean}\|$ | 3.6 |
| $\operatorname{Re} S_{\text{fw},\theta}$ | 3.6 |
| $\operatorname{Im} S_{\text{fw},\theta}$ | 2.9 |
| $\operatorname{Re} S_{\text{fw},\phi}$ | 4.9 |
| $\operatorname{Im} S_{\text{fw},\phi}$ | 1.8 |

These exponents are **not** convergence rates. The fit is dominated by
the drop between $R/5$ and $R/6$, and the errors at $R/6$ to $R/8$
scatter between 0.003 % and 0.41 % without a monotone trend. A
convergence rate cannot be read from these four levels. The origin of
the scatter was not investigated. Candidates are the AIM approximation
of the far-field interactions and the change of the unstructured mesh
in the gap region from level to level.

The earlier results (April 2026) gave $p = 2.1$–$2.5$ on every
component. They were computed on meshes without the volume rescale,
whose errors were dominated by the geometric error of the faceted
spheres. That error decreases as $\ell_c^2$: the volume deficit of the
four meshes (1.40 %, 1.00 %, 0.72 %, 0.53 %) has the exponent 2.07. The
earlier exponents therefore measured the geometric error of the
unscaled mesh. The volume rescale removes most of that error, and the
remaining errors are too small and too irregular for a fit.

### 8.3 Phase A AIM-operator memory

`total tracked` = `Base.summarysize(AIMOperator)` — every sparse
matrix, FFT kernel, and mass matrix in the operator. For
comparison the original Phase-1a `half_swg_extra` matrix alone
reached 486 MiB at $R/5$ and 1074 MiB
at $R/6$ on the same geometry; Phase A reduces those totals to
97 MiB and 164 MiB — a 5.0 × / 6.5 × /
11 × reduction at $R/5$ / $R/6$ / $R/7$ respectively. Asymptotic:
total tracked grows as $\mathcal O(N_\text{dof}^{1.05})$ —
essentially linear in $N$.

| ℓ_c / R | N DOF  | N_bnd | $W^S$    | precorrection | total tracked | t_setup | t_solve (3 orient.) | iterations |
| ------- | ------ | ----- | -------- | ------------- | ------------- | ------- | ------------------- | ---------- |
| 1/5     | 11 501 | 1 610 | 4.9 MiB  | 67.3 MiB      | **96.8 MiB**  | 143 s   | 173 s               | 167        |
| 1/6     | 18 917 | 2 254 | 8.1 MiB  | 116.5 MiB     | **164.3 MiB** | 236 s   | 290 s               | 166        |
| 1/7     | 29 616 | 3 176 | 12.6 MiB | 188.8 MiB     | **262.5 MiB** | 369 s   | 537 s               | 218        |
| 1/8     | 45 697 | 4 210 | 19.4 MiB | 302.4 MiB     | **415.2 MiB** | 578 s   | 821 s               | 234        |

The setup time scales as $\mathcal{O}(N_\text{dof})$ (fitted exponent
1.01). The solve time per iteration grows from 1.0 s to 3.5 s (fitted
exponent 0.87), consistent with the AIM MVP cost
$\mathcal{O}(N_g \log N_g + N_\text{dof}\, n_\text{near})$ at fixed
wavelength. The total solve time grows faster (exponent 1.15) because
the iteration count rises from 167 to 234. With sub-300 MiB budgets
through $R/7$, refinement to $R/8$ used 415 MiB operator memory. The
process RSS was 3.2 GiB after the $R/8$ solve and peaked at 5.2 GiB
over the four levels, which run in one process. The projected
760 MiB operator memory at $R/10$ is feasible on the
same 16 GB machine.

The times of the earlier measurement (April 2026, Intel i7-1265U,
block BiCGSTAB, mesh without the volume rescale) were 105–410 s for the
setup and 201–925 s for the solve. The operator memory was the same
within 0.5 %.

---

## 9. Per-DoF accuracy vs block-DDA_Py

Driver: [benchmarks/cas_v2/low_contrast_sphere.jl](../benchmarks/cas_v2/low_contrast_sphere.jl)
(VIEM side) and the equivalent block-DDA_Py sweep on the same
sphere. The Mie analytical solution is the common reference.

### 9.1 Low-contrast: $m_p = 1.5 + 0.01\,i$, $r = 1\,\text{μm}$, $\lambda_0 = 10\,\text{μm}$, $x \approx 0.63$

Single orientation, Intel i7-1265U, single-threaded
(`t_setup` drops by ~ 2.5 × on a 4-thread machine after the
parallel setup added in v0.6.0).

| Code          | N DOF | $C_\text{abs}$ rel. err. | t_setup | t_solve | memory |
| ------------- | ----- | ------------------------ | ------- | ------- | ------ |
| block-DDA_Py  | 302   | 1.7 %  | < 0.1 s | < 0.1 s | 6 MB   |
| block-DDA_Py  | 4 419 | 0.8 %  | < 0.1 s | 0.1 s   | 64 MB  |
| block-VIEM.jl | 589   | 8.4 %  | 7.9 s   | 2.2 s   | 6 MB   |
| block-VIEM.jl | 1 986 | 3.6 %  | 23.9 s  | 0.7 s   | 63 MB  |
| block-VIEM.jl | 7 868 | 1.4 %  | 145.8 s | 2.7 s   | 991 MB |

DDA achieves 1.7 % accuracy with 302 dipoles in under 0.1 s. VIEM
needs ~ 2 000 DOFs and ~ 10 s of setup (4 threads) for comparable
accuracy. The DDA cubic lattice has an exact FFT-MVP with no
near-field precorrection, so its setup cost is effectively zero.
**Recommendation:** for $|m_p / m_m| < 2$ and moderate aspect
ratios, block-DDA_Py is the right tool.

### 9.2 High-contrast: $m_p = 3.17 + 0.16\,i$ (legacy "high"), same sphere

| Code          | N DOF | $C_\text{abs}$ rel. err. | convergence    |
| ------------- | ----- | ------------------------ | -------------- |
| block-DDA_Py  | 1 496 | 6.8 %                    | slow ($1/N^{0.5}$) |
| block-DDA_Py  | 4 759 | 4.2 %                    | slow ($1/N^{0.5}$) |
| block-VIEM.jl | 1 986 | 3.6 %                    | $\mathcal O(h^2)$ |
| block-VIEM.jl | 7 868 | 1.4 %                    | $\mathcal O(h^2)$ |

At $N \sim 5000$, DDA still has 4 % error while VIEM reaches 1.4 %.
For plasmonic Au ($m_p = 0.175 + 3.48\,i$, §2 above) VIEM
converges monotonically to sub-1 % at $N = 2134$ DOFs; DDA may not
converge at all in such regimes.

The four reasons VIEM dominates DDA at high contrast:

1. **Conforming geometry.** Tetrahedral meshes represent curved
   boundaries to $\mathcal O(\ell_c^2)$; cubic lattices stair-case
   them to $\mathcal O(d)$.
2. **H(div)-conforming basis.** Normal D-continuity is built into
   the SWG basis, not approximated via a polarisability
   prescription.
3. **Linear vector basis.** Each tet carries up to 4 SWG functions
   (linear in position) versus one point dipole per DDA element.
4. **No polarisability tuning.** VIEM solves the integral
   equation directly; DDA requires Clausius–Mossotti +
   $(2/3)(ka)^3$ radiative correction.

**Recommendation:** for $|m_p / m_m| > 2 - 3$, plasmonic metals,
extreme aspect ratios, or any problem where block-DDA_Py fails to
converge, use block-VIEM.jl.

### 9.3 Per-DoF efficiency on Mie sphere — direct comparison

Same sphere, single orientation, Mie reference for $C_\text{abs}$:

| Method                             | $N_\text{dof}$ | $C_\text{abs}$ rel. err. |
| ---------------------------------- | -------------- | ------------------------ |
| DDA, $d = 0.14\,\text{μm}$ | 4488           | 1.25 %                   |
| half-SWG VIEM, $\ell_c = 0.50\,\text{μm}$ | 589  | 0.21 % |
| half-SWG VIEM, $\ell_c = 0.18\,\text{μm}$ | 7868 | 0.15 % |

VIEM is **10–12× more accurate per DOF** than DDA on this test:
matching the 1.25 % DDA error needs only ~ 400 half-SWG DOFs vs
~ 4500 DDA dipoles, and at the same DOF count (~ 8000) the VIEM
error is an order of magnitude smaller.

### 9.4 Asymptotic cost model

| Quantity | block-DDA_Py | block-VIEM.jl (half-SWG + AIM) |
| --- | --- | --- |
| Element geometry | cubic lattice | unstructured tetrahedra |
| Green kernel on grid | analytic dyadic, $\mathcal O(N_g)$ | scalar + moment-matched dyadic, $\mathcal O(N_g)$ |
| Near-field precorrection | none (exact on lattice) | Duffy volumetric, $\mathcal O(N\,n_\text{near})$ |
| MVP cost | $\mathcal O(N_g \log N_g)$ | $\mathcal O(N_g \log N_g + N\,n_\text{near})$ |
| Multi-orientation batch | block GMRES on $L$ RHS | shared LU (small $N$) or block GMRES on $L$ RHS |
| Dominant memory | $\mathcal O(N_g)$ FFT kernel | $\mathcal O(N_g)$ FFT kernel + $\mathcal O(N\,n_\text{near})$ precorr. |

DDA has a strictly translation-invariant interaction matrix —
a single FFT on the doubled lattice captures the entire MVP
exactly. VIEM has no such property: each basis function sees a
different discrete support of neighbouring tets, and the AIM
moment matching is only accurate for pairs whose separation is
large compared with the element size. Near pairs are computed
directly via Duffy quadrature and stored as a sparse precorrection
matrix. For typical settings this near-field work dominates both
matrix assembly and per-iteration MVP cost at small to moderate
$N$; only for very large meshes does the $N_g \log N_g$ FFT cost
become comparable.

**Trade-off summary.** VIEM spends more work per unknown
(near-field Duffy + precorrection), but uses an order of
magnitude fewer unknowns to reach a given accuracy for
high-contrast particles with curved boundaries. For sphere-like,
low-contrast problems DDA's simplicity wins on wall-clock time;
for the high-contrast iron-oxide aggregates and faceted gold
nanoparticles targeted by this project, VIEM is the clear winner.

---

## 10. Paper-production ℓ_c convergence protocol

Driver: [viem_results/paper/run_lc_convergence.jl](../viem_results/paper/run_lc_convergence.jl).

For each `(shape, material)` combination in the paper-production
matrix (see [CLAUDE.md §2](../CLAUDE.md), [docs/paper_simulation_conditions_viem.md](paper_simulation_conditions_viem.md)),
the script sweeps **five mesh-size factors**

```text
factor ∈ {1.5, 1.0, 0.7, 0.5, 0.35} × adaptive_lc(...)
```

at the representative size $a_{eq} = 0.1\,\text{μm}$,
single orientation, and writes `convergence_<shape>_<material>.hdf5`.
The file has its own layout: a group `/target/lc_convergence/` with one
entry per factor (`lc_factor`, the mean edge length `lc`, `n_tet`,
`n_dof`, the observables and the cost fields) and the Mie reference in
`/target/reference/`. The shapes are sphere, oblate and GRE.

**Scope.** Shows where the production mesh lies on the convergence
curve. The production sweeps (`run_viem.jl`) use **factor = 1.0**, the
adaptive `ℓ_c` itself.

**Reproduce one combination:**

```bash
julia --project=. -t auto viem_results/paper/run_lc_convergence.jl \
    sphere n20
```

Numerical tables for each `(shape, material)` are kept inside the
HDF5 outputs and — when published — alongside the paper figure
that uses them. They are not duplicated in this document because
they carry no benchmark interpretation beyond the per-row
observable values; the convergence rates are read off the paper
figure (VIEM $\mathcal O(h^2)$ vs DDA $\mathcal O(h^{0.5})$
contrast — see §9).

---

## 11. Shifted block GMRES for refractive-index sweeps

Drivers: [benchmarks/shifted/run_shift_scaling.jl](../benchmarks/shifted/run_shift_scaling.jl)
(speed-up), [benchmarks/shifted/p3_compare_h5.jl](../benchmarks/shifted/p3_compare_h5.jl)
(sweep-driver output), `benchmarks/shifted/p0_*.jl` (feasibility).
Raw results: `benchmarks/shifted/results/`.
Formulation: `docs/theory_note.pdf`, Sec. 11.

For a homogeneous isotropic particle,
$\mathbf{Z}(m_p) = (\kappa/\epsilon_\text{bg})(\tau\mathbf{M} - \mathbf{K})$
with $\tau = 1/((m_p/m_m)^2 - 1)$. With right preconditioning by
$\mathbf{M}^{-1}$ all refractive indices are shifts of
$\mathbf{C} = \mathbf{K}\mathbf{M}^{-1}$ and are solved in one block
Arnoldi process (`solve_cas_v2_orientations_shifted`, `--shifted-ri` in
the sweep drivers).

### 11.1 Factorisation of the impedance matrix

Unit sphere test mesh (333 tetrahedra, 765 unknowns, 198 half-SWG
boundary functions), $k_0 = 2$,
$\epsilon_p \in \{2.25 + 0.03i,\ 9 + i,\ -12.1 + 1.22i\}$,
$\epsilon_\text{bg} \in \{1,\ 1.7689\}$. $\mathbf{K}$ is recovered
from each $\mathbf{Z}$ and compared in relative Frobenius norm.

| Operator | Largest difference of $\mathbf{K}$ between two $\epsilon_p$ |
| --- | --- |
| Dense $\mathbf{Z}$ (with and without symmetrisation) | 6.4e-15 |
| AIM operator, applied to a random block | 3.7e-15 |

### 11.2 Right preconditioning with the mass matrix

Block-GMRES iterations to tol = 1e-5, $\lambda_0 = 0.638$ μm, particle
in vacuum, default mesh rule. The oblate spheroid has $b/c = 3$.

| $m_p$ | Shape | $a_\text{eq}$ (μm) | $L$ | $\mathbf{Z}$ | $\mathbf{Z}\mathbf{M}^{-1}$ |
| --- | --- | --- | --- | --- | --- |
| 1.5 + 0.01i | sphere | 0.1 | 5 | 24 | 7 |
| 1.5 + 0.01i | sphere | 0.2 | 5 | 27 | 11 |
| 1.5 + 0.01i | oblate | 0.1 | 5 | 30 | 7 |
| 1.5 + 0.01i | oblate | 0.2 | 5 | 31 | 10 |
| 2.0 | sphere | 0.1 | 5 | 24 | 11 |
| 2.0 | sphere | 0.2 | 5 | 34 | 20 |
| 2.0 | oblate | 0.1 | 5 | 40 | 15 |
| 2.0 | oblate | 0.2 | 5 | 44 | 20 |
| 2.9 + 0.1i | sphere | 0.1 | 5 | 36 | 21 |
| 2.9 + 0.1i | oblate | 0.1 | 5 | 41 | 22 |
| 0.17525 + 3.4830i | sphere | 0.1 | 1 | > 200 (residual 6.3e-4) | 176 |
| 0.17525 + 3.4830i | sphere | 0.1 | 64 | 105 | 47 |

Cost of $\mathbf{M}^{-1}$ (sphere, $a_\text{eq} = 0.4$ μm, 8 cores of a
Xeon E5-2620 v4, block of $L = 13$ columns):

| $N$ | Cholesky factorisation (s) | Factor (GB) | Cholesky solve (s) | Jacobi-CG to 1e-13 (s) | Block product with $\mathbf{K}$ (s) |
| --- | --- | --- | --- | --- | --- |
| 10,966 | 0.03 | 0.01 | 0.020 | 0.052 | 0.32 |
| 98,958 | 0.89 | 0.21 | 0.30 | 0.97 | 8.2 |
| 398,808 | 12.1 | 1.81 | 1.72 | 6.6 | 8.9 |

### 11.3 Measured speed-up

Oblate spheroid, $b/c = 2$, in vacuum at $\lambda_0 = 0.638$ μm, mesh
$\ell_c = \min(\lambda_0/(10\,|m_p|_\max),\ 0.3c)$, tol = 1e-5,
8 threads on one processor of `supermicro-647` (2 × Xeon E5-2620 v4),
2026-11-18. Iron oxide: $N_m$ equispaced indices from $2.7 + 0.1i$ to
$3.1 + 0.1i$. Dielectric: from 1.35 to 1.70.

- **B**: one `solve_cas_v2_orientations` (`:aim_gmres`) per index. Mesh,
  AIM projection and mass matrix shared. Operator rebuilt per index.
  This is the default mode of `ironoxide_sweep_h5.jl`.
- **C**: one `solve_cas_v2_orientations_shifted` for all indices
  (`--shifted-ri`).

$n^Z$ and $n$ are the ranges of the iteration counts of B and C.
$T_B$ and $T_C$ are total wall times including the shared set-up and
the observables. The last column is the largest difference of the
forward amplitudes of B and C, relative to the larger amplitude.

| Family | $D_{ve}$ (μm) | $N$ | $L$ | $N_m$ | $n^Z$ | $n$ | $\sum n^Z / \max n$ | $T_B$ (s) | $T_C$ (s) | $T_B / T_C$ | B vs C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iron oxide | 0.35 | 25680 | 5 | 1 | 61 | 44 | 1.4 | 145 | 139 | 1.0 | 3.2e-06 |
| | | | | 3 | 46–61 | 32–44 | 3.6 | 422 | 141 | 3.0 | 3.2e-06 |
| | | | | 10 | 46–61 | 32–44 | 12.1 | 1400 | 141 | 9.9 | 3.2e-06 |
| | | | | 30 | — | 32–44 | — | — | 142 | — | — |
| iron oxide | 0.35 | 25680 | 13 | 1 | 47 | 33 | 1.4 | 159 | 151 | 1.1 | 1.8e-06 |
| | | | | 3 | 39–47 | 26–33 | 3.9 | 461 | 154 | 3.0 | 1.8e-06 |
| | | | | 10 | 39–47 | 26–33 | 12.8 | 1534 | 154 | 10.0 | 2.0e-06 |
| | | | | 30 | 39–47 | 26–33 | 38.5 | 4605 | 158 | 29.1 | 2.1e-06 |
| iron oxide | 0.35 | 25680 | 64 | 1 | 37 | 25 | 1.5 | 217 | 191 | 1.1 | 7.1e-07 |
| | | | | 3 | 32–37 | 20–25 | 4.1 | 622 | 192 | 3.2 | 7.1e-07 |
| | | | | 10 | 32–37 | 20–25 | 13.6 | 2055 | 197 | 10.4 | 7.1e-07 |
| | | | | 30 | — | 20–25 | — | — | 211 | — | — |
| iron oxide | 0.60 | 124049 | 13 | 1 | 90 | 68 | 1.3 | 1072 | 991 | 1.1 | 4.2e-06 |
| | | | | 3 | 64–90 | 47–68 | 3.4 | 3029 | 992 | 3.1 | 4.2e-06 |
| | | | | 10 | — | 47–68 | — | — | 996 | — | — |
| | | | | 30 | — | 47–68 | — | — | 1014 | — | — |
| dielectric | 0.60 | 21903 | 13 | 1 | 31 | 16 | 1.9 | 133 | 126 | 1.1 | 1.7e-06 |
| | | | | 3 | 28–31 | 10–16 | 5.5 | 394 | 126 | 3.1 | 1.7e-06 |
| | | | | 10 | 28–31 | 10–16 | 18.3 | 1294 | 126 | 10.3 | 1.7e-06 |
| | | | | 30 | 28–31 | 10–16 | 54.8 | 3895 | 130 | 30.0 | 1.7e-06 |

A dash marks a combination for which method B was not run.

Breakdown of $T_C$ at $N_m = 30$ (seconds) and peak RSS of the process:

| Family | $D_{ve}$ (μm) | $N$ | $L$ | Shared set-up | Operator | Cholesky | Krylov | Residual check | Observables | Total | Observables (%) | Peak RSS (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iron oxide | 0.35 | 25680 | 5 | 2.6 | 124.2 | 0.10 | 12.5 | 0.2 | 1.2 | 142.3 | 0.9 | 4.0 |
| iron oxide | 0.35 | 25680 | 13 | 2.6 | 130.1 | 0.08 | 20.3 | 0.5 | 2.4 | 158.4 | 1.5 | 4.5 |
| iron oxide | 0.35 | 25680 | 64 | 2.6 | 124.5 | 0.12 | 63.7 | 2.1 | 7.3 | 210.9 | 3.4 | 4.9 |
| iron oxide | 0.60 | 124049 | 13 | 7.3 | 709.2 | 0.92 | 269.1 | 3.1 | 8.7 | 1014.1 | 0.9 | 9.2 |
| dielectric | 0.60 | 21903 | 13 | 2.6 | 114.5 | 0.07 | 8.5 | 0.5 | 2.1 | 130.0 | 1.6 | 4.3 |

Findings:

- $T_C$ grows by 10 % ($L = 64$) or by 5 % or less (other cases) between
  $N_m = 1$ and $N_m = 30$. The gain over B is close to $N_m$.
- The operator build is 59–88 % of $T_C$. It is the term that B pays
  $N_m$ times.
- The observables take 3.4 % of $T_C$ or less at $N_m = 30$. Projecting
  the far-field functionals onto the Krylov basis, as block-DDA_Py
  does, is not needed at these sizes.
- Method P (B with `precond = :mass_right`, run for $N_m = 3$) has the
  same iteration counts as C for every index. Its forward amplitudes
  differ from those of C by at most 1.3e-14. It is 1.02–1.13 times
  faster than B.
- For iron oxide at $D_{ve} = 0.60$ μm, 30 separate preconditioned
  solves would need 1723 block iterations instead of 68. At the measured
  3.9 s per iteration this is about 6800 s, against 269 s for the shared
  process.
- The recomputed residual of the slowest index was below tol in all runs.

### 11.4 Sweep drivers with `--shifted-ri`

Record: [benchmarks/shifted/results/p3_driver_check.txt](../benchmarks/shifted/results/p3_driver_check.txt).
Differences are relative to the largest $|S|$ of the wavelength group.

| Driver | Comparison | Largest difference |
| --- | --- | --- |
| `ironoxide_sweep_h5.jl` | default mode, before (e74a75f) vs after | 0 (numbers, attributes and config hash) |
| `ironoxide_sweep_h5.jl` | default vs `--shifted-ri`, `--tol 1e-8` | 3.9e-10 |
| `ironoxide_sweep_h5.jl` | same with `--triaxial 1.5 --n-psi 2` | 2.8e-10 |
| `spheroid_sweep_h5.jl` | default mode, before (e74a75f) vs after | 0 (numbers, attributes and config hash) |
| `spheroid_sweep_h5.jl` | default vs `--shifted-ri`, tol 1e-5 | 1.5e-06 |
| `viem_results/run_viem.jl` | default mode, before (8dfceee) vs after | 0 (all datasets of `/target/simulated_data`) |
| `viem_results/run_viem.jl` | default vs `--shifted-ri`, tol 1e-8 | 7.8e-10 |
| `viem_results/run_viem.jl` | default vs `--shifted-ri`, tol 1e-5 | 1.5e-06 |

Iron-oxide settings: hematite, 2 sizes (0.05 and 0.10 μm), 2 aspect
ratios, 3 real indices, 2 wavelengths, $N$ = 4056–7079. One row of 3
indices took 150–300 s in the default mode and 50–90 s with
`--shifted-ri` (4 threads, `supermicro-647`).

`run_viem.jl` settings
([run_viem_shifted_check.txt](../benchmarks/shifted/results/run_viem_shifted_check.txt)):
2 wavelengths, refractive indices 1.5, 1.6, 1.7 + 0.02i and one
anisotropic entry (1.5, 1.5, 1.6), one spheroid ($L = 3$ solved) and one
triaxial ellipsoid ($L = 12$). The differences are relative to the
largest magnitude of each dataset and cover `C_abs`, `C_ext` and the
three amplitudes. The three isotropic entries of a slot are solved in
one batch and the anisotropic entry by the existing solver. The sum of
`t_total_s` over the 16 conditions was 1787 s in the default mode and
769 s with `--shifted-ri` (tol 1e-5, laptop, 4 threads, other runs in
parallel).

---

## Appendix A. Block-Krylov RHS-scaling diagnostic

Driver: [viem_results/paper/run_rhs_scaling.jl](../viem_results/paper/run_rhs_scaling.jl).

Not a benchmark per se — diagnostic for the multi-orientation
block-Krylov solver. Sweeps RHS count $L \in \{1, 2, 4, 8, 16,
32, 64, 128\}$ at one shape × material slot (worst-case mesh
shared across $L$ values, fixed RNG seed for nested
uniform-sphere orientations). Records per-iteration residual
trace, end-to-end wall time, and peak RSS per $L$ value into the
HDF5 `/target/rhs_scaling/gmres/` subgroup. Used to verify that
shared-Krylov amortisation pays off for the production $L = 100$
orientation grids.

---
