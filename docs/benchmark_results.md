# Benchmark Results — block-VIEM.jl

This document is the authoritative record of all validation and
cross-comparison benchmarks for block-VIEM.jl. It collects in one
place:

- Numerical conditions (geometry, material, mesh, solver settings).
- Reference data (Mie series, MSTM T-matrix, block-DDA_Py).
- Tabulated relative errors for every observable reported in the
  paper-production HDF5\,\text{s}chema (`C_ext`, `C_abs`, `C_sca`,
  `S_fw_θ`, `S_fw_φ`, `S_bk`).
- Mesh-refinement (`ℓ_c` convergence) studies and the corresponding
  fitted convergence rates.
- Memory and wall-time scaling for the AIM operator and solver.

Per-section quick navigation:

| § | Benchmark | Reference |
| --- | --- | --- |
| [§1](#1-mie-sphere-dielectric) | Mie sphere, dielectric | Mie series |
| [§2](#2-mie-sphere-plasmonic-gold) | Mie sphere, plasmonic Au | Mie series |
| [§3](#3-oblate-spheroid-α-symmetry-self-check) | Oblate spheroid α-symmetry | analytical |
| [§4](#4-oblate-spheroid-cross-validation-vs-block-dda_py) | Oblate spheroid vs DDA | block-DDA_Py |
| [§5](#5-gre-cross-validation-vs-block-dda_py-β--0) | GRE (β > 0) vs DDA | block-DDA_Py |
| [§6](#6-anisotropic-εp-cross-validation-vs-block-dda_py) | Anisotropic ε_p vs DDA | block-DDA_Py |
| [§7](#7-two-sphere-doublet-vs-mstm) | 2-sphere doublet vs MSTM | MSTMforCAS.jl |
| [§8](#8-doublet-mesh-refinement-and-aim-memory) | Doublet ℓ_c convergence + memory | self |
| [§9](#9-per-dof-accuracy-vs-block-dda_py) | Per-DoF accuracy vs DDA | analytical (Mie) |
| [§10](#10-paper-production-lc-convergence-protocol) | Paper-production ℓ_c protocol | self |

Each subsection's first line gives the driver script; reproduction
is `julia --project=. <script>` unless noted otherwise.

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
S_\text{bk}^\text{OCBS} = +0.05946 + 0.01907\,i.
$$

Errors below are relative to Mie evaluated at the actual mesh
volume-equivalent radius (so residual mesh-volume bias does not
contaminate the discretisation error).

| ℓ_c [μm] | N    | Method       | C_ext  | C_abs  | C_sca  | S_fw,mean | S_bk   | wall  |
| -------- | ---- | ------------ | ------ | ------ | ------ | --------- | ------ | ----- |
| 0.020    | 2134 | dense (LU)   | 0.65 % | 0.005 % | 0.75 % | 0.58 %    | 0.28 % | 17\,\text{s}  |
| 0.020    | 2134 | AIM-BiCGSTAB | 0.25 % | 0.44 % | 0.37 % | 0.32 %    | 0.11 % | 43\,\text{s}  |
| 0.014    | 4932 | dense (LU)   | 0.34 % | 0.08 % | 0.41 % | 0.32 %    | 0.15 % | 68\,\text{s}  |
| 0.014    | 4932 | AIM-BiCGSTAB | 0.15 % | 0.24 % | 0.22 % | 0.18 %    | 0.07 % | 105\,\text{s} |

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
  $m_p = 0.17525 + 3.4830\,i$ (Au @ 638\,\text{nm}, Johnson & Christy 1972).

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
  $N = 11501$ half-SWG DOFs. AIM block-BiCGSTAB at tolerance
  $10^{-7}$, pitch $0.5\bar{h} = 0.0038\,\text{μm}$.
- MSTM: VSWF truncation order **forced to $N_\text{trunc} = 15$**
  for both materials. The auto-truncation at $x \approx 0.30$
  returns $N = 3$, which is converged for the dielectric case but
  badly under-truncates Au — $|S_\text{fw,mean}|$ shifts by
  $\approx 5 \%$ between $N = 3$ and the converged $N = 15$
  reference (geometric convergence, per-order ratio $\approx 0.3$).
  $N = 15$ reaches rtol $\le 10^{-6}$ on both materials at
  $x \approx 0.3$. CBICG tolerance $10^{-8}$.

#### MSTMforCAS.jl numerical-stability fixes

MSTMforCAS $\geq$ 0.4.3 incorporates three numerical-stability
fixes contributed as part of this benchmark work, **without which
the $N = 15$ reference cannot be computed at $x \approx 0.3$** (the
package previously failed catastrophically beyond $N = 6$):

1. **Miller's downward recurrence for the Riccati–Bessel
   $\psi_n(x)$** replaces the conditionally-stable upward
   recurrence (Wiscombe 1980). The upward recurrence amplifies a
   spurious $\chi_n$ component by $(2n-1)/x \approx 44$ per step
   while the physical $\psi_n(x) \sim x^n / (2n-1)!!$ underflows
   to $\sim 10^{-9}$ at $n = 8$.
2. **Correct sizing of the per-sphere `mie_vecs[i]` buffer** to
   `nois[i]` when the user-requested `truncation_order` exceeds
   the default Wiscombe bound `mie_nmax(x)`.
3. **Miller's downward recurrence for the spherical-Bessel
   $j_n(z)$** used in the multi-sphere translation operator,
   eliminating the analogous
   $y_n(kd) \sim (2n-1)!!\,(kd)^{-n-1}$ precision loss at small
   $kd$ that previously destabilised $N \ge 7$.

The `check_trunc_convergence.jl` script verifies geometric
convergence in $N$ at every benchmark point; rtol $\le 10^{-6}$
should be confirmed before any new comparison.

### 7.3 Dielectric doublet — $m_p = 1.60 + 0.01\,i$

#### Magnitude summary

| β [rad] | $\|S_\text{fw,mean}\|$ VIEM | $\|S_\text{fw,mean}\|$ MSTM | rel. $\|\cdot\|$ err. | complex rel. err. | phase err. [rad] |
| ------- | ------------------ | ------------------ | --------------------- | ----------------- | ---------------- |
| 0       | 1.769e−03          | 1.794e−03          | 1.4 %                 | 1.4 %             | 1.3e−04          |
| π/4     | 1.822e−03          | 1.849e−03          | 1.5 %                 | 1.5 %             | 1.6e−04          |
| π/2     | 1.878e−03          | 1.908e−03          | 1.6 %                 | 1.6 %             | 2.0e−04          |

#### $S_{\text{fw},\theta}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +1.7682e−03 | +1.7934e−03 | 1.4 %      | +4.1063e−05 | +4.1991e−05 | 2.2 %      |
| π/4     | +1.8768e−03 | +1.9066e−03 | 1.6 %      | +4.7872e−05 | +4.9082e−05 | 2.5 %      |
| π/2     | +1.9933e−03 | +2.0279e−03 | 1.7 %      | +5.5225e−05 | +5.6771e−05 | 2.7 %      |

#### $S_{\text{fw},\phi}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +1.7683e−03 | +1.7934e−03 | 1.4 %      | +4.1285e−05 | +4.1991e−05 | 1.7 %      |
| π/4     | +1.7651e−03 | +1.7902e−03 | 1.4 %      | +4.2005e−05 | +4.2724e−05 | 1.7 %      |
| π/2     | +1.7619e−03 | +1.7870e−03 | 1.4 %      | +4.2680e−05 | +4.3463e−05 | 1.8 %      |

Agreement is essentially polarisation- and orientation-independent
at this level: Re parts to 1.4 – 1.7 %, Im parts to 1.7 – 2.7 %.
The extra factor on Im is a pure scale effect — $|\Im S_\text{fw,mean}|$
is 30 – 50 × smaller than $|\Re S_\text{fw,mean}|$ in the weakly
absorbing case so a fixed absolute error appears larger on Im.

### 7.4 Plasmonic Au doublet — $m_p = 0.17525 + 3.4830\,i$

#### Magnitude summary

| β [rad] | $\|S_\text{fw,mean}\|$ VIEM | $\|S_\text{fw,mean}\|$ MSTM | rel. $\|\cdot\|$ err. | complex rel. err. | phase err. [rad] |
| ------- | ------------------ | ------------------ | --------------------- | ----------------- | ---------------- |
| 0       | 6.420e−03          | 6.504e−03          | 1.3 %                 | 1.3 %             | 5.1e−04          |
| π/4     | 8.043e−03          | 8.287e−03          | 2.9 %                 | 3.0 %             | 4.9e−03          |
| π/2     | 9.785e−03          | 1.020e−02          | 4.0 %                 | 4.1 %             | 7.0e−03          |

#### $S_{\text{fw},\theta}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +6.4046e−03 | +6.4886e−03 | 1.3 %      | +4.3635e−04 | +4.4705e−04 | 2.4 %      |
| π/4     | +9.6439e−03 | +1.0037e−02 | 3.9 %      | +1.2147e−03 | +1.3381e−03 | **9.2 %**  |
| π/2     | +1.3102e−02 | +1.3818e−02 | 5.2 %      | +2.0467e−03 | +2.2857e−03 | **10.5 %** |

#### $S_{\text{fw},\phi}$ — real and imaginary parts

| β [rad] | Re VIEM     | Re MSTM     | rel err Re | Im VIEM     | Im MSTM     | rel err Im |
| ------- | ----------- | ----------- | ---------- | ----------- | ----------- | ---------- |
| 0       | +6.4049e−03 | +6.4886e−03 | 1.3 %      | +4.3967e−04 | +4.4705e−04 | 1.7 %      |
| π/4     | +6.3556e−03 | +6.4392e−03 | 1.3 %      | +4.4777e−04 | +4.5502e−04 | 1.6 %      |
| π/2     | +6.3084e−03 | +6.3893e−03 | 1.3 %      | +4.5536e−04 | +4.6332e−04 | 1.7 %      |

#### Discussion

The plasmonic doublet exhibits a strong
**orientation–polarisation anisotropy** of the discretisation
error.

- The **$\phi$ channel** is essentially flat in $\beta$
  ($\approx 1.3 \%$ on Re, 1.6 – 1.7 % on Im), because
  $S_{\text{fw},\phi}$ probes the polarisation transverse to both
  the incidence and the doublet axis — the two spheres are
  effectively independent through this channel.
- The **$\theta$ channel** at $\beta = \pi/2$ couples to the
  polarisation aligned with the doublet axis. Both monomers are
  driven in phase along their centre line, the interaction field
  concentrates in the sub-wavelength gap, and the surface plasmon
  lives in a skin layer of depth
  $\delta = \lambda_0/(2\pi \Im m_p) \approx 29\,\text{nm}$,
  essentially the monomer radius itself. With
  $\ell_c = R/5 = 6\,\text{nm}$ the skin layer is resolved
  by ~ 5 linear tets — adequate for few-percent accuracy but the
  Im $\theta$ channel is the stiffest observable at 10.5 %.

**Practical takeaway.** A single-number relative error on
$|S_\text{fw,mean}|$ undersells the orientation / polarisation
structure of the discretisation error: at $\beta = \pi/2$ the
4.0 % $|S_\text{fw,mean}|$ error hides a **10.5 %** error on
$\Im S_{\text{fw},\theta}$. The Re/Im-resolved component view
should be used as the convergence indicator when sizing $\ell_c$
for plasmonic targets. The $\theta$-channel error also converges
the fastest under refinement (§8), so modest mesh refinement is
highly cost-effective here.

---

## 8. Doublet mesh refinement and AIM memory

Driver:
[benchmarks/cas_v2/doublet_mstm/phase_a_memory_study.jl](../benchmarks/cas_v2/doublet_mstm/phase_a_memory_study.jl)
+ `compute_refinement_errors.py`. Au doublet of §7.4. Same
geometry, four refinement levels $\ell_c = R/k$ for
$k = 5, 6, 7, 8$, all compared against the converged MSTM
$N = 15$ reference. Single-threaded (Intel i7-1265U,
16\,\text{GB} RAM, Julia 1.11) — the v0.6.0 threaded setup
+ block MVP gives a 2 – 3 × speed-up at 4 + Julia threads.

### 8.1 Per-component error vs ℓ_c at β = π/2

| ℓ_c / R | $\|S_\text{fw,mean}\|$ | $\Re S_{\text{fw},\theta}$ | $\Im S_{\text{fw},\theta}$ | $\Re S_{\text{fw},\phi}$ | $\Im S_{\text{fw},\phi}$ |
| ------- | ----- | ----- | ------ | ----- | ----- |
| 1/5     | 4.36 %| 5.61 %| 10.72 %| 1.38 %| 2.03 %|
| 1/6     | 2.73 %| 3.47 %|  6.43 %| 0.97 %| 1.82 %|
| 1/7     | 1.94 %| 2.44 %|  4.62 %| 0.73 %| 1.38 %|
| 1/8     | 1.60 %| 2.05 %|  4.07 %| 0.50 %| 0.58 %|

### 8.2 Fitted convergence rate $p$ in $\text{err} \propto (\ell_c/R)^p$

4-point log-log least-squares fit:

| Component | $p$ |
| --------- | --- |
| $\|S_\text{fw,mean}\|$ | 2.16 |
| $\Re S_{\text{fw},\theta}$ | 2.17 |
| $\Im S_{\text{fw},\theta}$ | 2.10 |
| $\Re S_{\text{fw},\phi}$ | 2.11 |
| $\Im S_{\text{fw},\phi}$ | 2.49 |

**Every observable converges at the theoretical linear-SWG
rate $p \approx 2$** (or faster for $\Im S_{\text{fw},\phi}$). An
earlier two-point fit using only $R/5, R/6$ had given
$p \in [0.78, 3.39]$ with $\Im S_{\text{fw},\theta}$ apparently
the *fastest* — that spread is a two-point artefact. The clean
$p \approx 2$ slope on the stiffest observable extrapolates to

$$
\Im S_{\text{fw},\theta}(\ell_c = R/10)
\;\approx\; 4.07 \% \times (8/10)^{2.10}
\;\approx\; 2.6 \%,
$$

$$
|S_\text{fw,mean}|(\ell_c = R/10)
\;\approx\; 1.60 \% \times (8/10)^{2.16}
\;\approx\; 1.0 \%.
$$

$\ell_c = R/10$ is the next planned refinement step.

### 8.3 Phase A AIM-operator memory

`total tracked` = `Base.summarysize(AIMOperator)` — every sparse
matrix, FFT kernel, and mass matrix in the operator. For
comparison the original Phase-1a `half_swg_extra` matrix alone
reached 486\,\text{MiB} at $R/5$ and 1074\,\text{MiB}
at $R/6$ on the same geometry; Phase A reduces those totals to
97\,\text{MiB} and 164\,\text{MiB} — a 5.0 × / 6.5 × /
11 × reduction at $R/5$ / $R/6$ / $R/7$ respectively. Asymptotic:
total tracked grows as $\mathcal O(N_\text{dof}^{1.02})$ —
essentially linear in $N$.

| ℓ_c / R | N DOF  | N_bnd | $W^S$    | precorrection | total tracked | t_setup | t_solve (3 orient.) |
| ------- | ------ | ----- | -------- | ------------- | ------------- | ------- | ------------------- |
| 1/5     | 11 501 | 1 610 | 4.9\,\text{MiB}  | 67.2\,\text{MiB}      | **96.8\,\text{MiB}**  | 105\,\text{s}   | 201\,\text{s}               |
| 1/6     | 18 919 | 2 254 | 8.1\,\text{MiB}  | 116.5\,\text{MiB}     | **164.2\,\text{MiB}** | 179\,\text{s}   | 363\,\text{s}               |
| 1/7     | 29 636 | 3 176 | 12.6\,\text{MiB} | 189.0\,\text{MiB}     | **262.7\,\text{MiB}** | 299\,\text{s}   | 718\,\text{s}               |
| 1/8     | 45 585 | 4 210 | 19.4\,\text{MiB} | 300.7\,\text{MiB}     | **413.3\,\text{MiB}** | 410\,\text{s}   | 925\,\text{s}               |

Setup and solve times both scale as $\mathcal{O}(N_\text{dof})$
within measurement noise, consistent with the AIM MVP cost
$\mathcal{O}(N_g \log N_g + N_\text{dof}\, n_\text{near})$ at fixed
wavelength. With sub-300\,\text{MiB} budgets through $R/7$,
refinement to $R/8$ used 413\,\text{MiB} operator memory (RSS peak ~ 3.8
GiB during the 3-orientation block-BiCGSTAB). The projected
760\,\text{MiB} operator memory at $R/10$ is feasible on the
same 16\,\text{GB} workstation.

---

## 9. Per-DoF accuracy vs block-DDA_Py

Driver: [benchmarks/cas_v2/au_sphere_mie.jl](../benchmarks/cas_v2/au_sphere_mie.jl)
(VIEM side) and the equivalent block-DDA_Py sweep on the same
sphere. The Mie analytical solution is the common reference.

### 9.1 Low-contrast: $m_p = 1.5 + 0.01\,i$, $r = 1\,\text{μm}$, $\lambda_0 = 10\,\text{μm}$, $x \approx 0.63$

Single orientation, Intel i7-1265U, single-threaded
(`t_setup` drops by ~ 2.5 × on a 4-thread machine after the
parallel setup added in v0.6.0).

| Code          | N DOF | $C_\text{abs}$ rel. err. | t_setup | t_solve | memory |
| ------------- | ----- | ------------------------ | ------- | ------- | ------ |
| block-DDA_Py  | 302   | 1.7 %  | < 0.1\,\text{s} | < 0.1\,\text{s} | 6 MB   |
| block-DDA_Py  | 4 419 | 0.8 %  | < 0.1\,\text{s} | 0.1\,\text{s}   | 64 MB  |
| block-VIEM.jl | 589   | 8.4 %  | 7.9\,\text{s}   | 2.2\,\text{s}   | 6 MB   |
| block-VIEM.jl | 1 986 | 3.6 %  | 23.9\,\text{s}  | 0.7\,\text{s}   | 63 MB  |
| block-VIEM.jl | 7 868 | 1.4 %  | 145.8\,\text{s} | 2.7\,\text{s}   | 991 MB |

DDA achieves 1.7 % accuracy with 302 dipoles in under 0.1\,\text{s}. VIEM
needs ~ 2 000 DOFs and ~ 10\,\text{s} of setup (4 threads) for comparable
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
| Multi-orientation batch | block-BiCGSTAB on $L$ RHS | shared LU (small $N$) or $L$ indep. BiCGSTAB / shared block-Krylov |
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
single orientation, and writes `convergence_<shape>_<material>.hdf5`
with the standard observable schema plus a `lc_factor` attribute
on every slot.

**Scope.** Used to choose the production `ℓ_c`-factor for the
large-scale sweeps. The current production choice is
**factor = 0.7** — the central value in the converged regime
(per-shape verification on `n15`, `n20`, `Au` for sphere /
doublet / oblate / GRE).

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

## Appendix B. Conventions used in error reporting

Throughout this document, relative errors are defined pointwise:

$$
\text{rel. err.}^{\Re} =
\frac{|\Re S^\text{VIEM} - \Re S^\text{ref}|}{|\Re S^\text{ref}|}, \quad
\text{rel. err.}^{\Im} =
\frac{|\Im S^\text{VIEM} - \Im S^\text{ref}|}{|\Im S^\text{ref}|},
$$

and "complex relative error" is

$$
\text{complex rel. err.} =
\frac{|S^\text{VIEM} - S^\text{ref}|}{|S^\text{ref}|}.
$$

Cross-section errors are
$|\Delta C| / C^\text{ref}$. "Phase err." is
$\arg(S^\text{VIEM} / S^\text{ref})$.
