# Roadmap: 2% Accuracy on CAS-v2 Observables

**Goal**: Achieve ≤ 2% relative error on the CAS-v2 observables `S_fw_s`,
`S_fw_p`, `S_bak`, and derived cross sections `C_ext`, `C_abs`, `C_sca`,
validated against Mie theory for a dielectric sphere.

**Current status (2026-04-12)**:
- RT0 (SWG) gives ~17% C_abs error at `ka = 0.1–0.3` on a 677-tet coarse
  mesh — matches buff-em to within ~1%, confirmed as intrinsic SWG
  discretization error, not a bug.
- RT1 gives ~37% C_abs error that does NOT converge with mesh refinement —
  a BlockVIEM-specific bug yet to be diagnosed.
- AIM framework exists (`src/aim_*.jl`) but only the `SWGBasis` path is
  typed; correctness vs dense Z and scalability not yet benchmarked at the
  scale needed for 2% accuracy.

## Strategy

Two convergent paths to 2%:
1. **Fine-mesh RT0 with AIM** (shorter path to target, even if RT1 bug
   persists). Requires O(10k–100k) DOFs at reasonable cost → AIM
   acceleration is essential.
2. **Correctly working RT1** (longer but more efficient). Higher-order
   convergence lets us hit 2% with fewer DOFs once the bug is fixed.

AIM is prioritized because it unblocks (1) directly and is a prerequisite
for scaling RT1 tests once (2) is diagnosed.

## Target observable behavior

| Observable | Convergence driver | Sensitivity |
|---|---|---|
| `C_abs` | Interior L² norm of `D` | Quadratic in `D` error |
| `S_fw` | Far-field projection of `D` | Linear in `D` error; Im part prone to cancellation |
| `S_bak` | Same as `S_fw` at `-k̂` | Same |
| `C_ext` (OT) | `Im(S_fw)` via optical theorem | Inherits `Im(S_fw)` accuracy |
| `C_sca` | Either `C_ext − C_abs` or far-field integral | Multiple paths |

For complex amplitudes, the real and imaginary parts have different
convergence rates (imag typically worse due to phase cancellation). The
most demanding observable is usually `Im(S_fw)` at small `ka`.

## Phase breakdown

### Phase A — AIM validation and baseline (PRIORITY)

**Goal**: make the AIM path trustworthy and fast enough to run fine meshes
in seconds, so that the rest of the roadmap is feasible.

**A.1** *Audit current AIM code.* Enumerate: what parts accept
`AbstractDivBasis`, what parts assume `SWGBasis`, what the moment-matching
order is, how near-pair precorrection is built.

**A.2** *AIM = dense Z for RT0 on tiny mesh.* Fix a bipyramid / unit-cube
mesh. Build dense `Z` and AIM operator with the same parameters. Check
`‖Z·x − AIM·x‖ / ‖Z·x‖ < 1e-6` for random `x`. Any gap here is a bug.

**A.3** *AIM convergence vs grid pitch and padding.* On the buff-em
Sphere_677 mesh, run AIM at a few grid pitches (fraction of mean edge
length) and padding values; compare `D†MD` and `S_fw` against the dense
solution. Pick default parameters that give `< 0.5%` AIM-vs-dense error.

**A.4** *AIM scalability benchmark.* Time AIM MVP on Sphere_677, 1675,
and a self-generated `lc = 0.15` mesh (expected ~5k–10k tets). Report
time-per-MVP and memory. This tells us what mesh sizes are tractable.

**A.5** *AIM + BiCGSTAB integration test.* Solve the full VIEM system
iteratively via `solve_iterative`. Confirm residual `< 1e-6` in `<100`
iterations at the pitch chosen in A.3.

**Acceptance for Phase A**: AIM reproduces dense results to within 0.5%
and solves a 10k-DOF system iteratively in <1 min. This makes subsequent
convergence studies affordable.

### Phase B — RT0 mesh convergence with AIM

**Goal**: establish whether fine-mesh RT0 alone can reach 2% on CAS-v2
observables, and identify the mesh size required.

**B.1** *Mesh-refinement study on Mie sphere* (`ε = 10 + 1i`, `ka = 0.1,
0.316, 0.631`). Mesh sizes `lc ∈ {0.5, 0.35, 0.25, 0.175, 0.125}`. Record
`C_abs, C_sca, C_ext, S_fw_s, S_fw_p, S_bak` and their relative errors vs
Mie. Fit `err = C h^p` to extract the convergence rate `p`. Expected
`p ≈ 1` for RT0.

**B.2** *Low-contrast case* (`m = 1.5 + 0.01i`) and iron-oxide case
(`m = 2.5 + 0.5i`) at the same mesh sizes — RT0 behavior differs with
contrast due to ill-conditioning.

**B.3** *Extrapolate to 2% target*. From the fitted rate `p`, estimate the
mesh size required. If `p = 1` and current `lc = 0.25` gives ~12% error,
reaching 2% requires `lc ≈ 0.04` (~40× more tets). Use this as the upper
bound on DOF count for AIM scaling tests.

**Acceptance for Phase B**: A documented `(mesh, k0)` operating point
where all CAS-v2 observables have `< 2%` error on a validated sphere,
using only RT0 + AIM. This is the minimum viable path to the target.

### Phase C — RT1 bug diagnosis and repair

**Goal**: enable RT1's higher convergence rate so that 2% can be hit with
~10× fewer DOFs than pure RT0. This halves the memory footprint and
dramatically shortens sweep times.

**C.1** *Isolate the RT1 regression.* Set up a single-tetrahedron test
where the VIEM reduces to a small dense system. Verify that mass and
radiation kernel elements match an analytical or symbolic reference for
RT1 basis functions. This is the cheapest diagnostic.

**C.2** *Spectral analysis of `Z` for RT1 vs RT0.* Compare eigenvalues of
`M`, `K`, and `M⁻¹K`. The RT1 anomaly (`cond(Z_RT1) ≈ 290` vs
`cond(Z_RT0) ≈ 10` on identical meshes) must have a spectral signature.

**C.3** *Verify reference-element integrals.* Compute `∫ φ̂_i · φ̂_j dV̂`
and `∫ (∇·φ̂_i)(∇·φ̂_j) dV̂` on the reference tet with exact quadrature
(or symbolically via a CAS). Compare with what `_mass_term` and
`_radiation_kernel` produce for a physically identical tet. Any
discrepancy pinpoints the bug.

**C.4** *Fix and re-run Phase B.* After the fix, re-run the mesh-
refinement study with RT1 to confirm `p ≈ 2` convergence and quantify
the DOF savings.

**Acceptance for Phase C**: RT1 passes the same 2% target with
significantly fewer DOFs than RT0, confirming the expected higher-order
convergence.

### Phase D — Accurate CAS-v2 postprocessing

**Goal**: make sure the far-field amplitude and optical theorem
evaluations do not limit accuracy below the 2% target.

**D.1** *Far-field amplitude quadrature study.* Vary the volume
quadrature used in `far_field_amplitude` and confirm `S_fw, S_bak`
converge as mesh is refined (not as the volume quadrature order
changes). Current default is `TET_QUAD_5PT` (degree 3) — insufficient
for RT1, and possibly marginal for fine-mesh RT0 at larger `ka`.

**D.2** *Optical theorem vs direct absorption.* At the target mesh,
compare `C_ext` from the optical theorem (`Im(S_fw)`) against
`C_abs + C_sca_far_field`. Discrepancy indicates which path is more
accurate.

**D.3** *Cancellation-sensitive regimes.* At small `ka` with low loss,
`Re(S_fw) ≫ Im(S_fw)`. Verify the imaginary part is computed with
sufficient precision (double-precision arithmetic may be insufficient
if intermediate quantities cancel).

**D.4** *Multi-orientation / block-Krylov path.* The CAS-v2 workflow
needs many incidence directions. Implement or validate the
block-Krylov RHS path in `solver.jl` so that the 2% target extends
from a single plane wave to an orientation sweep.

**Acceptance for Phase D**: CAS-v2 observables match Mie to `<2%` at
the mesh size determined in B.3 (or C.4), AND the optical theorem
and direct methods agree to within 1%.

### Phase E — Integration test against block-DDA_Py

**Goal**: confirm the 2% target holds on a non-spherical geometry
where block-DDA_Py (validated independently) provides the reference.

**E.1** *Spheroid or iron-oxide cluster.* Generate a mesh via the
GRE-to-msh pipeline. Run both BlockVIEM and block-DDA_Py. Compare
`S_fw_s`, `S_fw_p`, `S_bak`. Tolerance: 2% on magnitude and phase.

**E.2** *Multi-wavelength sweep.* Run a small sweep (3–5 wavelengths)
on the same target. Confirm `S_fw(λ)` dispersion is captured.

**Acceptance for Phase E**: BlockVIEM.jl can be used as a drop-in
replacement for block-DDA_Py for CAS-v2 at 2% tolerance, validated on
at least one non-trivial geometry.

## Priority ordering

The user has specified that AIM validation is the top priority (for
verification-time reduction). Execution order:

1. **Phase A** — AIM audit + validation (A.1–A.5) — IN PROGRESS
2. **Phase B** — RT0 fine-mesh convergence with AIM
3. **Phase D.1, D.2** — postprocessing quadrature (can overlap with B)
4. **Phase C** — RT1 bug diagnosis (defer if B already reaches 2%)
5. **Phase D.3, D.4** — cancellation and block-Krylov
6. **Phase E** — integration test

## Success criteria

- All phases except E: automated tests in `test/` that pass at 2% on the
  designated sphere meshes.
- Phase E: a reproducible script in `benchmarks/scripts/` that prints a
  pass/fail table against block-DDA_Py on at least one spheroid target.
- Documentation: this roadmap is kept live with status markers (`[done]`,
  `[in progress]`, `[blocked]`) as each phase completes.
