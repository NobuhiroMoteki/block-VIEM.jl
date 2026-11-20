# Changelog

Changes to block-VIEM.jl, newest first. Dates are year-month-day.

The first section lists the corrections that change numerical results. Read it if you
have results from an earlier version. The sections after it list all changes by version.

## Corrections that change results

| Fixed in | Affected versions | What was wrong | Results that change | What to do |
|---|---|---|---|---|
| 0.8.2 | 0.7.7 to 0.8.1 | The volume rescale of `mesh_sphere_aggregate` scaled the whole mesh about the origin. For an aggregate of spheres that do not touch, this also scaled the centre distances. | All observables of aggregates with disconnected monomers, such as the two-sphere doublet with a gap. With `overlap_factor = 0` the centre distance of the doublet was larger than the nominal `2.1 R` by the scale factor, up to 1.4 % (`2.13 R`) for the meshes of the paper runs. Connected aggregates and single particles are not affected. | Recompute. The files `viem_results/paper/doublet_*.hdf5` were regenerated with 0.8.2. |
| 0.8.1 | 0.7.0 to 0.8.0 | The production runner `viem_results/run_viem.jl` (also `run_rhs_scaling.jl` and `estimate_cost.jl`) meshed the doublet with the default `overlap_factor = 0.02`, which enlarges the radii by 2 %. From 0.7.7, with the volume rescale, the meshed gap was `0.0595 R` and not the documented `0.1 R` (centre distance `2.06 R` to `2.08 R`). | Doublet files written by the production runner. Against the `0.1 R` geometry the forward amplitudes differ by 0.5 to 2 % and `S_bk` by 3 to 28 %. For Au the difference of `C_ext` from the MSTM reference was 5 to 13 % and is 1.5 to 2 % with 0.8.2. | Recompute with 0.8.2 or later. The paper doublet files were regenerated with 0.8.2. |
| 0.8.1 | 0.4.0 to 0.8.0 | On the AIM path (`:aim_gmres`, `:aim_bicgstab`) the mass term of an anisotropic particle used the mean of `1/ε_α` in place of the weight `1/ε_α` of each Cartesian component. | All observables of particles with an anisotropic permittivity computed on the AIM path. They were off by 5 to 25 %. The dense solver (`:dense`) was correct. Isotropic particles are not affected. | Recompute anisotropic results that were made on the AIM path, for example the tables of `ironoxide_sweep_h5.jl --form-biref` and `--biaxial`. |
| 0.8.0 | all before 0.8.0 | The backward amplitude `S_bk` had the factor `1/√2` of the printed Eq. (28) of Moteki and Adachi (2024). The corrected equation, given in the erratum to that paper, has `1/2`. | `S_bk` (`S_bk_OCBS` in the HDF5 files). The old value is `√2` times the correct one, with the same phase. The forward amplitudes and the cross sections are not affected. | Divide old `S_bk` values by `√2`. The files under `viem_results/paper/` hold the corrected form (see 0.8.3). |
| 0.7.9 | 0.7.8 | The reference runner `viem_results/paper/run_mstm_reference.jl` computed `(S11 + S22 + i S12 − i S21)/√2` at the backward direction in place of the OCBS amplitude `S_bk`. | `S_bk` in the MSTM reference files `mstm_doublet_*.hdf5`. The values computed by block-VIEM.jl are not affected. | Use the reference files of 0.7.9 or later. |
| 0.7.7 | all before 0.7.7 | The mesh was used as Gmsh returned it. The volume of a faceted curved particle is smaller than the nominal volume, by 0.5 to 3 % depending on the mesh size. From 0.7.7 the mesh is scaled to the nominal volume. | All observables of curved particles. In the Rayleigh regime `C_ext`, `C_abs` and the amplitudes were too small by about the volume deficit. For the Au doublet of the README benchmark the errors were larger (up to 10.5 % at mesh size `R/5`, 2.3 % with 0.8.2). | Recompute, or treat old values as biased low by the volume deficit of their mesh. |
| 0.3.0 | 0.2.0 and earlier | The wavenumber of the incident wave was multiplied by the background index a second time. | All results with a background medium other than vacuum (`eps_bg ≠ 1`). Results in vacuum are not affected. | Recompute. |
| 0.1.1 | 0.1.0 and earlier | The Green function and the far-field integral followed the time convention `exp(+iωt)`, the permittivity the convention `exp(−iωt)`. | The imaginary part of the forward amplitude of absorbing particles (error of about 38 %). Results for lossless particles were correct. | Recompute. |
| 0.1.0 | 0.0.1 | The basis had no functions on the boundary faces, which forces the normal component of **D** to zero at the particle surface. | All observables. The cross sections of a sphere differed from the Mie values by 10 to 29 % and did not converge with the mesh size. | Recompute. |

## 0.8.4 (2026-11-20)

Same code in `src/` and same tests as 0.8.2 and 0.8.3. The documents are corrected.

- This changelog is added.
- The explanation of the earlier doublet benchmark results is corrected. The documents of
  0.8.3 said that the earlier errors (up to 10.5 % for Au) came from a centre distance
  that was about 1 % too large. That was wrong. The earlier results date from April 2026
  (0.4.2 to 0.5.0), before the volume rescale of 0.7.7. Their meshes had the nominal
  centre distance and gap, and a volume 1.40 % (mesh size `R/5`) to 0.53 % (`R/8`) below
  the nominal volume. Solving the unscaled mesh at `R/5` with the present solver
  reproduces the earlier errors (Au, β = π/2: 4.36 % on `|S_fw_mean|` and 10.62 % on
  Im `S_fw_θ`).
- The documents of 0.8.3 also said that the earlier convergence rate `p ≈ 2` was
  withdrawn. The rate was a correct measurement of the geometric error of the unscaled
  mesh, which decreases as the square of the mesh size. With the volume rescale that
  error is mostly removed, and the remaining errors (0.41 % or less from `R/6`) do not
  decrease monotonically.
- Corrected in: README, `docs/benchmark_results.md` (sections 7 and 8), the theory note,
  and the notes of the GitHub release 0.8.3. The numbers of the re-run of 0.8.3 are
  unchanged.

## 0.8.3 (2026-11-20)

Same code in `src/` as 0.8.2. First GitHub release since 0.7.9.

- The doublet benchmark against MSTM (`benchmarks/cas_v2/doublet_mstm/`) was run again
  with the mesher of 0.8.2. At mesh size `R/5` the forward amplitudes agree with MSTM
  within 0.5 % for the dielectric doublet and within 2.3 % for the Au doublet. At `R/6`
  to `R/8` every component of the Au doublet is within 0.41 %.
- The benchmark scripts use AIM block GMRES with the mass-matrix preconditioner
  (`config.jl`: `VIEM_METHOD`, `VIEM_PRECOND`) and record the iteration count, the
  residual and the convergence flag. Block BiCGSTAB, the earlier setting, does not
  converge for the Au doublet (residual 8e-3 after 400 iterations). Block GMRES without a
  preconditioner needs 621 iterations at `R/5`, and 167 with the preconditioner. The two
  converged solutions agree to eight digits.
- `compare_refinement.jl` fits an exponent only from three or more refinement levels.
  `compute_refinement_errors.py` marks the components whose error is not monotone.
- Section 8.3 of `docs/benchmark_results.md` (memory and timing) holds a new
  measurement.
- Stored data: in all files under `viem_results/paper/` except the doublet files, the
  stored backward amplitudes were multiplied by `1/√2`, without recomputation. The
  attributes `S_definition` or `S_bk_definition` and `S_bk_conversion_note` say so. The
  files as produced by 0.7.8 are in the git history (commit `416cece`).

The explanation that these documents gave for the earlier benchmark errors was wrong and
is corrected in 0.8.4.

## 0.8.2 (2026-11-19)

### Fixed

- `mesh_sphere_aggregate` scales each disconnected piece about its own centre to its
  nominal volume. See the first row of the table above. Connected aggregates are scaled
  about the origin as before.

### Changed

- The files `viem_results/paper/doublet_*.hdf5` were regenerated (centre distance
  `2.1 R`, attribute `doublet_gap_over_R = 0.1`, corrected `S_bk`).

## 0.8.1 (2026-11-19)

### Fixed

- AIM path for anisotropic particles: the mass term is `Σ_α (1/ε_α) M_α`. See the third
  row of the table above. `test/test_anisotropic.jl` compares the AIM path with the dense
  solver.
- The production runner meshes the doublet with `overlap_factor = 0`. See the second row
  of the table above. `run_viem.jl` writes the attribute `doublet_gap_over_R` and refuses
  to append to a doublet file that holds results without it.
- `run_viem.jl` continues with the next condition after a solver failure. It stopped
  after writing NaN.

### Changed

- The documents are brought in line with the code.

## 0.8.0 (2026-11-19)

### Added

- Refractive-index sweep with one Krylov process: `solve_cas_v2_orientations_shifted`
  solves all refractive indices of an isotropic, homogeneous particle with one shifted
  block GMRES (`block_gmres_shifted`). The drivers `viem_results/run_viem.jl`,
  `benchmarks/cas_v2/ironoxide_sweep_h5.jl` and `spheroid_sweep_h5.jl` accept
  `--shifted-ri`. The measured speed-up is about 3, 10 and 29 to 30 for 3, 10 and 30
  refractive indices. Default off. The output without the option is unchanged.
- `precond = :mass_right`: right preconditioning with the mass matrix in the ordinary
  block solve. Default `:none`.
- `benchmarks/shifted/`: feasibility scripts and the scaling benchmark of the shifted
  sweep. Theory note, Sec. 11.

### Fixed

- `S_bk` follows the corrected Eq. (28) of Moteki and Adachi (2024). See the fourth row
  of the table above. `run_viem.jl` refuses to append to a file that holds results in the
  earlier form.

### Added since 0.7.9 without a version of their own

- Spheroid-sweep HDF5 writer: a `/provenance` group with the configuration hash, and a
  command line (`--preset`, `--output`) (2026-07-05). File layout that the reader of the
  lookup tables expects: logarithmic size axis, C-order arrays, imaginary index per
  wavelength (2026-09-01).
- `benchmarks/cas_v2/ironoxide_sweep_h5.jl`: sweep for high-contrast iron-oxide particles
  (2026-09-01), with `--degenerate-ri` (2026-09-03), a uniaxial effective medium of
  aligned needles (`--form-biref`, 2026-09-08), biaxial particles with a spin axis
  (`--biaxial`, 2026-09-11), triaxial ellipsoids (`--triaxial`, 2026-09-14) and band
  selection (`--wavelengths`, 2026-10-14). The anisotropic tables made before 0.8.1 are
  affected by the third row of the table above.
- Faster setup of the AIM near-field correction and a threaded near-field product for
  block right-hand sides (2026-10-15). The results are bitwise identical.

## 0.7.9 (2026-05-01 to 2026-05-13)

GitHub release "Paper-production version". It was used for the VIEM results of the
preprint "Block-Krylov discrete-dipole and volume-integral-equation solvers for
multi-orientation light scattering across the dielectric–metal contrast range".

- Reference runners: corrected backward amplitude in the MSTM reference (see the fifth
  row of the table above). New T-matrix (EBCM) reference runners for the oblate spheroid,
  with `C_sca` and `C_abs` at fixed orientation.
- Mesh-size convergence runs: block size and iteration limit can be set by environment
  variables. The Au runs for the oblate spheroid and the GRE were regenerated.
- `benchmarks/cas_v2/low_contrast_sphere.jl`: low-contrast sphere benchmark.
- `viz/`: panel figure of the paper targets.

## 0.7.8 (2026-04-28)

- The paper-production sweep is complete: 26 HDF5 files under `viem_results/paper/`
  (sphere, doublet, oblate spheroid and GRE, three materials each, with diagnostics).
  The Au slots that reach the iteration limit are marked by `/target/cost/converged`.
- Runner fixes: `check_h5.jl` reads files with the `/target/cost/` group,
  `run_lc_convergence.jl` accepts a list of mesh factors, `run_rhs_scaling.jl` runs FFTW
  with one thread to avoid a crash.
- Shell scripts that run the phases of the paper production.
- New documents: `docs/descriptions_particle_shape_model.md`, `Installation_guide.md`,
  `docs/benchmark_results.md` (2026-04-25).

## 0.7.7 (2026-04-24)

- Volume rescale: `gre_mesh`, `gre_mesh_with_field` and `mesh_sphere_aggregate` scale the
  mesh to the nominal volume (`apply_scale!`). See the sixth row of the table above.
  `mesh_sphere_aggregate` has the keyword `rescale_to_target_volume` (default `true`).
- New document: `docs/paper_simulation_conditions_viem.md`.

## 0.7.6 (2026-04-24)

- The paper runners use at most 200 iterations (100 before).
- The RHS-scaling diagnostic measures block GMRES only.

## 0.7.5 (2026-04-24)

- `BlockSolveResult.residual_history`: residual at every iteration, also written to the
  HDF5 files.
- The "high" material of the paper runs is `2.0 + 0.0i` (`n20`). It was `3.17 + 0.16i`.

## 0.7.4 (2026-04-24)

- `block_gmres` updates the QR factorisation of the block Hessenberg matrix at every
  iteration. The cost per iteration no longer grows with the cube of the Krylov
  dimension. The results are the same.

## 0.7.3 (2026-04-22)

- `/target/cost/` group in the HDF5 files: times, peak memory, mesh size, iteration count
  and convergence flag of every slot. `run_viem.jl` refuses files without the group.

## 0.7.2 (2026-04-21)

- Default tolerance `1e-5` and default iteration limit 100 for the block solvers,
  `solve_iterative` and `solve_cas_v2_orientations`. They were `1e-6` and 200 or 400.
  Results computed with the defaults change within the tolerance.

## 0.7.1 (2026-04-21)

- The default solver of `solve_cas_v2_orientations` and of the runners is block GMRES
  (`:aim_gmres`). It was block BiCGSTAB, which remains available as `:aim_bicgstab`.
- The RHS-scaling diagnostic covers 1 to 128 right-hand sides.
- The largest volume-equivalent radius of the paper runs is 0.4 μm for the dielectric
  materials.

## 0.7.0 (2026-04-21)

- Scripts for the paper production under `viem_results/paper/`: templates per shape and
  material, the cost estimator, the RHS-scaling diagnostic, the MSTM reference for the
  doublet and the mesh-size convergence study.
- `run_viem.jl` handles the two-sphere doublet (`shape_kind = "doublet"`).
- `solve_cas_v2_orientations(...; return_solve_info = true)` returns the iteration count,
  the residual and the convergence flag.

## 0.6.0 (2026-04-19)

- Threaded AIM setup and matrix-vector product. `set_fft_threads` sets the number of FFTW
  threads.
- The projection and the mass matrix of a mesh can be reused for several wavelengths and
  refractive indices. `run_viem.jl` does so.

## 0.5.0 (2026-04-19)

- The boundary-face terms of the half-SWG basis are part of the FFT path of the AIM
  (surface-moment projection). The memory of the AIM operator grows about linearly with
  the number of unknowns. It grew with the power 5/3 before.
- Symmetric storage of the near-field correction, which halves its memory.

## 0.4.1 to 0.4.3 (2026-04-18 to 2026-04-19)

- `src/aggregate_mesh.jl`: `SphereAggregate` and `mesh_sphere_aggregate` for aggregates
  of spheres.
- Doublet benchmark against MSTMforCAS.jl (`benchmarks/cas_v2/doublet_mstm/`). The MSTM
  reference uses the truncation order 6 in 0.4.1 and 15 from 0.4.2. 0.4.3 adds a mesh
  refinement study. These results were replaced in 0.8.3.
- `viem_results/run_viem.jl`: production sweep over an HDF5 file. The default solver of
  `solve_cas_v2_orientations` is AIM block BiCGSTAB, with the grid pitch chosen from the
  mesh (2026-04-17).
- `CASv2Result.S_fw` is renamed to `S_fw_mean` (2026-04-18).
- `viz/`: wireframe figures of GRE particles, in an environment of its own.

## 0.4.0 (2026-04-17)

- Anisotropic permittivity `eps_p = diag(eps_x, eps_y, eps_z)`. On the AIM path the
  results were wrong until 0.8.1 (third row of the table above).

## 0.3.1 (2026-04-16)

- `viem_results/create_h5.jl` and `check_h5.jl`: sweep files in the HDF5 layout of
  block-DDA_Py.

## 0.3.0 (2026-04-16)

- Gaussian random ellipsoid (GRE) shape model, compared with block-DDA_Py.
- `solve_cas_v2_orientations` accepts the physical inputs `wl_0`, `m_m`, `m_p` of
  block-DDA_Py.
- Fixed: wavenumber of the incident wave in a background medium. See the seventh row of
  the table above.
- Mie validation for a plasmonic Au sphere.

## 0.2.0 (2026-04-13)

- Block Krylov solvers on the AIM operator (`block_bicgstab`, `block_gmres`): all
  orientations of a particle are solved together.
- Theory note (`docs/theory_note.pdf`). Comparison of spheroid amplitudes with
  block-DDA_Py.

## 0.1.2 (2026-04-13)

- HDF5 output of spheroid sweeps in the layout of block-DDA_Py.

## 0.1.1 (2026-04-13)

- CAS-v2 observables: forward amplitudes `S_fw_θ`, `S_fw_φ` and backward amplitude
  `S_bk`. Solve for many orientations. Oblate spheroid benchmark.
- Fixed: the code follows the time convention `exp(−iωt)` throughout. See the eighth row
  of the table above.

## 0.1.0 (2026-04-12)

- Half-SWG basis: basis functions on the boundary faces with the surface terms of the
  weak form. See the last row of the table above.

## 0.0.1 (2026-04-10 to 2026-04-11)

- First implementation: tetrahedral mesh and Gmsh reader, SWG basis, Duffy transformation
  for the singular integrals, assembly of the impedance matrix, AIM with FFT, Krylov
  solver, far field and cross sections, Mie validation.
- RT1 basis (second-order Raviart–Thomas). It is not used in production.

## Tags and releases

Git tags exist for 0.1.0 to 0.6.0 (except 0.3.1), 0.7.9 and 0.8.3 onwards. The versions
0.7.0 to 0.7.8 and 0.8.0 to 0.8.2 have no tag. Their commits are found by the version in
`Project.toml`. GitHub releases exist for 0.7.9 and from 0.8.3.
