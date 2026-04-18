# Mesh-refinement convergence: 2-sphere Au doublet

Refines the linear-SWG discretisation at a sequence of mesh sizes
`lc / R ∈ {R/5, R/6}` with the MSTM reference held fixed at
`N_trunc = 15` (converged to rtol ≤ 1e-6).  The relative errors of
the five CAS-v2 complex components (`|S_fw_mean|`, `Re/Im S_fw_θ`,
`Re/Im S_fw_φ`) are tabulated below for each β, followed by the
fitted convergence exponent `p` from `err ~ (lc/R)^p` (log-log fit).

A first-order linear-SWG basis is expected to give `p ≈ 2` on smooth
observables; a shallower slope indicates the error is dominated by
something other than the bulk mesh (e.g. boundary-layer resolution
of the plasmonic surface mode, geometric faceting).  Conversely a
slope `p > 2` is typical when the baseline error is already dominated
by higher-order (boundary-curvature) terms that cancel more rapidly
than the interior O(h²) contribution.


## β = 0.0000 rad (Au)

| lc / R | \|S_fw_mean\| err | Re S_fw_θ err | Im S_fw_θ err | Re S_fw_φ err | Im S_fw_φ err |
|--------|---------------------|---------------|---------------|---------------|---------------|
| 0.2000 | 1.30e-02 | 1.29e-02 | 2.39e-02 | 1.29e-02 | 1.65e-02 |
| 0.1667 | 8.76e-03 | 8.77e-03 | 1.40e-02 | 8.70e-03 | 1.38e-02 |

Fitted convergence exponent `p` from `err ~ (lc/R)^p`:

| component | p | err at smallest lc |
|-----------|---|--------------------|
| \|S_fw_mean\| | 2.14 | 8.76e-03 |
| Re S_fw_θ | 2.13 | 8.77e-03 |
| Im S_fw_θ | 2.96 | 1.40e-02 |
| Re S_fw_φ | 2.16 | 8.70e-03 |
| Im S_fw_φ | 1.00 | 1.38e-02 |

## β = 0.7854 rad (Au)

| lc / R | \|S_fw_mean\| err | Re S_fw_θ err | Im S_fw_θ err | Re S_fw_φ err | Im S_fw_φ err |
|--------|---------------------|---------------|---------------|---------------|---------------|
| 0.2000 | 2.94e-02 | 3.91e-02 | 9.22e-02 | 1.30e-02 | 1.59e-02 |
| 0.1667 | 1.76e-02 | 2.28e-02 | 4.90e-02 | 8.72e-03 | 1.50e-02 |

Fitted convergence exponent `p` from `err ~ (lc/R)^p`:

| component | p | err at smallest lc |
|-----------|---|--------------------|
| \|S_fw_mean\| | 2.83 | 1.76e-02 |
| Re S_fw_θ | 2.96 | 2.28e-02 |
| Im S_fw_θ | 3.46 | 4.90e-02 |
| Re S_fw_φ | 2.19 | 8.72e-03 |
| Im S_fw_φ | 0.33 | 1.50e-02 |

## β = 1.5708 rad (Au)

| lc / R | \|S_fw_mean\| err | Re S_fw_θ err | Im S_fw_θ err | Re S_fw_φ err | Im S_fw_φ err |
|--------|---------------------|---------------|---------------|---------------|---------------|
| 0.2000 | 4.03e-02 | 5.18e-02 | 1.05e-01 | 1.27e-02 | 1.72e-02 |
| 0.1667 | 2.35e-02 | 2.98e-02 | 5.64e-02 | 8.58e-03 | 1.49e-02 |

Fitted convergence exponent `p` from `err ~ (lc/R)^p`:

| component | p | err at smallest lc |
|-----------|---|--------------------|
| \|S_fw_mean\| | 2.95 | 2.35e-02 |
| Re S_fw_θ | 3.04 | 2.98e-02 |
| Im S_fw_θ | 3.39 | 5.64e-02 |
| Re S_fw_φ | 2.14 | 8.58e-03 |
| Im S_fw_φ | 0.78 | 1.49e-02 |

## Convergence summary

- β=0.0000, \|S_fw_mean\|: p = 2.14
- β=0.0000, Re S_fw_θ: p = 2.13
- β=0.0000, Im S_fw_θ: p = 2.96
- β=0.0000, Re S_fw_φ: p = 2.16
- β=0.0000, Im S_fw_φ: p = 1.00
- β=0.7854, \|S_fw_mean\|: p = 2.83
- β=0.7854, Re S_fw_θ: p = 2.96
- β=0.7854, Im S_fw_θ: p = 3.46
- β=0.7854, Re S_fw_φ: p = 2.19
- β=0.7854, Im S_fw_φ: p = 0.33
- β=1.5708, \|S_fw_mean\|: p = 2.95
- β=1.5708, Re S_fw_θ: p = 3.04
- β=1.5708, Im S_fw_θ: p = 3.39
- β=1.5708, Re S_fw_φ: p = 2.14
- β=1.5708, Im S_fw_φ: p = 0.78
