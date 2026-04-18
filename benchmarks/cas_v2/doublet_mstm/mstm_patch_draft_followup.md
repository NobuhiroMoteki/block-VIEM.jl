# MSTMforCAS.jl follow-up patch — size the Mie vectors to match `nois[i]`

The Miller downward-recurrence fix (PR #1, merged as commit `bb77980`)
restored numerical stability of `ψ_n(x)` at small x, but the doublet
benchmark still crashes at `truncation_order ≥ 9` for x ≈ 0.3 because
of a separate, independent bug in `solve_tmatrix`.

## Symptom

```
ERROR: BoundsError: attempt to access 8-element Vector{ComplexF64} at index [9]
  ...
  [3] _precompute_T_values(mie_vecs, nois)  @ TMatrixSolver.jl:141
  [4] solve_tmatrix(...; truncation_order = 15)  @ TMatrixSolver.jl:444
```

## Root cause

`src/TMatrixSolver.jl:441`:

```julia
for i in 1:N
    mie_vecs[i] = compute_mie_coefficients(radii[i], m_rel)
end
```

This call omits the `nmax` keyword, so `compute_mie_coefficients`
falls back to its default `nmax = mie_nmax(radii[i])` — the Wiscombe
upper bound for the isolated sphere, which is 8 at x ≈ 0.3.  Right
above (lines 425–429) `nois[i] = max(truncation_order, noi_auto)` can
be larger than this default (e.g. 15 when the user requests
`truncation_order = 15`).  `_precompute_T_values` then iterates
`for n in 1:nois[i]` and accesses `a_v[n]`, `b_v[n]` past the end of
the pre-allocated 8-element vectors.

## Fix (one-line change)

**File**: `src/TMatrixSolver.jl`
**Line 441**:

```diff
 for i in 1:N
-    mie_vecs[i] = compute_mie_coefficients(radii[i], m_rel)
+    mie_vecs[i] = compute_mie_coefficients(radii[i], m_rel; nmax = nois[i])
 end
```

`nois[i]` is known at this point (built in the preceding loop on
lines 425–429), and `compute_mie_coefficients` already accepts the
`nmax::Union{Int,Nothing}` keyword.  No other changes are required.

## Test

Add to `test/test_miller_recurrence.jl` (or wherever the Miller
tests live):

```julia
@testset "solve_tmatrix with truncation_order > mie_nmax(x)" begin
    # Regression for the doublet benchmark at x ≈ 0.3: user-requested
    # truncation_order=15 must not blow up even though the Wiscombe
    # default nmax(x=0.3) = 8.
    using MSTMforCAS: compute_scattering
    using LinearAlgebra

    k = 2π / 0.638               # μm⁻¹
    R = 0.030                    # μm
    d = 2R + 0.003               # centre-to-centre separation
    positions = [0.0 0.0; 0.0 0.0; -d/2 +d/2]
    radii     = [R, R]
    m_rel     = 0.17525 + 3.4830im   # Au @ 638 nm

    positions_x = positions .* k
    radii_x     = radii     .* k

    for N in (3, 5, 8, 10, 15)
        result, _ = compute_scattering(
            positions_x, radii_x, m_rel;
            truncation_order = N, tol = 1e-10)
        @test result.converged
        @test all(isfinite, result.S_forward)
        @test isfinite(result.Q_ext)
    end
end
```

## Why this wasn't caught by PR #1

PR #1 tested `compute_mie_coefficients` in isolation (the
Riccati-Bessel stability fix), but did not exercise the full
`solve_tmatrix` pipeline with a user-supplied `truncation_order`
exceeding `mie_nmax(x)`.  The follow-up test above closes that gap.

## Suggested commit message

    Fix Mie-vector undersizing when truncation_order > mie_nmax(x)

    compute_mie_coefficients was being called without nmax in
    solve_tmatrix, so mie_vecs[i] used the Wiscombe default (8 at
    x≈0.3) even when the user requested a larger truncation_order
    (up to nois[i]).  Pass nmax = nois[i] explicitly.  Adds a
    regression test covering N ∈ {3, 5, 8, 10, 15} at x ≈ 0.3.
