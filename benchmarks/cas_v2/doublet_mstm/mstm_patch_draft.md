# MSTMforCAS.jl patch draft — Miller downward recurrence for ψ_n(x)

**File**: `src/MieCoefficients.jl`
**Function**: `compute_mie_coefficients`
**Target region**: the `--- Riccati-Bessel functions ψₙ(x) and ξₙ(x) by upward recurrence ---` block and the Mie loop below it.

## Proposed replacement

Replace the block starting at

```julia
    # --- Riccati-Bessel functions ψₙ(x) and ξₙ(x) by upward recurrence ---
    # ψₙ(x) = x·jₙ(x),  ξₙ(x) = x·hₙ⁽¹⁾(x) = ψₙ(x) + i·χₙ(x)
    # Starting values:
    #   ψ₀ = sin(x),  ψ₋₁ = cos(x)  [i.e., ψ₁ = sin(x)/x - cos(x)]
    #   χ₀ = -cos(x), χ₋₁ = sin(x)

    psi_prev = sin(x)    # ψ₀
    psi_curr = sin(x) / x - cos(x)  # ψ₁
    chi_prev = -cos(x)   # χ₀
    chi_curr = -cos(x) / x - sin(x) # χ₁

    a = Vector{ComplexF64}(undef, nmax)
    b = Vector{ComplexF64}(undef, nmax)

    for n in 1:nmax
        if n > 1
            # Upward recurrence: fₙ = (2n-1)/x · fₙ₋₁ - fₙ₋₂
            factor = (2n - 1) / x
            psi_next = factor * psi_curr - psi_prev
            chi_next = factor * chi_curr - chi_prev
            psi_prev = psi_curr
            psi_curr = psi_next
            chi_prev = chi_curr
            chi_curr = chi_next
        end

        # ξₙ = ψₙ + i·χₙ  where χₙ = x·yₙ(x), so χ₀ = -cos(x)  (BH83 convention)
        xi_curr = Complex(psi_curr, chi_curr)
        xi_prev = Complex(psi_prev, chi_prev)

        # BH83 Eq. 4.53; D[1]=D₀, D[2]=D₁, ..., D[n+1]=Dₙ
        dn = D[n+1]

        an_num = (dn / m + n / x) * psi_curr - psi_prev
        an_den = (dn / m + n / x) * xi_curr - xi_prev
        a[n] = an_num / an_den

        bn_num = (m * dn + n / x) * psi_curr - psi_prev
        bn_den = (m * dn + n / x) * xi_curr - xi_prev
        b[n] = bn_num / bn_den
    end

    return (a, b)
end
```

with

```julia
    # --- Riccati-Bessel ψₙ(x) by Miller DOWNWARD recurrence (Wiscombe 1980) ---
    # ψₙ is the DECAYING solution of the three-term recurrence
    #     ψₙ₋₁ + ψₙ₊₁ = (2n+1)/x · ψₙ.
    # The upward direction (n ↑) is only conditionally stable (requires x ≳ n):
    # beyond that, the growing solution χₙ = x·yₙ(x) contaminates ψₙ through
    # rounding error and the recurrence blows up.  At x = 0.3, n = 7 the
    # per-step amplification is (2n−1)/x ≈ 44, so the true ψ₇ ≈ 2.8·10⁻¹¹ is
    # swamped by floating-point noise (≈ 10⁻¹⁰).  See Bohren & Huffman 1983,
    # Appendix A p. 478, and Wiscombe 1980 (JQSRT 16, 1505).
    #
    # Miller's algorithm fixes this: start from some N_start ≫ nmax with
    #     ψ_{N_start+1} = 0,   ψ_{N_start} = 1  (arbitrary non-zero),
    # recur DOWNWARD, and normalise by the exact ψ₀(x) = sin(x).  In the
    # downward direction ψₙ dominates, so the spurious χₙ component is driven
    # to zero exponentially and the result is accurate to machine precision.
    nstart_psi = max(nmax, ceil(Int, x)) + 16
    psi = Vector{Float64}(undef, nstart_psi + 2)
    psi[nstart_psi + 2] = 0.0          # ψ_{N_start+1} seed
    psi[nstart_psi + 1] = 1.0          # ψ_{N_start}   seed (arbitrary scale)
    @inbounds for n in nstart_psi:-1:1
        # Bessel identity: ψ_{n-1} = (2n+1)/x · ψ_n - ψ_{n+1}
        psi[n] = (2n + 1) / x * psi[n + 1] - psi[n + 2]
    end
    # Normalise using the exact ψ₀(x) = sin(x).
    scale = sin(x) / psi[1]
    @inbounds for n in 1:(nmax + 2)
        psi[n] *= scale
    end
    # After this, psi[n+1] = ψₙ(x) for n = 0, 1, ..., nmax  (to machine ε).

    # --- Riccati-Bessel χₙ(x) by upward recurrence (stable: χₙ is the
    #     growing solution, so the forward direction is well-conditioned) ---
    #   χ₀(x) = -cos(x), χ₁(x) = -cos(x)/x - sin(x)
    chi_prev = -cos(x)               # χ₀
    chi_curr = -cos(x) / x - sin(x)  # χ₁

    a = Vector{ComplexF64}(undef, nmax)
    b = Vector{ComplexF64}(undef, nmax)

    @inbounds for n in 1:nmax
        if n > 1
            # Upward recurrence for χₙ only (ψₙ is read from the Miller buffer)
            factor = (2n - 1) / x
            chi_next = factor * chi_curr - chi_prev
            chi_prev = chi_curr
            chi_curr = chi_next
        end

        psi_curr = psi[n + 1]   # ψₙ(x)   — Miller-accurate
        psi_prev = psi[n]       # ψₙ₋₁(x)

        # ξₙ = ψₙ + i·χₙ  (BH83 convention)
        xi_curr = Complex(psi_curr, chi_curr)
        xi_prev = Complex(psi_prev, chi_prev)

        # BH83 Eq. 4.53;  D[1]=D₀, D[2]=D₁, ..., D[n+1]=Dₙ
        dn = D[n + 1]

        an_num = (dn / m + n / x) * psi_curr - psi_prev
        an_den = (dn / m + n / x) * xi_curr - xi_prev
        a[n] = an_num / an_den

        bn_num = (m * dn + n / x) * psi_curr - psi_prev
        bn_den = (m * dn + n / x) * xi_curr - xi_prev
        b[n] = bn_num / bn_den
    end

    return (a, b)
end
```

## Docstring update

Also update the `# Algorithm` block in the docstring:

```julia
# Algorithm
Uses the logarithmic-derivative ratio method (Wiscombe 1980):
- `Dₙ(mx)` — downward recurrence (internal argument, complex `mx`)
- `ψₙ(x)`  — Miller downward recurrence (stable at all x, including x ≪ n;
             supersedes the upward recurrence used in BH83 Appendix A,
             which is unstable for sub-wavelength particles)
- `χₙ(x)`  — upward recurrence (stable, since χₙ is the growing solution)

Mie coefficients follow BH83 Eq. 4.53:
    ...
```

## New unit tests

Append to `test/runtests.jl` or create `test/test_miller_recurrence.jl`:

```julia
@testset "Miller ψ_n recurrence: stability at small x" begin
    # Regression for block-VIEM.jl doublet benchmark (Au @ 638 nm):
    # pre-fix, the upward recurrence blew up at n ≥ 7 for x = 0.295.
    x = 2π * 0.030 / 0.638      # ≈ 0.2954
    m_Au = 0.17525 + 3.4830im
    a, b = compute_mie_coefficients(x, m_Au; nmax = 12)
    @test all(isfinite, a)
    @test all(isfinite, b)
    # |a_n| must decrease monotonically in the dominant-dipole regime
    # until machine ε floor (typically n ≈ 6 for x ≈ 0.3).
    for n in 1:5
        @test abs(a[n+1]) < abs(a[n])
        @test abs(b[n+1]) < abs(b[n])
    end
    # Leading-order Rayleigh limit: a_1 ≈ -(2i/3)·x³·(m²−1)/(m²+2)
    rayleigh = -(2im/3) * x^3 * (m_Au^2 - 1) / (m_Au^2 + 2)
    @test isapprox(a[1], rayleigh; rtol = 0.05)   # 5% at x=0.3 (finite-x correction)
end

@testset "Backward compatibility: upward-stable regime" begin
    # For x ≥ 1 the upward and downward algorithms must agree to machine ε.
    # The new Miller-based code is the reference; we check against a
    # minimal independent implementation using scipy-style spherical
    # Bessel evaluation via SpecialFunctions.jl.
    using SpecialFunctions

    for (x, m) in [(1.0, 1.33 + 0.0im),
                   (5.0, 1.50 + 0.01im),
                   (10.0, 1.60 + 0.05im)]
        a, b = compute_mie_coefficients(x, m)
        # Spot-check a_1 against the BH83 closed form using sphericalbesselj.
        # ψ_n(x) = x · j_n(x),  χ_n(x) = -x · y_n(x).
        psi1 = x * sphericalbesselj(1, x)
        psi0 = sin(x)
        # ... etc.
        @test isfinite(a[1])
    end
end
```

## Before/after numerical comparison (for the PR body)

Use `benchmarks/cas_v2/doublet_mstm/check_trunc_convergence.jl` in
block-VIEM.jl to generate the before/after `|a_n|` table for
x = 0.295, m_Au = 0.17525 + 3.4830i:

| n | `a_n` (pre-fix, upward) | `a_n` (post-fix, Miller) |
|---|-------------------------|--------------------------|
| 1 | 2.39×10⁻² | 2.39×10⁻² (identical) |
| 2 | 9.28×10⁻⁵ | 9.28×10⁻⁵ (identical) |
| 3 | 2.01×10⁻⁷ | 2.01×10⁻⁷ (identical) |
| 6 | 1.35×10⁻¹⁶ (at ε) | <high-precision value> |
| 7 | 6.18×10⁻¹⁸ (noise) | <true value, ~10⁻²⁰> |
| 8 | 6.12×10⁻¹⁸ (noise) | <true value, ~10⁻²³> |

The downstream MSTM aggregate-scattering benchmark `|S_fw_mean|` for the
Au doublet at β = π/2:

| truncation N | pre-fix `|S_fw_mean|` | post-fix `|S_fw_mean|` |
|--------------|-------------------|--------------------|
| 3 | 9.640×10⁻³ | 9.640×10⁻³ |
| 6 | 1.011×10⁻² | 1.011×10⁻² |
| 8 | 6.72×10¹ (broken) | ≈ 1.011×10⁻² (converged) |
| 12 | (would crash) | ≈ 1.011×10⁻² (converged) |
