# Mie theory reference implementation for validation tests.
# Based on Bohren & Huffman (1983) and the MieScat_Py code by N. Moteki.
#
# NOT part of the BlockVIEM package; used only in tests.
#
# Convention: exp(−iωt) physics convention (BH83).
#   ξ_n = ψ_n + iχ_n  (BH83 Eq. 4.13, p.100) = x h_n^(2)(x) (INCOMING wave)
#   where ψ_n = ρ j_n(ρ), χ_n = -ρ y_n(ρ).
#   Wronskian: W = ψ_n ξ_n' - ψ_n' ξ_n = -i.
#
# Cross sections Q_ext, Q_sca, Q_abs are correct because they depend only on
# Re(a_n+b_n) and |a_n|²+|b_n|², which are invariant under h^(1)↔h^(2) for
# real x (lossless background). Validated against MieScat_Py to 1e-15.
#
# CAUTION: The partial-wave coefficients a_n, b_n, c_n, d_n from this file
# use h^(2) (incoming wave) and are NOT suitable for internal field evaluation.
# For internal field coefficients, use mie_internal_field.jl which uses
# ξ^out = ψ - iχ = x h^(1) (outgoing wave, matching MATLAB MIE.m).

"""
    mie_cross_sections(; wl_0, m_m, r_p, m_p) -> NamedTuple

Compute Mie theory cross sections for a homogeneous sphere.

# Arguments
- `wl_0` — vacuum wavelength (same length unit as `r_p`)
- `m_m`  — real refractive index of the surrounding medium
- `r_p`  — sphere radius
- `m_p`  — complex refractive index of the particle

# Returns
Named tuple with fields `C_ext`, `C_sca`, `C_abs`, `Q_ext`, `Q_sca`, `Q_abs`,
`x` (size parameter), `nstop`, `a` and `b` (partial-wave coefficient vectors,
1-indexed with index 1 = n=0 unused).
"""
function mie_cross_sections(; wl_0::Real, m_m::Real, r_p::Real, m_p::Number)
    k_bg = 2π * m_m / wl_0          # wavenumber in the medium
    x = k_bg * r_p                    # size parameter
    m_r = ComplexF64(m_p) / m_m       # relative refractive index

    nstop = floor(Int, abs(x) + 4 * abs(x)^(1 / 3) + 2)
    nstop = max(nstop, 3)             # at least 3 terms

    DD, psi, xi = _mie_bessel(x, m_r, nstop)

    # Partial-wave coefficients a_n, b_n  (BH83 Eq. 4.88)
    # Index convention: a[n+1] = a_n for n = 1, ..., nstop.
    a = zeros(ComplexF64, nstop + 1)
    b = zeros(ComplexF64, nstop + 1)
    @inbounds for n in 1:nstop
        n_over_x = n / x
        # a_n
        num_a = (DD[n + 1] / m_r + n_over_x) * psi[n + 1] - psi[n]
        den_a = (DD[n + 1] / m_r + n_over_x) * xi[n + 1]  - xi[n]
        a[n + 1] = num_a / den_a
        # b_n
        num_b = (m_r * DD[n + 1] + n_over_x) * psi[n + 1] - psi[n]
        den_b = (m_r * DD[n + 1] + n_over_x) * xi[n + 1]  - xi[n]
        b[n + 1] = num_b / den_b
    end

    # Efficiency factors
    Q_sca = 0.0
    Q_ext = 0.0
    @inbounds for n in 1:nstop
        fn1 = 2n + 1
        Q_sca += fn1 * (abs2(a[n + 1]) + abs2(b[n + 1]))
        Q_ext += fn1 * real(a[n + 1] + b[n + 1])
    end
    Q_sca *= 2 / x^2
    Q_ext *= 2 / x^2
    Q_abs = Q_ext - Q_sca

    G = π * r_p^2
    C_ext = Q_ext * G
    C_sca = Q_sca * G
    C_abs = Q_abs * G

    return (; C_ext, C_sca, C_abs, Q_ext, Q_sca, Q_abs, x, nstop, a, b)
end

# ---------------------------------------------------------------------------
# Internal: Ricatti-Bessel functions and logarithmic derivative
# ---------------------------------------------------------------------------
function _mie_bessel(x::Real, m_r::ComplexF64, nstop::Int)
    y = m_r * x
    nmx = floor(Int, max(nstop, abs(y)) + 15)

    # Logarithmic derivative D_n(y) by downward recurrence
    DD = zeros(ComplexF64, nmx + 1)
    @inbounds for n in nmx:-1:1
        DD[n] = n / y - 1 / (DD[n + 1] + n / y)
    end
    DD = DD[1:(nstop + 1)]

    # psi_n(x) = x j_n(x) via ratio downward recurrence
    R = zeros(Float64, nmx + 1)
    R[nmx + 1] = x / (2 * nmx + 1)
    @inbounds for n in (nmx - 1):-1:0
        R[n + 1] = 1.0 / ((2n + 1) / x - R[n + 2])
    end
    psi = zeros(Float64, nstop + 1)
    psi[1] = R[1] * cos(x)
    @inbounds for n in 1:nstop
        psi[n + 1] = R[n + 1] * psi[n]
    end

    # chi_n(x) = −x y_n(x) by forward recurrence
    chi = zeros(Float64, nstop + 1)
    chi[1] = -cos(x)
    chi[2] = chi[1] / x - sin(x)
    @inbounds for n in 2:nstop
        chi[n + 1] = ((2n - 1) / x) * chi[n] - chi[n - 1]
    end

    xi = psi .+ im .* chi
    return DD, psi, xi
end

"""
    mie_cas_observables(; wl_0, m_m, r_p, m_p) -> NamedTuple

Reference CAS-v2 forward and backward scattering observables for a
homogeneous sphere, **physics convention** (matching block-DDA_Py
`analytical_scattering_theories/homogeneous_sphere.mie_compute_q_and_s`).

Returns `(; S_fw, S_bk, S1_fw, S2_fw, S1_bk, S2_bk, k)` where
`S_fw = (S11(0)+S22(0))/2` and `S_bk = (-S11(π)+S22(π))/√2` with
`S11 = S2/(-ik)`, `S22 = S1/(-ik)`. For a sphere `S1(0)=S2(0)`, so
`S_fw = S1(0)/(-ik)` independent of polarization.
"""
function mie_cas_observables(; wl_0::Real, m_m::Real, r_p::Real, m_p::Number)
    k_bg = 2π * m_m / wl_0
    x = k_bg * r_p
    m_r = ComplexF64(m_p) / m_m

    nstop = floor(Int, abs(x) + 4 * abs(x)^(1/3) + 2)
    nstop = max(nstop, 3)
    DD, psi, xi = _mie_bessel(x, m_r, nstop)

    a = zeros(ComplexF64, nstop)
    b = zeros(ComplexF64, nstop)
    @inbounds for n in 1:nstop
        n_over_x = n / x
        num_a = (DD[n+1] / m_r + n_over_x) * psi[n+1] - psi[n]
        den_a = (DD[n+1] / m_r + n_over_x) * xi[n+1]  - xi[n]
        a[n] = num_a / den_a
        num_b = (m_r * DD[n+1] + n_over_x) * psi[n+1] - psi[n]
        den_b = (m_r * DD[n+1] + n_over_x) * xi[n+1]  - xi[n]
        b[n] = num_b / den_b
    end

    # Angular functions at θ = 0 and θ = π
    # π_n(μ) and τ_n(μ): π_1=1, π_2=3μ, recurrence; τ_n = n μ π_n - (n+1) π_{n-1}
    # At θ=0 (μ=1): π_n(1) = n(n+1)/2, τ_n(1) = n(n+1)/2
    # At θ=π (μ=-1): π_n(-1) = (-1)^(n+1) n(n+1)/2, τ_n(-1) = (-1)^n n(n+1)/2 · (-1)
    #                Actually use recurrences directly to avoid sign mistakes.
    pie_fw = zeros(Float64, nstop); tau_fw = zeros(Float64, nstop)
    pie_bk = zeros(Float64, nstop); tau_bk = zeros(Float64, nstop)
    mu_fw = 1.0;  mu_bk = -1.0
    pie_fw[1] = 1.0; tau_fw[1] = mu_fw
    pie_bk[1] = 1.0; tau_bk[1] = mu_bk
    if nstop >= 2
        pie_fw[2] = 3.0 * mu_fw; tau_fw[2] = 6 * mu_fw^2 - 3
        pie_bk[2] = 3.0 * mu_bk; tau_bk[2] = 6 * mu_bk^2 - 3
    end
    @inbounds for n in 3:nstop
        pie_fw[n] = ((2n - 1)/(n - 1)) * mu_fw * pie_fw[n-1] - (n/(n - 1)) * pie_fw[n-2]
        tau_fw[n] = n * mu_fw * pie_fw[n] - (n + 1) * pie_fw[n-1]
        pie_bk[n] = ((2n - 1)/(n - 1)) * mu_bk * pie_bk[n-1] - (n/(n - 1)) * pie_bk[n-2]
        tau_bk[n] = n * mu_bk * pie_bk[n] - (n + 1) * pie_bk[n-1]
    end

    S1_fw = ComplexF64(0); S2_fw = ComplexF64(0)
    S1_bk = ComplexF64(0); S2_bk = ComplexF64(0)
    @inbounds for n in 1:nstop
        fn2 = (2n + 1) / (n * (n + 1))
        S1_fw += fn2 * (a[n] * pie_fw[n] + b[n] * tau_fw[n])
        S2_fw += fn2 * (a[n] * tau_fw[n] + b[n] * pie_fw[n])
        S1_bk += fn2 * (a[n] * pie_bk[n] + b[n] * tau_bk[n])
        S2_bk += fn2 * (a[n] * tau_bk[n] + b[n] * pie_bk[n])
    end

    S11_fw = S2_fw / (-im * k_bg); S22_fw = S1_fw / (-im * k_bg)
    S11_bk = S2_bk / (-im * k_bg); S22_bk = S1_bk / (-im * k_bg)

    S_fw = (S11_fw + S22_fw) / 2
    S_bk = (-S11_bk + S22_bk) / sqrt(2.0)

    return (; S_fw, S_bk,
              S1_fw, S2_fw, S1_bk, S2_bk,
              S11_fw, S22_fw, S11_bk, S22_bk,
              k = k_bg, x, nstop)
end
