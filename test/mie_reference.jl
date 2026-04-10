# Mie theory reference implementation for validation tests.
# Based on Bohren & Huffman (1983) and the MieScat_Py code by N. Moteki.
#
# NOT part of the BlockVIEM package; used only in tests.
#
# Convention: exp(−iωt) physics convention for Mie coefficients a_n, b_n.
# Cross sections are convention-independent (real, positive quantities).

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
