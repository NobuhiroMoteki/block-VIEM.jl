# Mie internal field evaluation for diagnostic comparison with VIEM.
#
# Computes the electric field INSIDE a homogeneous sphere illuminated by an
# x-polarized plane wave propagating in +z, using the exact Mie series.
#
# Convention: exp(-iωt) (Bohren & Huffman 1983).
#   E^inc = E0 x̂ exp(+ikz)
#   G = exp(+ikR)/(4πR)
#
# To compare with VIEM (exp(+jωt) convention):
#   D_VIEM(r) = ε_p * conj(E_int^BH(r))
#
# References:
#   BH83 = Bohren & Huffman, "Absorption and Scattering of Light by Small
#          Particles" (1983), Chapter 4.

using SpecialFunctions: besselj, bessely

# ============================================================================
# Spherical Bessel functions of complex argument
# ============================================================================

"""
    sph_besselj(n, z) -> ComplexF64

Spherical Bessel function of the first kind: j_n(z) = √(π/(2z)) J_{n+1/2}(z).
"""
function sph_besselj(n::Int, z::Number)
    zc = ComplexF64(z)
    if abs(zc) < 1e-30
        return n == 0 ? one(ComplexF64) : zero(ComplexF64)
    end
    return sqrt(π / (2zc)) * besselj(n + 0.5, zc)
end

"""
    ricatti_jn(n, z) -> ComplexF64

Ricatti-Bessel function ψ_n(z) = z j_n(z).
"""
function ricatti_jn(n::Int, z::Number)
    return ComplexF64(z) * sph_besselj(n, z)
end

"""
    ricatti_jn_deriv(n, z) -> ComplexF64

Derivative ψ_n'(z) = d[z j_n(z)]/dz, computed via the recurrence:
  ψ_n'(z) = ψ_{n-1}(z) - (n/z) ψ_n(z)   for n ≥ 1
  ψ_0'(z) = cos(z)
"""
function ricatti_jn_deriv(n::Int, z::Number)
    if n == 0
        return ComplexF64(cos(z))
    end
    return ricatti_jn(n - 1, z) - n / ComplexF64(z) * ricatti_jn(n, z)
end

# ============================================================================
# Mie internal field coefficients c_n, d_n  (BH83 Eq. 4.88)
# ============================================================================

"""
    mie_internal_coefficients(; wl_0, m_m, r_p, m_p)
        -> (c, d, nstop, x, m_r)

Compute internal Mie coefficients c_n, d_n (BH83 convention, exp(-iωt)).

Uses the OUTGOING-WAVE Ricatti-Hankel function (MATLAB MIE.m convention):
  ξ_n^out = ψ_n - i χ_n = x h_n^(1)(x)   [NIST convention for h^(1)]

where χ_n = -x y_n (BH83 definition). The Wronskian is:
  W = ψ_n ξ_n^out' - ψ_n' ξ_n^out = +i

NOTE: BH83 Eq. 4.13 defines ξ_n = ψ_n + iχ_n = x h_n^(2), which is the
INCOMING wave. Using that for scattered-field coefficients gives correct
cross sections (Q_ext, Q_sca only involve Re/|·|² which are invariant)
but WRONG internal field coefficients c_n, d_n. The MATLAB MIE.m code
by Moteki uses the outgoing wave, which gives correct c_n, d_n.

From BH83 Eq. 4.88 with the outgoing wave (numerator = m_r × W = +i m_r):

    c_n = +i m_r / [ψ_n(mx) ξ_n'(x) - m_r ψ_n'(mx) ξ_n(x)]
    d_n = +i m_r / [m_r ψ_n(mx) ξ_n'(x) - ψ_n'(mx) ξ_n(x)]

Returns 1-indexed arrays: c[n+1] = c_n, d[n+1] = d_n for n = 1..nstop.
Index 1 (n=0) is unused (set to zero).
"""
function mie_internal_coefficients(; wl_0::Real, m_m::Real, r_p::Real, m_p::Number)
    k_bg = 2π * m_m / wl_0
    x = k_bg * r_p
    m_r = ComplexF64(m_p) / m_m
    y = m_r * x  # = k_p * r_p

    nstop = floor(Int, abs(x) + 4 * abs(x)^(1 / 3) + 2)
    nstop = max(nstop, 3)

    c = zeros(ComplexF64, nstop + 1)
    d = zeros(ComplexF64, nstop + 1)

    @inbounds for n in 1:nstop
        # Ricatti-Bessel at x (real argument)
        psi_x = ricatti_jn(n, x)
        psi_x_d = ricatti_jn_deriv(n, x)
        chi_x = -real(x) * Float64(bessely(n + 0.5, Float64(x))) * sqrt(π / (2x))
        chi_x_d = if n == 0
            sin(x)
        else
            # chi_n = -x y_n(x), chi_n' via recurrence:
            # chi_n'(x) = chi_{n-1}(x) - (n/x) chi_n(x)
            chi_prev = -Float64(x) * Float64(bessely(n - 1 + 0.5, Float64(x))) * sqrt(π / (2x))
            chi_prev - n / x * chi_x
        end

        # ξ_n^out(x) = ψ_n(x) - i χ_n(x) = x h_n^(1)(x)  (outgoing wave)
        # where χ_n = -x y_n (BH83 definition).
        # This matches MATLAB MIE.m: qsi = complex(psi, chi) with chi = x*y = -χ_BH.
        # Wronskian: W = ψ ξ^out' - ψ' ξ^out = +i.
        # NOTE: mie_reference.jl uses ξ = ψ + iχ = x h^(2) (incoming wave),
        # which gives correct cross sections but wrong c_n, d_n.
        xi_x = ComplexF64(psi_x) - im * chi_x
        xi_x_d = ComplexF64(psi_x_d) - im * chi_x_d

        # Ricatti-Bessel at y = m_r x (complex argument)
        psi_y = ricatti_jn(n, y)
        psi_y_d = ricatti_jn_deriv(n, y)

        # Internal coefficients (BH83 Eq. 4.88, outgoing wave, W = +i)
        # Numerator = m_r × W = +i m_r
        # d_n (TM internal): relates to a_n denominator
        den_d = m_r * psi_y * xi_x_d - psi_y_d * xi_x
        d[n + 1] = im * m_r / den_d

        # c_n (TE internal): relates to b_n denominator
        den_c = psi_y * xi_x_d - m_r * psi_y_d * xi_x
        c[n + 1] = im * m_r / den_c
    end

    return (; c, d, nstop, x, m_r)
end

# ============================================================================
# Angular functions π_n, τ_n  (BH83 Eqs. 4.46, 4.47)
# ============================================================================

"""
    compute_pi_tau(nstop, cos_theta) -> (pi_n, tau_n)

Compute π_n(cos θ) and τ_n(cos θ) for n = 0..nstop via upward recurrence.

  π_n = P_n^1 / sin θ
  τ_n = dP_n^1/dθ

Recurrences (BH83 p.94):
  π_n = ((2n-1)/(n-1)) cos θ π_{n-1} - (n/(n-1)) π_{n-2}
  τ_n = n cos θ π_n - (n+1) π_{n-1}

with π_0 = 0, π_1 = 1.

Returns 1-indexed arrays: pi_n[n+1] = π_n, tau_n[n+1] = τ_n.
"""
function compute_pi_tau(nstop::Int, cos_theta::Float64)
    mu = cos_theta
    pi_n = zeros(Float64, nstop + 1)
    tau_n = zeros(Float64, nstop + 1)

    # π_0 = 0  (index 1)
    # π_1 = 1  (index 2)
    if nstop >= 1
        pi_n[2] = 1.0
        tau_n[2] = mu  # τ_1 = cos θ
    end

    @inbounds for n in 2:nstop
        pi_n[n + 1] = ((2n - 1) / (n - 1)) * mu * pi_n[n] - (n / (n - 1)) * pi_n[n - 1]
        tau_n[n + 1] = n * mu * pi_n[n + 1] - (n + 1) * pi_n[n]
    end

    return pi_n, tau_n
end

# ============================================================================
# Internal field evaluation at a point (BH83 Eq. 4.40)
# ============================================================================

"""
    mie_internal_field(r_vec; wl_0, m_m, r_p, m_p,
                       coeffs=nothing) -> SVector{3,ComplexF64}

Evaluate the exact Mie internal electric field at position `r_vec` (Cartesian)
inside a sphere of radius `r_p` centered at the origin.

Incident field: x-polarized, z-propagating plane wave with unit amplitude
  E^inc = x̂ exp(+ik_bg z)  (BH convention, exp(-iωt))

The internal field (BH83 Eq. 4.40):
  E_int = Σ_n E_n [c_n M_{o1n}^(1)(k_p r) - i d_n N_{e1n}^(1)(k_p r)]

where E_n = i^n (2n+1)/(n(n+1)), k_p = m_r k_bg.

VSH components in spherical coordinates (BH83 Eq. 4.50):
  M_{o1n,r} = 0
  M_{o1n,θ} = cos φ π_n(cos θ) j_n(ρ)
  M_{o1n,φ} = -sin φ τ_n(cos θ) j_n(ρ)

  N_{e1n,r} = cos φ n(n+1) sin θ π_n(cos θ) j_n(ρ)/ρ
  N_{e1n,θ} = cos φ τ_n(cos θ) ψ_n'(ρ)/ρ
  N_{e1n,φ} = -sin φ π_n(cos θ) ψ_n'(ρ)/ρ

where ρ = k_p r = m_r k_bg r.
"""
function mie_internal_field(r_vec::SVector{3,Float64};
                            wl_0::Real, m_m::Real, r_p::Real, m_p::Number,
                            coeffs=nothing)
    # Compute coefficients if not provided
    if coeffs === nothing
        coeffs = mie_internal_coefficients(; wl_0=wl_0, m_m=m_m, r_p=r_p, m_p=m_p)
    end
    (; c, d, nstop, x, m_r) = coeffs

    k_bg = 2π * m_m / wl_0
    k_p = m_r * k_bg

    # Cartesian → Spherical
    x_c, y_c, z_c = r_vec[1], r_vec[2], r_vec[3]
    r = sqrt(x_c^2 + y_c^2 + z_c^2)
    if r < 1e-30
        # At origin: only n=1 dipole term contributes, E ≈ c_1 E_1 ... simplified
        # For small r, E_int → some finite value. Use a tiny offset.
        r = 1e-20
    end
    cos_theta = z_c / r
    sin_theta = sqrt(x_c^2 + y_c^2) / r

    # Handle polar axis (sin_theta ≈ 0)
    if sin_theta < 1e-15
        phi = 0.0
    else
        phi = atan(y_c, x_c)
    end
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    # Angular functions
    pi_n, tau_n = compute_pi_tau(nstop, cos_theta)

    # Complex argument for internal Bessel functions
    rho = k_p * r  # complex in general

    # Accumulate field in spherical components
    E_r = zero(ComplexF64)
    E_theta = zero(ComplexF64)
    E_phi = zero(ComplexF64)

    @inbounds for n in 1:nstop
        # Expansion coefficient E_n = i^n (2n+1) / (n(n+1))
        En = (im)^n * (2n + 1) / (n * (n + 1))

        jn_rho = sph_besselj(n, rho)
        psi_rho = rho * jn_rho
        psi_rho_d = ricatti_jn_deriv(n, rho)

        cn = c[n + 1]
        dn = d[n + 1]
        pn = pi_n[n + 1]
        tn = tau_n[n + 1]

        # M_{o1n}^(1) contributions (no r-component)
        # M_θ = cos φ π_n j_n,  M_φ = -sin φ τ_n j_n
        M_theta = cos_phi * pn * jn_rho
        M_phi = -sin_phi * tn * jn_rho

        # N_{e1n}^(1) contributions
        # N_r = cos φ n(n+1) sin θ π_n j_n/ρ
        N_r = if abs(rho) > 1e-30
            cos_phi * n * (n + 1) * sin_theta * pn * jn_rho / rho
        else
            zero(ComplexF64)
        end
        # N_θ = cos φ τ_n ψ_n'/ρ
        N_theta = if abs(rho) > 1e-30
            cos_phi * tn * psi_rho_d / rho
        else
            zero(ComplexF64)
        end
        # N_φ = -sin φ π_n ψ_n'/ρ
        N_phi = if abs(rho) > 1e-30
            -sin_phi * pn * psi_rho_d / rho
        else
            zero(ComplexF64)
        end

        # E_int += E_n [c_n M_{o1n} - i d_n N_{e1n}]
        E_r += En * (-im * dn * N_r)
        E_theta += En * (cn * M_theta - im * dn * N_theta)
        E_phi += En * (cn * M_phi - im * dn * N_phi)
    end

    # Spherical → Cartesian
    # E_x = sin θ cos φ E_r + cos θ cos φ E_θ - sin φ E_φ
    # E_y = sin θ sin φ E_r + cos θ sin φ E_θ + cos φ E_φ
    # E_z = cos θ E_r - sin θ E_θ
    Ex = sin_theta * cos_phi * E_r + cos_theta * cos_phi * E_theta - sin_phi * E_phi
    Ey = sin_theta * sin_phi * E_r + cos_theta * sin_phi * E_theta + cos_phi * E_phi
    Ez = cos_theta * E_r - sin_theta * E_theta

    return SVector{3,ComplexF64}(Ex, Ey, Ez)
end

# ============================================================================
# Convenience: evaluate D_Mie in VIEM convention at multiple points
# ============================================================================

"""
    mie_D_field_viem_convention(points; wl_0, m_m, r_p, m_p)

Compute D_Mie = ε_p * conj(E_int^BH) at each point in `points`, converting
from BH (exp(-iωt)) to VIEM engineering (exp(+jωt)) convention.

Returns a vector of SVector{3,ComplexF64}.
"""
function mie_D_field_viem_convention(points::AbstractVector{SVector{3,Float64}};
                                     wl_0::Real, m_m::Real, r_p::Real, m_p::Number)
    eps_p = ComplexF64(m_p)^2
    coeffs = mie_internal_coefficients(; wl_0=wl_0, m_m=m_m, r_p=r_p, m_p=m_p)

    D_mie = Vector{SVector{3,ComplexF64}}(undef, length(points))
    for (i, r) in enumerate(points)
        E_bh = mie_internal_field(r; wl_0=wl_0, m_m=m_m, r_p=r_p, m_p=m_p,
                                  coeffs=coeffs)
        # Convention conversion: E_eng = conj(E_BH)
        D_mie[i] = eps_p * conj(E_bh)
    end
    return D_mie
end
