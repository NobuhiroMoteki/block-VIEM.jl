# Far-field scattering amplitudes, cross sections, and CAS-v2 observables.
#
# Phase 5 of BlockVIEM.jl.
#
# The scattered far field from a homogeneous particle with solved D-field
# expansion coefficients D_n is
#
#   E^sca(r) ≈ [exp(-jk0 r) / (4π r)] F(r̂)
#
# where the vector scattering amplitude is
#
#   F(r̂) = (k0² κ / ε_bg) (I − r̂r̂) · P(r̂)
#
# and P(r̂) = Σ_n D_n ∫ f_n(r') exp(+jk0 r̂·r') dV' is the Fourier-
# transformed polarization. The transverse projector (I − r̂r̂) removes the
# longitudinal component, ensuring the far field is purely transverse.

using LinearAlgebra: dot, norm
import LinearAlgebra

"""
    far_field_amplitude(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                        k_hat_sca::Vec3,
                        k0::Number,
                        eps_p::Number,
                        eps_bg::Number = 1,
                        rule::TetQuadRule = TET_QUAD_5PT)
        -> SVector{3,ComplexF64}

Compute the vector scattering amplitude `F(k̂_sca)` at observation direction
`k_hat_sca` (unit vector). Works for any `AbstractDivBasis`.
"""
function far_field_amplitude(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                             k_hat_sca::Vec3,
                             k0::Number,
                             eps_p::Number,
                             eps_bg::Number = 1,
                             rule::TetQuadRule = TET_QUAD_5PT)
    k0c = ComplexF64(k0)
    kappa = (ComplexF64(eps_p) - ComplexF64(eps_bg)) / ComplexF64(eps_p)
    eps_bg_c = ComplexF64(eps_bg)

    # P(r̂) = Σ_n D_n ∫ f_n(r') exp(+jk0 r̂·r') dV'
    Px, Py, Pz = zero(ComplexF64), zero(ComplexF64), zero(ComplexF64)
    @inbounds for n in eachindex(D_coeffs)
        Dn = ComplexF64(D_coeffs[n])
        iszero(Dn) && continue
        for tet in support_tets(basis, Int(n))
            tet == 0 && continue
            verts = _tet_vertices(basis.mesh, tet)
            V = tet_volume(verts...)
            for i in 1:rule.n
                r = bary_to_point(rule.bary[i], verts)
                fn = evaluate(basis, Int(n), r, tet)
                phase = exp(im * k0c * dot(k_hat_sca, r))
                w = rule.weights[i] * V * Dn * phase
                Px += w * fn[1]
                Py += w * fn[2]
                Pz += w * fn[3]
            end
        end
    end
    P = SVector{3,ComplexF64}(Px, Py, Pz)

    # Transverse projection: F = (k0²κ/ε_bg)(P − (k̂·P)k̂)
    # The positive sign follows from E^sca = (κ/ε_bg)(k0²+∇∇·)∫ D' G dV'
    # → far-field: (κ/ε_bg) k0² (I − r̂r̂) P  (see derivation below).
    P_trans = P - dot(k_hat_sca, P) * SVector{3,ComplexF64}(k_hat_sca)
    coeff = k0c^2 * kappa / eps_bg_c
    return coeff * P_trans
end

"""
    ScatteringResult

Container for orientation-resolved scattering observables.

# Fields
- `S_fw_s::ComplexF64`  — forward scattering amplitude, s-polarization
- `S_fw_p::ComplexF64`  — forward scattering amplitude, p-polarization
- `S_bak::ComplexF64`   — backward scattering amplitude (scalar)
- `C_ext::Float64`      — extinction cross section (optical theorem)
- `C_abs::Float64`      — absorption cross section (volumetric Joule loss)
- `C_sca::Float64`      — scattering cross section (`C_ext − C_abs`)
"""
struct ScatteringResult
    S_fw_s::ComplexF64
    S_fw_p::ComplexF64
    S_bak::ComplexF64
    C_ext::Float64
    C_abs::Float64
    C_sca::Float64
end

## NOTE: The original compute_scattering with only optical theorem has been
## replaced by the extended version below that supports both :farfield and
## :optical_theorem methods via the `csca_method` keyword.

@inline function _real_unit(E0::SVector{3,ComplexF64})
    v = Vec3(real(E0[1]), real(E0[2]), real(E0[3]))
    n = norm(v)
    return n > 0 ? v / n : Vec3(1, 0, 0)
end

# ============================================================================
# Spherical quadrature for C_sca far-field integration
# ============================================================================

"""
    SphericalQuadRule

Product Gauss-Legendre (θ) × trapezoid (φ) quadrature on the unit sphere.

# Fields
- `directions::Vector{Vec3}` — unit observation vectors r̂
- `weights::Vector{Float64}`  — solid angle weights (sum to 4π)
"""
struct SphericalQuadRule
    directions::Vector{Vec3}
    weights::Vector{Float64}
end

"""
    spherical_product_rule(n_theta::Int) -> SphericalQuadRule

Build a product rule with `n_theta` GL points in θ and `2*n_theta` uniform
points in φ. The rule integrates spherical harmonics of degree up to
`2*n_theta - 1` exactly. Total points: `2 * n_theta^2`.
"""
function spherical_product_rule(n_theta::Int)
    n_phi = 2 * n_theta
    # GL nodes on [0,1] and weights for ∫₀¹ g(t) dt
    t_nodes, t_weights = gauss_legendre_unit(n_theta)
    # Map to θ ∈ [0, π]: θ = π t, dθ = π dt
    # Solid angle element: dΩ = sin θ dθ dφ = sin(πt) π dt dφ
    dphi = 2π / n_phi

    n_total = n_theta * n_phi
    dirs = Vector{Vec3}(undef, n_total)
    wts = Vector{Float64}(undef, n_total)

    idx = 0
    @inbounds for i in 1:n_theta
        theta = π * t_nodes[i]
        st, ct = sincos(theta)
        w_theta = t_weights[i] * π * st * dphi   # GL weight × π × sin θ × Δφ
        for j in 1:n_phi
            phi = dphi * (j - 1)
            sp, cp = sincos(phi)
            idx += 1
            dirs[idx] = Vec3(st * cp, st * sp, ct)
            wts[idx] = w_theta
        end
    end
    return SphericalQuadRule(dirs, wts)
end

"""
    compute_csca_farfield(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                          k0::Number, eps_p::Number, eps_bg::Number = 1,
                          E0_sq::Float64,
                          rule::TetQuadRule = TET_QUAD_5PT,
                          n_theta::Int = 10) -> Float64

Compute the scattering cross section by integrating the far-field power:

    C_sca = (1 / (16π² |E₀|²)) ∫ |F(r̂)|² dΩ

where `F(r̂)` is the vector scattering amplitude from [`far_field_amplitude`](@ref).
This integral is always non-negative and avoids the cancellation sensitivity
of the optical theorem.

The integral is evaluated with a product GL×trapezoid rule of `2 n_theta²`
directions. Default `n_theta = 10` (200 directions) suffices for size
parameters up to ~5. Use `n_theta ≥ 20` for larger particles.
"""
function compute_csca_farfield(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                               k0::Number,
                               eps_p::Number,
                               eps_bg::Number = 1,
                               E0_sq::Float64,
                               rule::TetQuadRule = TET_QUAD_5PT,
                               n_theta::Int = 10)
    sq = spherical_product_rule(n_theta)
    n_dirs = length(sq.directions)

    # Compute |F|² at each direction (threaded)
    F_sq_vals = Vector{Float64}(undef, n_dirs)
    Threads.@threads for k in 1:n_dirs
        F = far_field_amplitude(basis, D_coeffs;
                                k_hat_sca = sq.directions[k],
                                k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                rule = rule)
        F_sq_vals[k] = real(F[1] * conj(F[1]) + F[2] * conj(F[2]) + F[3] * conj(F[3]))
    end
    integral = sum(sq.weights[k] * F_sq_vals[k] for k in 1:n_dirs)

    # C_sca = ∫|F|² dΩ / (16π² |E₀|²)
    return integral / (16π^2 * E0_sq)
end

"""
    compute_scattering(basis, D_coeffs; ..., csca_method=:farfield, n_theta=10)

Extended version with `csca_method` keyword:

- `:farfield` (default) — compute `C_sca` via far-field power integration,
  then `C_ext = C_abs + C_sca`. Robust on coarse meshes.
- `:optical_theorem` — compute `C_ext` via the optical theorem,
  then `C_sca = C_ext - C_abs`. May be inaccurate on coarse meshes.
"""
function compute_scattering(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                            k_hat::Vec3,
                            E0::SVector{3,ComplexF64},
                            k0::Number,
                            eps_p::Number,
                            eps_bg::Number = 1,
                            rule::TetQuadRule = TET_QUAD_5PT,
                            csca_method::Symbol = :farfield,
                            n_theta::Int = 10)
    k0c = ComplexF64(k0)
    eps_p_c = ComplexF64(eps_p)
    eps_bg_c = ComplexF64(eps_bg)

    # --- polarization basis ---
    e_s = _real_unit(E0)
    e_p = Vec3(k_hat[2] * e_s[3] - k_hat[3] * e_s[2],
               k_hat[3] * e_s[1] - k_hat[1] * e_s[3],
               k_hat[1] * e_s[2] - k_hat[2] * e_s[1])

    # --- forward scattering amplitude F(k̂) ---
    F_fw = far_field_amplitude(basis, D_coeffs;
                               k_hat_sca = k_hat, k0 = k0,
                               eps_p = eps_p, eps_bg = eps_bg, rule = rule)
    S_fw_s = dot(SVector{3,ComplexF64}(e_s), F_fw)
    S_fw_p = dot(SVector{3,ComplexF64}(e_p), F_fw)

    # --- backward scattering amplitude F(-k̂) ---
    F_bak = far_field_amplitude(basis, D_coeffs;
                                k_hat_sca = -k_hat, k0 = k0,
                                eps_p = eps_p, eps_bg = eps_bg, rule = rule)
    S_bak = dot(SVector{3,ComplexF64}(e_s), F_bak)

    E0_sq = real(dot(E0, E0))

    # --- absorption cross section (always computed this way) ---
    M = assemble_mass_matrix(basis; rule = rule)
    DhMD = real(dot(D_coeffs, M * D_coeffs))
    C_abs = real(k0c) * imag(eps_p_c) / (real(eps_bg_c) * abs2(eps_p_c) * E0_sq) *
            DhMD

    if csca_method == :farfield
        C_sca = compute_csca_farfield(basis, D_coeffs;
                                      k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                      E0_sq = E0_sq, rule = rule,
                                      n_theta = n_theta)
        C_ext = C_abs + C_sca
        # S_fw_s, S_fw_p: kept as direct far-field values (O(h) accuracy).
        # Cross sections C_ext, C_abs, C_sca are more accurate (O(h²) as
        # integrated quantities). For S(0) accuracy, refine the mesh.
    elseif csca_method == :optical_theorem
        C_ext = -imag(dot(conj.(E0), F_fw)) / (real(k0c) * E0_sq)
        C_sca = C_ext - C_abs
    else
        throw(ArgumentError("Unknown csca_method: $csca_method. Use :farfield or :optical_theorem"))
    end

    return ScatteringResult(S_fw_s, S_fw_p, S_bak, C_ext, C_abs, C_sca)
end

# ============================================================================
# CAS-v2 observables (block-DDA_Py / Moteki & Adachi 2024 compatible)
# ============================================================================
#
# CAS-v2 (Complex Amplitude Sensing v2) observables are defined for circular
# incident polarization in a fixed laboratory frame. The particle orientation
# is parameterized by ZYZ Euler angles (alpha, beta, gamma) — same convention
# as block-DDA_Py: scipy `Rotation.from_euler('ZYZ', [alpha, beta, gamma])`
# rotates the particle coordinate system from the laboratory frame.
#
# Lab geometry (matches block-DDA_Py bl_dda/scatterer.py):
#   u_inc_L      = +z       theta_inc_L     = +x       phi_inc_L     = +y
#   u_sca_fw_L   = +z       theta_sca_fw_L  = +x       phi_sca_fw_L  = +y
#   u_sca_bk_L   = -z       theta_sca_bk_L  = -x       phi_sca_bk_L  = +y
#
# Incident circular polarization:
#   E0 = (theta_inc + j phi_inc) / sqrt(2)
#
# Sign convention note: VIEM internally uses the engineering convention
# (e^{+jωt}, outgoing wave exp(-jk0R)/(4πR)), so `far_field_amplitude` returns
# F^eng with integrand exp(+jk r̂·r'). block-DDA_Py and the Mie reference use
# the physics convention (e^{-iωt}). For the same physical system one has
# F^phys = conj(F^eng), and the CAS observables below are returned in the
# **physics convention** so they can be compared directly to block-DDA_Py and
# to the Mie reference (S_fw = (S11(0)+S22(0))/2 from
# `analytical_scattering_theories/homogeneous_sphere.py`).

"""
    CASOrientation

Per-orientation laboratory→particle-frame geometry vectors used by the
CAS-v2 observables. All vectors are unit vectors expressed in the particle
frame. `e0_inc` is the complex circular polarization vector
`(theta_inc + j phi_inc)/√2`.
"""
struct CASOrientation
    u_inc::Vec3
    theta_inc::Vec3
    phi_inc::Vec3
    e0_inc::SVector{3,ComplexF64}
    u_sca_fw::Vec3
    theta_sca_fw::Vec3
    phi_sca_fw::Vec3
    u_sca_bk::Vec3
    theta_sca_bk::Vec3
    phi_sca_bk::Vec3
end

# Inverse of intrinsic ZYZ rotation R_lp = Rz(α) Ry(β) Rz(γ).
# Returns the matrix that maps lab-frame vectors to particle-frame vectors.
@inline function _zyz_inverse_apply(α::Real, β::Real, γ::Real, v::Vec3)
    sα, cα = sincos(α)
    sβ, cβ = sincos(β)
    sγ, cγ = sincos(γ)
    # R_lp (intrinsic ZYZ) =
    #   [ cα cβ cγ - sα sγ,  -cα cβ sγ - sα cγ,  cα sβ ]
    #   [ sα cβ cγ + cα sγ,  -sα cβ sγ + cα cγ,  sα sβ ]
    #   [          -sβ cγ,             sβ sγ,       cβ ]
    # R_pl = R_lp^T (rows of R_lp become columns of R_pl).
    # particle = R_pl · lab
    x = (cα*cβ*cγ - sα*sγ)*v[1] + (sα*cβ*cγ + cα*sγ)*v[2] + (-sβ*cγ)*v[3]
    y = (-cα*cβ*sγ - sα*cγ)*v[1] + (-sα*cβ*sγ + cα*cγ)*v[2] + (sβ*sγ)*v[3]
    z = (cα*sβ)*v[1] + (sα*sβ)*v[2] + (cβ)*v[3]
    return Vec3(x, y, z)
end

"""
    cas_orientation(alpha::Real, beta::Real, gamma::Real) -> CASOrientation

Build the per-orientation geometry for CAS-v2 observables. The Euler angles
(`alpha`, `beta`, `gamma`) follow the ZYZ intrinsic convention used by
`scipy.spatial.transform.Rotation` and by block-DDA_Py.
"""
function cas_orientation(alpha::Real, beta::Real, gamma::Real)
    α = float(alpha); β = float(beta); γ = float(gamma)

    u_inc_L         = Vec3(0, 0, 1)
    theta_inc_L     = Vec3(1, 0, 0)
    phi_inc_L       = Vec3(0, 1, 0)
    u_sca_fw_L      = Vec3(0, 0, 1)
    theta_sca_fw_L  = Vec3(1, 0, 0)
    phi_sca_fw_L    = Vec3(0, 1, 0)
    u_sca_bk_L      = Vec3(0, 0, -1)
    theta_sca_bk_L  = Vec3(-1, 0, 0)
    phi_sca_bk_L    = Vec3(0, 1, 0)

    u_inc          = _zyz_inverse_apply(α, β, γ, u_inc_L)
    theta_inc      = _zyz_inverse_apply(α, β, γ, theta_inc_L)
    phi_inc        = _zyz_inverse_apply(α, β, γ, phi_inc_L)
    u_sca_fw       = _zyz_inverse_apply(α, β, γ, u_sca_fw_L)
    theta_sca_fw   = _zyz_inverse_apply(α, β, γ, theta_sca_fw_L)
    phi_sca_fw     = _zyz_inverse_apply(α, β, γ, phi_sca_fw_L)
    u_sca_bk       = _zyz_inverse_apply(α, β, γ, u_sca_bk_L)
    theta_sca_bk   = _zyz_inverse_apply(α, β, γ, theta_sca_bk_L)
    phi_sca_bk     = _zyz_inverse_apply(α, β, γ, phi_sca_bk_L)

    inv_sqrt2 = 1 / sqrt(2.0)
    e0_inc = SVector{3,ComplexF64}(
        inv_sqrt2 * theta_inc[1] + im * inv_sqrt2 * phi_inc[1],
        inv_sqrt2 * theta_inc[2] + im * inv_sqrt2 * phi_inc[2],
        inv_sqrt2 * theta_inc[3] + im * inv_sqrt2 * phi_inc[3])

    return CASOrientation(u_inc, theta_inc, phi_inc, e0_inc,
                          u_sca_fw, theta_sca_fw, phi_sca_fw,
                          u_sca_bk, theta_sca_bk, phi_sca_bk)
end

"""
    CASv2Result

Per-orientation CAS-v2 forward and backward scattering observables, in the
**physics convention** (e^{-iωt}, matching block-DDA_Py and Mie reference).

# Fields
- `S_fw_theta::ComplexF64` — forward scattering amplitude, θ-channel
- `S_fw_phi::ComplexF64`   — forward scattering amplitude, φ-channel
- `S_fw::ComplexF64`       — PCAS observable, `(S_fw_theta + S_fw_phi)/2`
- `S_bk_theta::ComplexF64` — backward scattering amplitude, θ-channel
- `S_bk_phi::ComplexF64`   — backward scattering amplitude, φ-channel
- `S_bk::ComplexF64`       — OCBS observable, `(-S_bk_theta + S_bk_phi)/√2`
"""
struct CASv2Result
    S_fw_theta::ComplexF64
    S_fw_phi::ComplexF64
    S_fw::ComplexF64
    S_bk_theta::ComplexF64
    S_bk_phi::ComplexF64
    S_bk::ComplexF64
end

@inline function _project_far_field(F::SVector{3,ComplexF64}, hat::Vec3)
    return ComplexF64(hat[1]) * F[1] + ComplexF64(hat[2]) * F[2] + ComplexF64(hat[3]) * F[3]
end

"""
    compute_cas_observables(basis, D_coeffs;
                            orientation::CASOrientation,
                            k0, eps_p, eps_bg = 1,
                            rule = TET_QUAD_5PT) -> CASv2Result

Compute the CAS-v2 forward and backward scattering observables for one
orientation. The result is returned in the physics convention (matching
block-DDA_Py and Mie reference).

The DDA observable formulas (block-DDA_Py `compute_PCAS_observable_S_fw`,
`compute_OCBS_observable_S_bk`) are reproduced verbatim with VIEM's
`far_field_amplitude` after taking its complex conjugate to convert from
VIEM's engineering convention (e^{+jωt}) to the physics convention.
"""
function compute_cas_observables(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                                 orientation::CASOrientation,
                                 k0::Number,
                                 eps_p::Number,
                                 eps_bg::Number = 1,
                                 rule::TetQuadRule = TET_QUAD_5PT)
    F_fw = far_field_amplitude(basis, D_coeffs;
                               k_hat_sca = orientation.u_sca_fw,
                               k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                               rule = rule)
    F_bk = far_field_amplitude(basis, D_coeffs;
                               k_hat_sca = orientation.u_sca_bk,
                               k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                               rule = rule)

    # Convention reconciliation:
    # - VIEM's `far_field_amplitude` returns F defined by
    #     E_sca = exp(-jk0 r)/(4π r) · F        (engineering, e^{+jωt})
    # - block-DDA_Py's CAS observables and the Mie reference use the BH83
    #   convention (physics, e^{-iωt}) with E_sca = exp(jkr)/r · f(r̂), so
    #     f = F/(4π) up to overall conjugation between conventions.
    # - The DDA formulas (S_θ = √2·k²·Σ(P·θ̂)e^{-jk r̂·r_n} etc.) applied
    #   verbatim to F^eng give the same complex value as the physics
    #   formulas applied to F^phys for the dominant (β,jβ,0)-type
    #   structure. Computing in eng then conjugating the final scalar at
    #   the end gives the right Re(S_fw) (matches Mie to ~0.05%).
    #
    # CONVENTION DIFFERENCE — α-dependence sign:
    #   For the spheroid analytical expansion S_θ(α) = A + B exp(±2jα),
    #   block-DDA_Py uses +2jα (right-circular pol in physics convention)
    #   while this VIEM implementation produces -2jα (the SAME complex
    #   amplitude (1, j)/√2 represents LEFT-circular pol in the
    #   engineering convention, which gives the opposite chirality).
    #   To match DDA exactly, callers should pass `-α` for the alpha Euler
    #   angle, OR negate the imaginary part of B in the analytical
    #   expansion. The PCAS S_fw = A and OCBS S_bk observables themselves
    #   are α-independent (= the (s_∥+s_⊥)/2 invariant), so this affects
    #   only the per-channel S_θ, S_φ when sweeping α at fixed β.
    #
    # KNOWN ISSUE (Stage 5.1): Im(S_fw) on a Mie sphere is ~38% low and
    # does not converge — pre-existing far_field_amplitude inconsistency.
    inv_4pi = 1 / (4π)
    sqrt2 = sqrt(2.0)

    # Forward (PCAS) — engineering convention
    S_fw_theta_eng = inv_4pi * sqrt2 * _project_far_field(F_fw, orientation.theta_sca_fw)
    S_fw_phi_eng   = inv_4pi * (-im) * sqrt2 * _project_far_field(F_fw, orientation.phi_sca_fw)

    # Backward (OCBS) — engineering convention
    S_bk_theta_eng = inv_4pi * sqrt2 * _project_far_field(F_bk, orientation.theta_sca_bk)
    S_bk_phi_eng   = inv_4pi * (-im) * sqrt2 * _project_far_field(F_bk, orientation.phi_sca_bk)

    # Engineering → physics: complex conjugation of the final scalar.
    S_fw_theta = conj(S_fw_theta_eng)
    S_fw_phi   = conj(S_fw_phi_eng)
    S_fw       = (S_fw_theta + S_fw_phi) / 2

    S_bk_theta = conj(S_bk_theta_eng)
    S_bk_phi   = conj(S_bk_phi_eng)
    S_bk       = (-S_bk_theta + S_bk_phi) / sqrt2

    return CASv2Result(S_fw_theta, S_fw_phi, S_fw,
                       S_bk_theta, S_bk_phi, S_bk)
end

"""
    solve_cas_v2_orientations(basis::AbstractDivBasis,
                              euler_angles::AbstractVector;
                              k0::Number,
                              eps_p::Number,
                              eps_bg::Number = 1,
                              duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                              outer_rule::TetQuadRule = TET_QUAD_5PT,
                              ff_rule::TetQuadRule = TET_QUAD_5PT,
                              symmetrize::Bool = true)
        -> Vector{CASv2Result}

Solve the VIEM system for many particle orientations and return the
CAS-v2 forward/backward observables for each. The impedance matrix `Z`
is assembled and LU-factorized once; each orientation reuses the same
factorization for the back-substitution.

`euler_angles` is an iterable of `(alpha, beta, gamma)` tuples (or any
indexable container with three elements), in the ZYZ convention used by
block-DDA_Py.

Use this for moderate-size problems where dense `Z` fits in memory
(N ≲ a few thousand DOFs). For larger problems, see the AIM-iterative
multi-orientation interface (TODO).
"""
function solve_cas_v2_orientations(basis::AbstractDivBasis,
                                   euler_angles::AbstractVector;
                                   k0::Number,
                                   eps_p::Number,
                                   eps_bg::Number = 1,
                                   duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                                   outer_rule::TetQuadRule = TET_QUAD_5PT,
                                   ff_rule::TetQuadRule = TET_QUAD_5PT,
                                   symmetrize::Bool = true)
    Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p,
                                  eps_bg = eps_bg,
                                  outer_rule = outer_rule,
                                  duffy_rule = duffy_rule,
                                  symmetrize = symmetrize)
    F = LinearAlgebra.lu(Z)
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))

    results = Vector{CASv2Result}(undef, length(euler_angles))
    for (i, ea) in enumerate(euler_angles)
        α, β, γ = ea[1], ea[2], ea[3]
        ori = cas_orientation(α, β, γ)
        b = project_plane_wave(basis; k_hat = ori.u_inc,
                               E0 = ori.e0_inc, k_bg = k_bg)
        D = F \ b
        results[i] = compute_cas_observables(basis, D; orientation = ori,
                                             k0 = k0, eps_p = eps_p,
                                             eps_bg = eps_bg, rule = ff_rule)
    end
    return results
end
