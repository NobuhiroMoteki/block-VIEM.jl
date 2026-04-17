# Far-field scattering amplitudes, cross sections, and CAS-v2 observables.
#
# Phase 5 of BlockVIEM.jl.
#
# The scattered far field from a homogeneous particle with solved D-field
# expansion coefficients D_n is (physics convention, e^{-iωt}):
#
#   E^sca(r) ≈ [exp(+ik0 r) / (4π r)] F(r̂)
#
# where the vector scattering amplitude is
#
#   F(r̂) = (k0² κ / ε_bg) (I − r̂r̂) · P(r̂)
#
# and P(r̂) = Σ_n D_n ∫ f_n(r') exp(-ik0 r̂·r') dV' is the Fourier-
# transformed polarization. The transverse projector (I − r̂r̂) removes the
# longitudinal component, ensuring the far field is purely transverse. The
# standard scattering amplitude is f(r̂) = F(r̂) / (4π).

using LinearAlgebra: dot, norm
import LinearAlgebra

"""
    far_field_amplitude(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                        k_hat_sca::Vec3,
                        k0::Number,
                        eps_p,
                        eps_bg::Number = 1,
                        rule::TetQuadRule = TET_QUAD_5PT)
        -> SVector{3,ComplexF64}

Compute the vector scattering amplitude `F(k̂_sca)` at observation direction
`k_hat_sca` (unit vector). Works for any `AbstractDivBasis`.
"""
function far_field_amplitude(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                             k_hat_sca::Vec3,
                             k0::Number,
                             eps_p,
                             eps_bg::Number = 1,
                             rule::TetQuadRule = TET_QUAD_5PT)
    k0c = ComplexF64(k0)
    _, eps_bg_c, _, kappa_v, _ = _aniso_params(eps_p, eps_bg)

    # PHYSICS CONVENTION: P(r̂) = Σ_n D_n ∫ f_n(r') exp(-ik0 r̂·r') dV'
    # (Far-field expansion of exp(+ik0|r-r'|)/(4π|r-r'|) for r→∞.)
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
                phase = exp(-im * k0c * dot(k_hat_sca, r))
                w = rule.weights[i] * V * Dn * phase
                Px += w * fn[1]
                Py += w * fn[2]
                Pz += w * fn[3]
            end
        end
    end
    P = SVector{3,ComplexF64}(Px, Py, Pz)

    # Anisotropic far-field: F = (k0²/ε_bg)(I − r̂r̂)·(κ̄·P)
    # For isotropic κ this reduces to F = (k0²κ/ε_bg)(I − r̂r̂)·P.
    kP = SVector{3,ComplexF64}(kappa_v[1]*P[1], kappa_v[2]*P[2], kappa_v[3]*P[3])
    kP_trans = kP - dot(k_hat_sca, kP) * SVector{3,ComplexF64}(k_hat_sca)
    coeff = k0c^2 / eps_bg_c
    return coeff * kP_trans
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

"""
Anisotropic absorption: C_abs = (k₀/ε_bg) Σ_α Im(ε_α)/|ε_α|² ∫|D_α|²dV.
Requires component-wise mass integrals ∫ (f_m)_α (f_n)_α dV.
"""
function _absorption_anisotropic(basis, D_coeffs, eps_p_v, eps_bg_c,
                                  k0c, E0_sq, rule)
    N = length(D_coeffs)
    C_abs = 0.0
    for α in 1:3
        abs_coeff = imag(eps_p_v[α]) / abs2(eps_p_v[α])
        iszero(abs_coeff) && continue
        # Component-α mass: M_α[m,n] = ∫ (f_m)_α (f_n)_α dV
        DhMaD = 0.0
        for m in 1:N, n in 1:N
            Dm = D_coeffs[m]; Dn = D_coeffs[n]
            (iszero(Dm) || iszero(Dn)) && continue
            tets_m = support_tets(basis, m)
            tets_n = support_tets(basis, n)
            val = 0.0
            @inbounds for tm in tets_m, tn in tets_n
                (tm == 0 || tn == 0) && continue
                tm == tn || continue
                verts = _tet_vertices(basis.mesh, tm)
                V = tet_volume(verts...)
                for i in 1:rule.n
                    r = bary_to_point(rule.bary[i], verts)
                    fm = evaluate(basis, m, r, tm)
                    fn = evaluate(basis, n, r, tm)
                    val += rule.weights[i] * V * fm[α] * fn[α]
                end
            end
            DhMaD += real(conj(Dm) * Dn * val)
        end
        C_abs += abs_coeff * DhMaD
    end
    C_abs *= real(k0c) / (real(eps_bg_c) * E0_sq)
    return C_abs
end

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
                          k0::Number, eps_p, eps_bg::Number = 1,
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
                               eps_p,
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
                            eps_p,
                            eps_bg::Number = 1,
                            rule::TetQuadRule = TET_QUAD_5PT,
                            csca_method::Symbol = :farfield,
                            n_theta::Int = 10)
    k0c = ComplexF64(k0)
    eps_p_v, eps_bg_c, _, _, _ = _aniso_params(eps_p, eps_bg)

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

    # --- absorption cross section ---
    # Anisotropic: C_abs = (k₀/ε_bg) Σ_α Im(ε_α)/|ε_α|² ∫|D_α|²dV
    # For isotropic this reduces to k₀ Im(ε_p)/(ε_bg |ε_p|²) D†MD.
    M = assemble_mass_matrix(basis; rule = rule)
    is_iso = (eps_p_v[1] == eps_p_v[2] == eps_p_v[3])
    if is_iso
        DhMD = real(dot(D_coeffs, M * D_coeffs))
        C_abs = real(k0c) * imag(eps_p_v[1]) /
                (real(eps_bg_c) * abs2(eps_p_v[1]) * E0_sq) * DhMD
    else
        C_abs = _absorption_anisotropic(basis, D_coeffs, eps_p_v, eps_bg_c,
                                         k0c, E0_sq, rule)
    end

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
        # PHYSICS convention: C_ext = +Im(E0*·F)/k (was -Im in eng convention).
        C_ext = imag(dot(conj.(E0), F_fw)) / (real(k0c) * E0_sq)
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
# Sign convention: VIEM uses the physics convention (e^{-iωt}) consistently
# in green.jl, incident.jl, and far_field_amplitude after the 2026-04-13
# convention switch. The CAS-v2 observables returned here are directly
# comparable to block-DDA_Py and to the Mie reference (S_fw = (S11(0)+S22(0))/2
# from `analytical_scattering_theories/homogeneous_sphere.py`).

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

# Apply the lab→particle coordinate transform for intrinsic ZYZ Euler
# angles (α, β, γ), matching `scipy.spatial.transform.Rotation` and
# block-DDA_Py. Convention follows §2.2 of docs/theory_note.tex:
#
#   R(α,β,γ) = Rz(α) Ry(β) Rz(γ)
#
# is the active rotation mapping particle-frame coordinates to lab-frame
# coordinates (equivalently, the fixed-frame composition obtained by
# applying Rz(γ), Ry(β), Rz(α) in that order about lab axes). The
# lab→particle transform is therefore R(α,β,γ)^T, and this routine
# computes R(α,β,γ)^T · v for a lab-frame vector v.
@inline function _zyz_inverse_apply(α::Real, β::Real, γ::Real, v::Vec3)
    sα, cα = sincos(α)
    sβ, cβ = sincos(β)
    sγ, cγ = sincos(γ)
    # R(α,β,γ) (particle→lab active rotation) =
    #   [ cα cβ cγ - sα sγ,  -cα cβ sγ - sα cγ,  cα sβ ]
    #   [ sα cβ cγ + cα sγ,  -sα cβ sγ + cα cγ,  sα sβ ]
    #   [          -sβ cγ,             sβ sγ,       cβ ]
    # v_particle = R(α,β,γ)^T · v_lab
    # (rows of R become columns below).
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
`compute_OCBS_observable_S_bk`) are reproduced directly using VIEM's
`far_field_amplitude`, which is in physics convention since the
2026-04-13 convention switch.
"""
function compute_cas_observables(basis::AbstractDivBasis, D_coeffs::AbstractVector;
                                 orientation::CASOrientation,
                                 k0::Number,
                                 eps_p,
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

    # PHYSICS CONVENTION (e^{-iωt}, matching block-DDA_Py and Mie reference):
    # `far_field_amplitude` returns F defined by E_sca = exp(+ik0 r)/(4π r)·F
    # with integrand exp(-ik0 r̂·r'). The physics scattering amplitude is
    # f = F/(4π), and the DDA observable formulas apply directly:
    #     S_θ = √2 · (f · θ̂)
    #     S_φ = -j · √2 · (f · φ̂)
    inv_4pi = 1 / (4π)
    sqrt2 = sqrt(2.0)

    # Forward (PCAS)
    S_fw_theta = inv_4pi * sqrt2 * _project_far_field(F_fw, orientation.theta_sca_fw)
    S_fw_phi   = inv_4pi * (-im) * sqrt2 * _project_far_field(F_fw, orientation.phi_sca_fw)
    S_fw       = (S_fw_theta + S_fw_phi) / 2

    # Backward (OCBS)
    S_bk_theta = inv_4pi * sqrt2 * _project_far_field(F_bk, orientation.theta_sca_bk)
    S_bk_phi   = inv_4pi * (-im) * sqrt2 * _project_far_field(F_bk, orientation.phi_sca_bk)
    S_bk       = (-S_bk_theta + S_bk_phi) / sqrt2

    return CASv2Result(S_fw_theta, S_fw_phi, S_fw,
                       S_bk_theta, S_bk_phi, S_bk)
end

# Resolve the two equivalent physical-input forms accepted by
# `solve_cas_v2_orientations` into the `(k0, eps_p, eps_bg)` triple
# consumed by `assemble_impedance_matrix`, `build_aim_operator`, and
# `compute_cas_observables`.
#
# Form A (block-DDA_Py compatible): supply `(wl_0, m_m, m_p)`. The
# resolver sets
#     k0     = 2π · m_m / wl_0       (wavenumber in background medium)
#     eps_p  = m_p^2                 (absolute particle permittivity)
#     eps_bg = m_m^2                 (absolute background permittivity)
#
# Form B (raw VIEM): supply `(k0, eps_p[, eps_bg])`. `k0` must already
# be the background-medium wavenumber. `eps_bg` defaults to `1`.
function _resolve_physical_inputs(wl_0, m_m, m_p, k0, eps_p, eps_bg)
    have_phys = (wl_0 !== nothing) || (m_m !== nothing) || (m_p !== nothing)
    have_raw  = (k0  !== nothing) || (eps_p !== nothing)
    if have_phys && have_raw
        throw(ArgumentError(
            "solve_cas_v2_orientations: pass either (wl_0, m_m, m_p) OR " *
            "(k0, eps_p, eps_bg), not both."))
    elseif have_phys
        (wl_0 !== nothing && m_m !== nothing && m_p !== nothing) ||
            throw(ArgumentError(
                "solve_cas_v2_orientations: wl_0, m_m, m_p must all be " *
                "provided together."))
        k0_out     = ComplexF64(2π * m_m / wl_0)
        eps_bg_out = ComplexF64(m_m)^2
        # m_p can be scalar (isotropic) or 3-vector (anisotropic)
        if m_p isa Number
            eps_p_out = ComplexF64(m_p)^2
        else
            eps_p_out = _eps_p_vec(ComplexF64.(m_p) .^ 2)
        end
        return k0_out, eps_p_out, eps_bg_out
    elseif have_raw
        (k0 !== nothing && eps_p !== nothing) ||
            throw(ArgumentError(
                "solve_cas_v2_orientations: both k0 and eps_p must be " *
                "provided in the raw form."))
        k0_out     = ComplexF64(k0)
        # eps_p can be scalar or 3-vector
        eps_p_out  = eps_p isa Number ? ComplexF64(eps_p) : _eps_p_vec(eps_p)
        eps_bg_out = ComplexF64(eps_bg === nothing ? 1 : eps_bg)
        return k0_out, eps_p_out, eps_bg_out
    else
        throw(ArgumentError(
            "solve_cas_v2_orientations: must supply either " *
            "(wl_0, m_m, m_p) or (k0, eps_p[, eps_bg])."))
    end
end

"""
    solve_cas_v2_orientations(basis::AbstractDivBasis,
                              euler_angles::AbstractVector;
                              # --- block-DDA_Py-compatible physical inputs ---
                              wl_0::Union{Real,Nothing}   = nothing,
                              m_m::Union{Real,Nothing}    = nothing,
                              m_p::Union{Number,Nothing}  = nothing,
                              # --- or the raw VIEM inputs ---
                              k0::Union{Number,Nothing}    = nothing,
                              eps_p::Union{Number,Nothing} = nothing,
                              eps_bg::Union{Number,Nothing} = nothing,
                              duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                              outer_rule::TetQuadRule = TET_QUAD_5PT,
                              ff_rule::TetQuadRule = TET_QUAD_5PT,
                              symmetrize::Bool = true,
                              method::Symbol = :aim_bicgstab,
                              pitch::Union{Float64,Nothing} = nothing,
                              padding::Integer = 4,
                              tol::Float64 = 1e-6,
                              maxiter::Integer = 200,
                              verbose::Bool = false)
        -> Vector{CASv2Result}

Solve the VIEM system for many particle orientations and return the
CAS-v2 forward/backward observables for each.

# Physical inputs — two equivalent forms

Either pass the **block-DDA_Py-compatible form**
- `wl_0`  — vacuum wavelength (same length unit as the mesh)
- `m_m`   — background medium refractive index (real)
- `m_p`   — particle complex refractive index (absorbing: `Im(m_p) > 0`)

and the function will internally set
`k0 = 2π·m_m/wl_0`, `eps_p = m_p^2`, `eps_bg = m_m^2` — exactly matching
the convention used by block-DDA_Py so results are directly comparable.

Or pass the **raw VIEM form**
- `k0`    — wavenumber **in the background medium** (`2π·m_m/λ₀`, NOT
            the vacuum wavenumber)
- `eps_p` — absolute particle permittivity
- `eps_bg` — absolute background permittivity (default `1`)

Mixing the two forms, or supplying neither complete set, raises an
`ArgumentError`.

# Solver selection

`method` selects how the multi-RHS system is solved:
- `:aim_bicgstab` — AIM FFT-MVP + Block BiCGSTAB (**default**). If `pitch`
                    is not supplied it is set to `0.5 × mean_edge_length`.
- `:aim_gmres`    — AIM FFT-MVP + Block GMRES (same pitch auto-detection).
- `:dense`        — assemble `Z` and LU-factorize once, then reuse
                    across orientations. Only for small problems (`N ≲ 10³`).

`euler_angles` is an iterable of `(alpha, beta, gamma)` tuples in the
intrinsic ZYZ convention, matching `scipy.spatial.transform.Rotation`
and block-DDA_Py.
"""
function solve_cas_v2_orientations(basis::AbstractDivBasis,
                                   euler_angles::AbstractVector;
                                   wl_0::Union{Real,Nothing}   = nothing,
                                   m_m::Union{Real,Nothing}    = nothing,
                                   m_p = nothing,
                                   k0::Union{Number,Nothing}    = nothing,
                                   eps_p = nothing,
                                   eps_bg::Union{Number,Nothing} = nothing,
                                   duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                                   outer_rule::TetQuadRule = TET_QUAD_5PT,
                                   ff_rule::TetQuadRule = TET_QUAD_5PT,
                                   symmetrize::Bool = true,
                                   method::Symbol = :aim_bicgstab,
                                   pitch::Union{Float64,Nothing} = nothing,
                                   padding::Integer = 4,
                                   tol::Float64 = 1e-6,
                                   maxiter::Integer = 200,
                                   verbose::Bool = false,
                                   return_D::Bool = false)
    k0_c, eps_p_c, eps_bg_c = _resolve_physical_inputs(wl_0, m_m, m_p,
                                                       k0, eps_p, eps_bg)
    # `k0_c` is the wavenumber in the background medium, matching
    # block-DDA_Py's `self.k = 2π·m_m/wl_0`.
    k_bg = k0_c
    orientations = [cas_orientation(ea[1], ea[2], ea[3]) for ea in euler_angles]
    L = length(orientations)

    local D_block::Matrix{ComplexF64}

    if method === :dense
        Z = assemble_impedance_matrix(basis; k0 = k0_c, eps_p = eps_p_c,
                                      eps_bg = eps_bg_c,
                                      outer_rule = outer_rule,
                                      duffy_rule = duffy_rule,
                                      symmetrize = symmetrize)
        F = LinearAlgebra.lu(Z)
        N = size(Z, 1)
        D_block = Matrix{ComplexF64}(undef, N, L)
        for (i, ori) in enumerate(orientations)
            b = project_plane_wave(basis; k_hat = ori.u_inc,
                                   E0 = ori.e0_inc, k_bg = k_bg)
            @views D_block[:, i] .= F \ b
        end
    elseif method === :aim_bicgstab || method === :aim_gmres
        # Auto-detect pitch from mean edge length if not supplied
        if pitch === nothing
            pitch = 0.5 * mean_edge_length(basis.mesh)
        end
        N = n_basis(basis)
        B = Matrix{ComplexF64}(undef, N, L)
        for (i, ori) in enumerate(orientations)
            B[:, i] = project_plane_wave(basis; k_hat = ori.u_inc,
                                         E0 = ori.e0_inc, k_bg = k_bg,
                                         rule = outer_rule)
        end
        op = build_aim_operator(basis; k0 = k0_c, eps_p = eps_p_c,
                                eps_bg = eps_bg_c,
                                pitch = pitch, padding = padding,
                                outer_rule = outer_rule,
                                duffy_rule = duffy_rule)
        A = _AIMLinOp(op, N)
        sub = method === :aim_bicgstab ? :bicgstab : :gmres
        res = _block_solve(A, B, sub; tol = tol, maxiter = maxiter,
                           verbose = verbose)
        D_block = res.X
        verbose && @info "solve_cas_v2_orientations (AIM block Krylov)" method =
            method iterations = res.iterations residual = res.residual_norm
    else
        throw(ArgumentError("unknown method: $method " *
                            "(expected :dense, :aim_bicgstab, or :aim_gmres)"))
    end

    results = Vector{CASv2Result}(undef, L)
    for i in 1:L
        results[i] = compute_cas_observables(basis, @view D_block[:, i];
                                             orientation = orientations[i],
                                             k0 = k0_c, eps_p = eps_p_c,
                                             eps_bg = eps_bg_c, rule = ff_rule)
    end
    return return_D ? (results, D_block) : results
end
