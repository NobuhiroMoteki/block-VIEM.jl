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
