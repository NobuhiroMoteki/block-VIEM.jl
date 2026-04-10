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
    far_field_amplitude(basis::SWGBasis, D_coeffs::AbstractVector;
                        k_hat_sca::Vec3,
                        k0::Number,
                        eps_p::Number,
                        eps_bg::Number = 1,
                        rule::TetQuadRule = TET_QUAD_5PT)
        -> SVector{3,ComplexF64}

Compute the vector scattering amplitude `F(k̂_sca)` at observation direction
`k_hat_sca` (unit vector) for a particle with permittivity `eps_p` in
background `eps_bg`, given the solved SWG expansion coefficients `D_coeffs`.
"""
function far_field_amplitude(basis::SWGBasis, D_coeffs::AbstractVector;
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
        for tet in (basis.tet_plus[n], basis.tet_minus[n])
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

"""
    compute_scattering(basis::SWGBasis, D_coeffs::AbstractVector;
                       k_hat::Vec3,
                       E0::SVector{3,ComplexF64},
                       k0::Number,
                       eps_p::Number,
                       eps_bg::Number = 1,
                       rule::TetQuadRule = TET_QUAD_5PT)
        -> ScatteringResult

Compute all CAS-v2 scattering observables for a single orientation.

Arguments:
- `k_hat`: unit propagation direction of the incident plane wave
- `E0`: complex polarization vector (`E^inc = E0 exp(-jk₀ k̂·r)`)
- The s/p decomposition uses `ê_s = Re(E0)/|Re(E0)|` and
  `ê_p = k̂ × ê_s` (valid for linearly-polarized incidence).
"""
function compute_scattering(basis::SWGBasis, D_coeffs::AbstractVector;
                            k_hat::Vec3,
                            E0::SVector{3,ComplexF64},
                            k0::Number,
                            eps_p::Number,
                            eps_bg::Number = 1,
                            rule::TetQuadRule = TET_QUAD_5PT)
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
    # Scalar backward amplitude: project onto the back-scattered s-polarization
    # (convention: same ê_s basis).
    S_bak = dot(SVector{3,ComplexF64}(e_s), F_bak)

    # --- extinction cross section (optical theorem) ---
    # With the consistent exp(+jωt) convention where G = exp(-jkR)/(4πR) and
    # F = (k0²κ/ε_bg)(I−r̂r̂)P, the optical theorem gives:
    #
    #   C_ext = Im[E0* · F(k̂)] / (k0 |E0|²)
    #
    E0_sq = real(dot(E0, E0))     # |E0|² for unit amplitude
    C_ext = imag(dot(conj.(E0), F_fw)) / (real(k0c) * E0_sq)

    # --- absorption cross section ---
    # C_abs = k0 Im(ε_p) / (ε_bg |ε_p|² |E0|²) D^H M D
    M = assemble_mass_matrix(basis; rule = rule)
    DhMD = real(dot(D_coeffs, M * D_coeffs))
    C_abs = real(k0c) * imag(eps_p_c) / (real(eps_bg_c) * abs2(eps_p_c) * E0_sq) *
            DhMD

    C_sca = C_ext - C_abs

    return ScatteringResult(S_fw_s, S_fw_p, S_bak, C_ext, C_abs, C_sca)
end

@inline function _real_unit(E0::SVector{3,ComplexF64})
    v = Vec3(real(E0[1]), real(E0[2]), real(E0[3]))
    n = norm(v)
    return n > 0 ? v / n : Vec3(1, 0, 0)
end
