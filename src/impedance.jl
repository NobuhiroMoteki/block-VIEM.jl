# Impedance matrix element Z_mn for the SWG/EFVIE-D formulation.
# Reference: technical_note.md §3-§4 (weakened pair integral).
#
#     Z_mn = ∫ (f_m · f_n) / ε(r) dV
#            - (1/ε_bg) ∫∫ κ(r') [k0² f_m·f_n' - (∇·f_m)(∇'·f_n')] G(R) dV' dV
#
# This implementation assumes a homogeneous particle: every tetrahedron in
# the SWG basis support shares the same complex permittivity `eps_p`. The
# contrast κ = (eps_p - eps_bg) / eps_p is then constant and factors out
# of the inner integral, yielding
#
#     Z_mn = (1/eps_p) M_mn - (κ/eps_bg) K_mn
#
# where
#
#     M_mn = ∫_{T_m ∩ T_n} f_m·f_n dV                              (mass)
#     K_mn = Σ_{σ,τ ∈ ±} ∫_{T_m^σ} ∫_{T_n^τ}
#              [ k0² f_m^σ(r)·f_n^τ(r') - (∇·f_m^σ)(∇'·f_n^τ) ]
#              G(|r-r'|) dV' dV                                  (radiation)
#
# Quadrature strategy
# -------------------
# - Mass term: 5-point tetrahedron Gauss rule (degree 3) — exact for the
#   degree-2 product `f_m · f_n`.
# - Radiation outer integral: 5-point tetrahedron Gauss rule.
# - Radiation inner integral, when `m_tet != n_tet`: 5-point Gauss rule.
# - Radiation inner integral, when `m_tet == n_tet` (self pair):
#   `duffy_quadrature_around` with the outer Gauss point as the singular
#   vertex. The Duffy `u1²` Jacobian cancels the 1/R singularity in `G`.
#
# The accuracy of the simple "non-self uses Gauss" choice degrades for
# edge/face-adjacent pairs; a true near-singular Duffy treatment with a
# virtual singular vertex on the shared feature is deferred to a later
# part of Phase 2.

using LinearAlgebra: dot

"""
    impedance_element(basis::SWGBasis, m::Integer, n::Integer;
                      k0::Number,
                      eps_p::Number = 1,
                      eps_bg::Number = 1,
                      outer_rule::TetQuadRule = TET_QUAD_5PT,
                      duffy_rule::DuffyQuadRule = duffy_reference_rule(5))
        -> ComplexF64

Compute the Galerkin impedance matrix element `Z_mn` between SWG basis
functions `m` and `n` for a homogeneous particle of complex permittivity
`eps_p` immersed in a background of permittivity `eps_bg`. `k0` is the
background-medium wavenumber.

Returns a `ComplexF64` regardless of the numeric types of `k0`, `eps_p`,
`eps_bg`. With `eps_p == eps_bg` the contrast `κ` vanishes and the result
reduces to the geometric mass inner product `(1/eps_bg) ∫ f_m·f_n dV`.
"""
function impedance_element(basis::SWGBasis, m::Integer, n::Integer;
                           k0::Number,
                           eps_p::Number = 1,
                           eps_bg::Number = 1,
                           outer_rule::TetQuadRule = TET_QUAD_5PT,
                           duffy_rule::DuffyQuadRule = duffy_reference_rule(5))
    eps_p_c = ComplexF64(eps_p)
    eps_bg_c = ComplexF64(eps_bg)
    k0_c = ComplexF64(k0)
    inv_eps = 1 / eps_p_c
    kappa = (eps_p_c - eps_bg_c) / eps_p_c

    M = _mass_term(basis, Int(m), Int(n), outer_rule)
    Z = inv_eps * M
    if !iszero(kappa)
        K = _radiation_kernel(basis, Int(m), Int(n), k0_c, outer_rule, duffy_rule)
        Z -= (kappa / eps_bg_c) * K
    end
    return Z
end

# ---------------------------------------------------------------------------
# Mass term  M_mn = ∫_{T_m ∩ T_n} f_m·f_n dV  (real for real geometry)
# ---------------------------------------------------------------------------
function _mass_term(basis::SWGBasis, m::Int, n::Int, rule::TetQuadRule)
    tets_m = (basis.tet_plus[m], basis.tet_minus[m])
    tets_n = (basis.tet_plus[n], basis.tet_minus[n])
    s = 0.0
    @inbounds for tm in tets_m, tn in tets_n
        tm == tn || continue
        verts = _tet_vertices(basis.mesh, tm)
        V = tet_volume(verts...)
        for i in 1:rule.n
            r = bary_to_point(rule.bary[i], verts)
            f_m_r = evaluate(basis, m, r, tm)
            f_n_r = evaluate(basis, n, r, tm)
            s += rule.weights[i] * V * dot(f_m_r, f_n_r)
        end
    end
    return s
end

# ---------------------------------------------------------------------------
# Radiation kernel  K_mn  (sum over the four (T_m^σ, T_n^τ) pairs)
# ---------------------------------------------------------------------------
function _radiation_kernel(basis::SWGBasis, m::Int, n::Int, k0::ComplexF64,
                           outer_rule::TetQuadRule, duffy_rule::DuffyQuadRule)
    tets_m = (basis.tet_plus[m], basis.tet_minus[m])
    tets_n = (basis.tet_plus[n], basis.tet_minus[n])
    s = zero(ComplexF64)
    @inbounds for tm in tets_m, tn in tets_n
        s += _radiation_pair(basis, m, n, tm, tn, k0, outer_rule, duffy_rule)
    end
    return s
end

function _radiation_pair(basis::SWGBasis, m::Int, n::Int,
                         m_tet::Int, n_tet::Int, k0::ComplexF64,
                         outer_rule::TetQuadRule, duffy_rule::DuffyQuadRule)
    verts_m = _tet_vertices(basis.mesh, m_tet)
    verts_n = _tet_vertices(basis.mesh, n_tet)
    Vm = tet_volume(verts_m...)
    Vn = tet_volume(verts_n...)
    div_m = divergence(basis, m, m_tet)
    div_n = divergence(basis, n, n_tet)
    k0_sq = k0 * k0
    self_pair = (m_tet == n_tet)

    s = zero(ComplexF64)
    for i in 1:outer_rule.n
        r = bary_to_point(outer_rule.bary[i], verts_m)
        outer_w = outer_rule.weights[i] * Vm
        f_m_r = evaluate(basis, m, r, m_tet)

        if self_pair
            inner_pts, inner_wts = duffy_quadrature_around(verts_n, r, duffy_rule)
        else
            inner_pts, inner_wts = _gauss_pts_wts(outer_rule, verts_n, Vn)
        end

        inner = zero(ComplexF64)
        @inbounds for j in eachindex(inner_pts)
            rp = inner_pts[j]
            R = norm(r - rp)
            R == 0 && continue
            G = helmholtz_green(R, k0)
            f_n_rp = evaluate(basis, n, rp, n_tet)
            kernel = k0_sq * dot(f_m_r, f_n_rp) - div_m * div_n
            inner += inner_wts[j] * kernel * G
        end
        s += outer_w * inner
    end
    return s
end

@inline function _tet_vertices(mesh::TetMesh, tet::Int)
    a, b, c, d = mesh.tets[tet]
    return (mesh.nodes[a], mesh.nodes[b], mesh.nodes[c], mesh.nodes[d])
end

@inline function _gauss_pts_wts(rule::TetQuadRule, verts::NTuple{4,Vec3}, V::Float64)
    n = rule.n
    pts = Vector{Vec3}(undef, n)
    wts = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        pts[i] = bary_to_point(rule.bary[i], verts)
        wts[i] = rule.weights[i] * V
    end
    return pts, wts
end
