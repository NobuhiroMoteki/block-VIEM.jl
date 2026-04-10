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

"""
    _tets_share_nodes(mesh, t1, t2) -> Bool

True if tetrahedra `t1` and `t2` share at least one vertex (face/edge/vertex
adjacency). Used to trigger near-singular Duffy quadrature on cross-tet pairs.
"""
@inline function _tets_share_nodes(mesh::TetMesh, t1::Int, t2::Int)
    n1 = mesh.tets[t1]
    n2 = mesh.tets[t2]
    @inbounds for a in n1, b in n2
        a == b && return true
    end
    return false
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
    # Use Duffy quadrature only for self-tet pairs (r is inside n_tet by
    # construction). For cross-tet pairs — including adjacent tets that share
    # nodes — the observation point r lies outside n_tet, so the 1/R
    # integrand has no singularity but may be nearly singular. We handle
    # this via higher-order Gauss quadrature on the inner tet.
    self_pair = (m_tet == n_tet)

    # For adjacent cross-tet pairs, use the Duffy rule's underlying GL
    # points as a higher-order product rule on the inner tet, giving better
    # near-singular accuracy than the default 5-point rule.
    near_cross = !self_pair && _tets_share_nodes(basis.mesh, m_tet, n_tet)

    s = zero(ComplexF64)
    for i in 1:outer_rule.n
        r = bary_to_point(outer_rule.bary[i], verts_m)
        outer_w = outer_rule.weights[i] * Vm
        f_m_r = evaluate(basis, m, r, m_tet)

        if self_pair
            inner_pts, inner_wts = duffy_quadrature_around(verts_n, r, duffy_rule)
        elseif near_cross
            # Use Duffy rule as a high-order tet quadrature (singular vertex
            # at tet vertex 1, but r is outside so no true singularity — we
            # just need more quadrature points).
            inner_pts, inner_wts = duffy_quadrature(verts_n, 1, duffy_rule)
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

# ---------------------------------------------------------------------------
# Full Z-matrix assembly
# ---------------------------------------------------------------------------

"""
    assemble_impedance_matrix(basis::SWGBasis;
                              k0::Number,
                              eps_p::Number = 1,
                              eps_bg::Number = 1,
                              outer_rule::TetQuadRule = TET_QUAD_5PT,
                              duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                              symmetrize::Bool = false)
        -> Matrix{ComplexF64}

Build the dense `N × N` SWG impedance matrix for `basis` by calling
[`impedance_element`](@ref) on every pair `(m, n)`. The outer loop over
`m` is parallelized with `Threads.@threads`; set the environment variable
`JULIA_NUM_THREADS` (or launch Julia with `-t auto`) to obtain a speed-up.

If `symmetrize = true`, the result is replaced by `(Z + transpose(Z)) / 2`
to absorb the small (`~1e-5`) reciprocity gap introduced by the
asymmetric outer-Gauss / inner-Duffy quadrature on self-tet pairs. Use
this for symmetric Krylov solvers; leave it `false` if you want the raw
quadrature output (e.g., for AIM precorrection comparisons).

Both `outer_rule` and `duffy_rule` are constructed once and reused across
the entire matrix to avoid repeated allocation.
"""
function assemble_impedance_matrix(basis::SWGBasis;
                                   k0::Number,
                                   eps_p::Number = 1,
                                   eps_bg::Number = 1,
                                   outer_rule::TetQuadRule = TET_QUAD_5PT,
                                   duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                                   symmetrize::Bool = false)
    N = n_basis(basis)
    Z = Matrix{ComplexF64}(undef, N, N)
    Threads.@threads for m in 1:N
        @inbounds for n in 1:N
            Z[m, n] = impedance_element(basis, m, n;
                                        k0 = k0,
                                        eps_p = eps_p,
                                        eps_bg = eps_bg,
                                        outer_rule = outer_rule,
                                        duffy_rule = duffy_rule)
        end
    end
    if symmetrize
        @inbounds for m in 1:N
            for n in (m + 1):N
                avg = (Z[m, n] + Z[n, m]) / 2
                Z[m, n] = avg
                Z[n, m] = avg
            end
        end
    end
    return Z
end
