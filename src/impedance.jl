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
using SparseArrays: SparseMatrixCSC, sparse, spzeros

# ── Anisotropic permittivity helpers ─────────────────────────────────────────
#
# For diagonal-tensor anisotropic ε_p = diag(ε_x, ε_y, ε_z), the EFVIE-D
# weakened form becomes:
#
#   Z_mn = Σ_α (1/ε_α) ∫ (f_m)_α (f_n)_α dV
#        - (1/ε_bg) ∫∫ [k0² Σ_α κ_α (f_m)_α (f_n)_α
#                        − κ_avg (∇·f_m)(∇'·f_n)] G dV' dV
#
# where κ_α = (ε_α − ε_bg)/ε_α and κ_avg = (κ_x + κ_y + κ_z)/3.
# The κ_avg simplification for the ∇∇· term exploits the SWG structure:
# ∂(f_n)_α/∂r_α = a_n/(3V) for all α, so ∇·(κ̄·f_n) = κ_avg · ∇·f_n.
#
# Half-SWG surface terms (K^B, K^C, K^D) use κ_avg as an approximation;
# this is exact for isotropic ε_p and introduces O(|Δκ|/|κ|) error for
# mildly anisotropic materials.

"""
    _eps_p_vec(eps_p) -> SVector{3,ComplexF64}

Promote any `eps_p` input (scalar or 3-vector) to `SVector{3,ComplexF64}`.
"""
_eps_p_vec(eps_p::Number) = let e = ComplexF64(eps_p); SVector{3,ComplexF64}(e,e,e) end
_eps_p_vec(eps_p::SVector{3}) = SVector{3,ComplexF64}(ComplexF64(eps_p[1]),
                                                       ComplexF64(eps_p[2]),
                                                       ComplexF64(eps_p[3]))
_eps_p_vec(eps_p::AbstractVector) = begin
    length(eps_p) == 3 || throw(ArgumentError("eps_p vector must have length 3"))
    SVector{3,ComplexF64}(ComplexF64(eps_p[1]), ComplexF64(eps_p[2]), ComplexF64(eps_p[3]))
end

"""
    _aniso_params(eps_p, eps_bg) -> (eps_p_v, eps_bg_c, inv_eps_v, kappa_v, kappa_avg)

Compute anisotropic derived quantities from `eps_p` (scalar or 3-vector)
and scalar `eps_bg`. Returns `SVector{3}` quantities for component-wise use.
"""
function _aniso_params(eps_p, eps_bg)
    ev = _eps_p_vec(eps_p)
    eb = ComplexF64(eps_bg)
    inv_eps_v = SVector{3,ComplexF64}(1/ev[1], 1/ev[2], 1/ev[3])
    kappa_v   = SVector{3,ComplexF64}((ev[1]-eb)/ev[1], (ev[2]-eb)/ev[2], (ev[3]-eb)/ev[3])
    kappa_avg = (kappa_v[1] + kappa_v[2] + kappa_v[3]) / 3
    return ev, eb, inv_eps_v, kappa_v, kappa_avg
end

"""
    impedance_element(basis::AbstractDivBasis, m::Integer, n::Integer;
                      k0::Number,
                      eps_p = 1,
                      eps_bg::Number = 1,
                      outer_rule::TetQuadRule = TET_QUAD_5PT,
                      duffy_rule::DuffyQuadRule = duffy_reference_rule(5))
        -> ComplexF64

Compute the Galerkin impedance matrix element `Z_mn` between basis
functions `m` and `n` for a homogeneous particle of complex permittivity
`eps_p` immersed in a background of permittivity `eps_bg`. `k0` is the
background-medium wavenumber.

Works for any `AbstractDivBasis` (SWGBasis, RT1Basis, etc.).
"""
function impedance_element(basis::AbstractDivBasis, m::Integer, n::Integer;
                           k0::Number,
                           eps_p = 1,
                           eps_bg::Number = 1,
                           outer_rule::TetQuadRule = TET_QUAD_5PT,
                           duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                           tri_rule::TriQuadRule = tri_collapsed_rule(4),
                           tri_duffy_rule::TriDuffyRule = tri_duffy_reference_rule(6))
    _, eps_bg_c, inv_eps_v, kappa_v, kappa_avg = _aniso_params(eps_p, eps_bg)
    k0_c = ComplexF64(k0)

    # Mass: Σ_α (1/ε_α) ∫ (f_m)_α (f_n)_α dV
    Z = _mass_term_weighted(basis, Int(m), Int(n), outer_rule, inv_eps_v)

    # Radiation: (1/ε_bg) [k0² Σ_α κ_α (f_m)_α(f_n)_α − κ_avg (∇·f_m)(∇·f_n)] G
    if !all(iszero, kappa_v)
        K = _radiation_kernel_weighted(basis, Int(m), Int(n), k0_c,
                                       kappa_v, kappa_avg,
                                       outer_rule, duffy_rule,
                                       tri_rule, tri_duffy_rule)
        Z -= (1 / eps_bg_c) * K
    end
    return Z
end

# ---------------------------------------------------------------------------
# Mass term  M_mn = ∫_{T_m ∩ T_n} f_m·f_n dV  (real for real geometry)
# ---------------------------------------------------------------------------
function _mass_term(basis::AbstractDivBasis, m::Int, n::Int, rule::TetQuadRule)
    tets_m = support_tets(basis, m)
    tets_n = support_tets(basis, n)
    s = 0.0
    @inbounds for tm in tets_m, tn in tets_n
        (tm == 0 || tn == 0) && continue
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

"""
Weighted mass term: Σ_α w_α ∫ (f_m)_α (f_n)_α dV.  For isotropic
`w = (1/ε, 1/ε, 1/ε)` this equals `(1/ε) M_mn`.
"""
function _mass_term_weighted(basis::AbstractDivBasis, m::Int, n::Int,
                             rule::TetQuadRule, w::SVector{3,ComplexF64})
    tets_m = support_tets(basis, m)
    tets_n = support_tets(basis, n)
    s = zero(ComplexF64)
    @inbounds for tm in tets_m, tn in tets_n
        (tm == 0 || tn == 0) && continue
        tm == tn || continue
        verts = _tet_vertices(basis.mesh, tm)
        V = tet_volume(verts...)
        for i in 1:rule.n
            r = bary_to_point(rule.bary[i], verts)
            f_m_r = evaluate(basis, m, r, tm)
            f_n_r = evaluate(basis, n, r, tm)
            s += rule.weights[i] * V * (w[1] * f_m_r[1] * f_n_r[1] +
                                        w[2] * f_m_r[2] * f_n_r[2] +
                                        w[3] * f_m_r[3] * f_n_r[3])
        end
    end
    return s
end

# ---------------------------------------------------------------------------
# Radiation kernel  K_mn  (sum over the four (T_m^σ, T_n^τ) pairs)
#
# For the standard (interior-face-only) SWG basis this is just the bulk-bulk
# double volume integral. When the basis contains half-SWG boundary DOFs
# (`basis.is_boundary[n] == true`), additional surface-correction terms
# from the half-SWG extension (§9 of technical_note.md) are added:
#
#   K^B  (n boundary)         : ∫_{T_m}(∇·f_m) ∫_{S_n} G dS dV
#   K^C  (m boundary)         : analogous to K^B with m ↔ n  (implemented in
#                               Stage 2.4)
#   K^D  (both boundary)      : ∫_{S_m}∫_{S_n} G dS'dS × (−1)   (Stage 2.5)
#
# The sign of K^B is + (per §9.4 derivation).
# ---------------------------------------------------------------------------
function _radiation_kernel(basis::AbstractDivBasis, m::Int, n::Int, k0::ComplexF64,
                           outer_rule::TetQuadRule, duffy_rule::DuffyQuadRule,
                           tri_rule::TriQuadRule, tri_duffy_rule::TriDuffyRule)
    s = _bulk_radiation_kernel(basis, m, n, k0, outer_rule, duffy_rule)
    s += _half_swg_surface_kernel(basis, m, n, k0, outer_rule, duffy_rule,
                                    tri_rule, tri_duffy_rule)
    return s
end

# Bulk-only part (k0² ∫∫ f·f' G + K^A). This is exactly what the AIM
# grid convolution approximates, so `assemble_precorrection` uses this
# instead of the full `_radiation_kernel` to avoid double-counting the
# Stage-2 half-SWG surface terms (which live entirely in the separate
# `half_swg_extra` sparse matrix).
function _bulk_radiation_kernel(basis::AbstractDivBasis, m::Int, n::Int,
                                  k0::ComplexF64,
                                  outer_rule::TetQuadRule,
                                  duffy_rule::DuffyQuadRule)
    tets_m = support_tets(basis, m)
    tets_n = support_tets(basis, n)
    s = zero(ComplexF64)
    @inbounds for tm in tets_m, tn in tets_n
        (tm == 0 || tn == 0) && continue
        s += _radiation_pair(basis, m, n, tm, tn, k0, outer_rule, duffy_rule)
    end
    return s
end

# Weighted variants for anisotropic ε_p.  The kernel becomes
#   k0² Σ_α κ_α (f_m)_α (f_n)_α G  −  κ_avg (∇·f_m)(∇'·f_n) G
# which reduces to κ [k0² f_m·f_n − (∇·f_m)(∇·f_n)] G for isotropic κ.

function _radiation_kernel_weighted(basis::AbstractDivBasis, m::Int, n::Int,
                                     k0::ComplexF64,
                                     kappa_v::SVector{3,ComplexF64},
                                     kappa_avg::ComplexF64,
                                     outer_rule::TetQuadRule,
                                     duffy_rule::DuffyQuadRule,
                                     tri_rule::TriQuadRule,
                                     tri_duffy_rule::TriDuffyRule)
    s = _bulk_radiation_kernel_weighted(basis, m, n, k0, kappa_v, kappa_avg,
                                        outer_rule, duffy_rule)
    # Half-SWG surface terms use κ_avg (exact for isotropic, approximate for aniso)
    s += kappa_avg * _half_swg_surface_kernel(basis, m, n, k0, outer_rule, duffy_rule,
                                               tri_rule, tri_duffy_rule)
    return s
end

function _bulk_radiation_kernel_weighted(basis::AbstractDivBasis, m::Int, n::Int,
                                          k0::ComplexF64,
                                          kappa_v::SVector{3,ComplexF64},
                                          kappa_avg::ComplexF64,
                                          outer_rule::TetQuadRule,
                                          duffy_rule::DuffyQuadRule)
    tets_m = support_tets(basis, m)
    tets_n = support_tets(basis, n)
    s = zero(ComplexF64)
    @inbounds for tm in tets_m, tn in tets_n
        (tm == 0 || tn == 0) && continue
        s += _radiation_pair_weighted(basis, m, n, tm, tn, k0,
                                       kappa_v, kappa_avg, outer_rule, duffy_rule)
    end
    return s
end

# Half-SWG surface correction K^B + K^C + K^D (unscaled by -κ/ε_bg).
# Called both directly from `_radiation_kernel` (dense path) and from
# `assemble_half_swg_correction` (AIM path) — both preserve the same
# formula. The result is zero when neither m nor n is a boundary DOF.
function _half_swg_surface_kernel(basis::AbstractDivBasis, m::Int, n::Int,
                                    k0::ComplexF64,
                                    outer_rule::TetQuadRule,
                                    duffy_rule::DuffyQuadRule,
                                    tri_rule::TriQuadRule,
                                    tri_duffy_rule::TriDuffyRule)
    tets_m = support_tets(basis, m)
    tets_n = support_tets(basis, n)
    s = zero(ComplexF64)

    if _is_boundary_dof(basis, n)
        @inbounds for tm in tets_m
            tm == 0 && continue
            s += _bulk_surface_pair(basis, m, n, tm, k0,
                                     outer_rule, tri_rule, tri_duffy_rule)
        end
    end
    if _is_boundary_dof(basis, m)
        @inbounds for tn in tets_n
            tn == 0 && continue
            s += _surface_bulk_pair(basis, m, n, tn, k0,
                                     outer_rule, duffy_rule,
                                     tri_rule, tri_duffy_rule)
        end
    end
    if _is_boundary_dof(basis, m) && _is_boundary_dof(basis, n)
        s += _surface_surface_pair(basis, m, n, k0, tri_rule, tri_duffy_rule)
    end
    return s
end

# True for SWG boundary DOFs (half-SWG); false for any other basis.
@inline _is_boundary_dof(basis::SWGBasis, n::Int) = basis.is_boundary[n]
@inline _is_boundary_dof(::AbstractDivBasis, ::Int) = false

# ---------------------------------------------------------------------------
# Bulk-surface pair contribution  K^B_{mn}  (n is a boundary half-SWG DOF)
#
#     K^B_{mn} = + ∫_{T_m^±} (∇·f_m)(r) × ∫_{S_n} G(|r - r'|) dS' dV
#
# The outer integral is a standard tet Gauss over T_m^±; (∇·f_m) is
# constant inside the tet (equal to ±a_m / V_m^±). The inner surface
# integral I_{S_n}(r) is computed by `surface_integral_G`, which picks
# the appropriate singularity handling (:self when m_tet owns S_n,
# :near when m_tet shares a vertex with S_n, :far otherwise).
# ---------------------------------------------------------------------------
function _bulk_surface_pair(basis::AbstractDivBasis, m::Int, n::Int,
                             m_tet::Int, k0::ComplexF64,
                             outer_rule::TetQuadRule,
                             tri_rule::TriQuadRule,
                             tri_duffy_rule::TriDuffyRule)
    verts_m = _tet_vertices(basis.mesh, m_tet)
    Vm = tet_volume(verts_m...)
    s = zero(ComplexF64)
    @inbounds for i in 1:outer_rule.n
        r = bary_to_point(outer_rule.bary[i], verts_m)
        outer_w = outer_rule.weights[i] * Vm
        div_m = divergence(basis, m, r, m_tet)
        div_m == 0 && continue
        I_S = surface_integral_G(basis, n, r, k0;
                                  mode = :auto, m_tet = m_tet,
                                  tri_rule = tri_rule,
                                  tri_duffy_rule = tri_duffy_rule)
        s += outer_w * div_m * I_S
    end
    return s
end

# ---------------------------------------------------------------------------
# Surface-bulk pair contribution  K^C_{mn}  (m is a boundary half-SWG DOF)
#
#     K^C_{mn} = + ∫_{S_m} (f_m·n̂)(r) × [∫_{T_n^±} (∇'·f_n)(r') G(|r-r'|) dV'] dS
#              = + ∫_{S_m} [∫_{T_n^±} (∇·f_n)(const) G dV'] dS
#
# where (f_m·n̂) = 1 on S_m (half-SWG identity) and (∇·f_n) is constant
# inside each support tet of n. The outer integral runs over the boundary
# face S_m; the inner is the volume integral of G over n_tet.
#
# Singularity cases for the inner volume integral at outer point r ∈ S_m:
#   :self — n_tet == m_tet (n shares the owning tet of S_m): r lies on a
#           face of n_tet. Use `duffy_quadrature_around` at r.
#   :near — n_tet shares at least one vertex with m_tet but is not the
#           owning tet: Duffy at each shared vertex (averaged).
#   :far  — no shared vertices: plain tet Gauss.
# ---------------------------------------------------------------------------
function _surface_bulk_pair(basis::AbstractDivBasis, m::Int, n::Int,
                             n_tet::Int, k0::ComplexF64,
                             outer_rule::TetQuadRule,
                             duffy_rule::DuffyQuadRule,
                             tri_rule::TriQuadRule,
                             tri_duffy_rule::TriDuffyRule)
    m_tet = basis.tet_plus[m]          # owning tet of the boundary face S_m
    verts_face = boundary_face_vertices(basis, m)
    a_m = triangle_area(verts_face...)

    verts_n = _tet_vertices(basis.mesh, n_tet)
    centroid_n = 0.25 * (verts_n[1] + verts_n[2] + verts_n[3] + verts_n[4])
    div_n_const = divergence(basis, n, centroid_n, n_tet)
    div_n_const == 0 && return zero(ComplexF64)

    mode = if n_tet == m_tet
        :self
    elseif _tets_share_nodes(basis.mesh, m_tet, n_tet)
        :near
    else
        :far
    end

    s = zero(ComplexF64)
    @inbounds for i in 1:tri_rule.n
        r = tri_bary_to_point(tri_rule.bary[i], verts_face)
        w_out = tri_rule.weights[i] * a_m
        I_V = _volume_integral_G(basis, n_tet, r, k0;
                                  mode = mode, m_tet = m_tet,
                                  outer_rule = outer_rule,
                                  duffy_rule = duffy_rule)
        s += w_out * div_n_const * I_V
    end
    return s
end

# ---------------------------------------------------------------------------
# Volume integral of the scalar Green's function over a single tet
#     I_V(r; T) = ∫_T G(|r - r'|) dV'
# with mode-dependent singularity handling. Used by `_surface_bulk_pair`.
# ---------------------------------------------------------------------------
function _volume_integral_G(basis::AbstractDivBasis, n_tet::Int, r::Vec3,
                             k0::ComplexF64;
                             mode::Symbol,
                             m_tet::Int,
                             outer_rule::TetQuadRule,
                             duffy_rule::DuffyQuadRule)
    verts = _tet_vertices(basis.mesh, n_tet)
    V = tet_volume(verts...)

    if mode === :self
        pts, wts = duffy_quadrature_around(verts, r, duffy_rule)
        I = zero(ComplexF64)
        @inbounds for j in eachindex(pts)
            R = norm(r - pts[j])
            R == 0 && continue
            I += wts[j] * helmholtz_green(R, k0)
        end
        return I
    elseif mode === :near
        shared = _shared_local_vertices(basis.mesh, m_tet, n_tet)
        if isempty(shared)
            return _volume_integral_G(basis, n_tet, r, k0;
                                       mode = :far, m_tet = 0,
                                       outer_rule = outer_rule,
                                       duffy_rule = duffy_rule)
        end
        I = zero(ComplexF64)
        for sv in shared
            pts, wts = duffy_quadrature(verts, sv, duffy_rule)
            acc = zero(ComplexF64)
            @inbounds for j in eachindex(pts)
                R = norm(r - pts[j])
                R == 0 && continue
                acc += wts[j] * helmholtz_green(R, k0)
            end
            I += acc
        end
        return I / length(shared)
    elseif mode === :far
        I = zero(ComplexF64)
        @inbounds for j in 1:outer_rule.n
            rp = bary_to_point(outer_rule.bary[j], verts)
            w = outer_rule.weights[j] * V
            R = norm(r - rp)
            R == 0 && continue
            I += w * helmholtz_green(R, k0)
        end
        return I
    else
        throw(ArgumentError("_volume_integral_G: unknown mode $mode"))
    end
end

# ---------------------------------------------------------------------------
# Surface-surface pair contribution  K^D_{mn}  (BOTH m and n boundary)
#
#     K^D_{mn} = - ∮_{S_m} ∮_{S_n} (f_m·n̂)(f_n·n̂') G(|r-r'|) dS' dS
#              = - ∮_{S_m} ∮_{S_n} G dS' dS         (since f·n̂ = 1)
#
# Singularity classification (by counting nodes shared between face m
# and face n):
#
#   :self     m == n              outer Gauss × inner Duffy-around-r
#   :edge     2 shared nodes       inner Duffy averaged over the 2 shared
#                                  vertices (singular vertex of S_n)
#   :vertex   1 shared node        inner Duffy at the single shared vertex
#   :far      0 shared nodes       outer Gauss × inner Gauss
#
# Note the leading minus sign from §9.4.
# ---------------------------------------------------------------------------
function _surface_surface_pair(basis::AbstractDivBasis, m::Int, n::Int,
                                k0::ComplexF64,
                                tri_rule::TriQuadRule,
                                tri_duffy_rule::TriDuffyRule)
    verts_m = boundary_face_vertices(basis, m)
    verts_n = boundary_face_vertices(basis, n)
    a_m = triangle_area(verts_m...)
    a_n = triangle_area(verts_n...)

    if m == n
        return -_ss_self(verts_m, a_m, k0, tri_rule, tri_duffy_rule)
    end

    # Find local indices in S_n that are shared with S_m
    shared_in_n = _shared_face_local_nodes(basis, m, n)

    if isempty(shared_in_n)
        return -_ss_far(verts_m, verts_n, a_m, a_n, k0, tri_rule)
    else
        return -_ss_near(verts_m, verts_n, a_m, shared_in_n,
                          k0, tri_rule, tri_duffy_rule)
    end
end

# Self: outer Gauss point r ∈ S_m, inner Duffy-around-r over S_m.
function _ss_self(verts::NTuple{3,Vec3}, a::Float64, k0::ComplexF64,
                   tri_rule::TriQuadRule, tri_duffy_rule::TriDuffyRule)
    s = zero(ComplexF64)
    @inbounds for i in 1:tri_rule.n
        r = tri_bary_to_point(tri_rule.bary[i], verts)
        w_out = tri_rule.weights[i] * a
        pts, wts = tri_duffy_quadrature_around(verts, r, tri_duffy_rule)
        I = zero(ComplexF64)
        for j in eachindex(pts)
            R = norm(r - pts[j])
            R == 0 && continue
            I += wts[j] * helmholtz_green(R, k0)
        end
        s += w_out * I
    end
    return s
end

# Near (vertex/edge shared): outer Gauss × inner Duffy at shared vertices.
function _ss_near(verts_m::NTuple{3,Vec3}, verts_n::NTuple{3,Vec3},
                   a_m::Float64, shared_in_n::Vector{Int},
                   k0::ComplexF64,
                   tri_rule::TriQuadRule, tri_duffy_rule::TriDuffyRule)
    s = zero(ComplexF64)
    nshared = length(shared_in_n)
    @inbounds for i in 1:tri_rule.n
        r = tri_bary_to_point(tri_rule.bary[i], verts_m)
        w_out = tri_rule.weights[i] * a_m
        I_avg = zero(ComplexF64)
        for sv in shared_in_n
            pts, wts = tri_duffy_quadrature(verts_n, sv, tri_duffy_rule)
            acc = zero(ComplexF64)
            for j in eachindex(pts)
                R = norm(r - pts[j])
                R == 0 && continue
                acc += wts[j] * helmholtz_green(R, k0)
            end
            I_avg += acc
        end
        I_avg /= nshared
        s += w_out * I_avg
    end
    return s
end

# Far: regular Gauss × Gauss on the two faces.
function _ss_far(verts_m::NTuple{3,Vec3}, verts_n::NTuple{3,Vec3},
                  a_m::Float64, a_n::Float64, k0::ComplexF64,
                  tri_rule::TriQuadRule)
    s = zero(ComplexF64)
    @inbounds for i in 1:tri_rule.n
        r = tri_bary_to_point(tri_rule.bary[i], verts_m)
        w_out = tri_rule.weights[i] * a_m
        I = zero(ComplexF64)
        for j in 1:tri_rule.n
            rp = tri_bary_to_point(tri_rule.bary[j], verts_n)
            w_in = tri_rule.weights[j] * a_n
            R = norm(r - rp)
            R == 0 && continue
            I += w_in * helmholtz_green(R, k0)
        end
        s += w_out * I
    end
    return s
end

"""
    _shared_face_local_nodes(basis, m, n) -> Vector{Int}

Return local indices (1..3) in `face_nodes[n]` that are shared with face `m`.
"""
function _shared_face_local_nodes(basis::SWGBasis, m::Int, n::Int)
    fm = basis.face_nodes[m]
    fn = basis.face_nodes[n]
    shared = Int[]
    @inbounds for b in 1:3
        for a in fm
            if fn[b] == a
                push!(shared, b)
                break
            end
        end
    end
    return shared
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

function _radiation_pair(basis::AbstractDivBasis, m::Int, n::Int,
                         m_tet::Int, n_tet::Int, k0::ComplexF64,
                         outer_rule::TetQuadRule, duffy_rule::DuffyQuadRule)
    verts_m = _tet_vertices(basis.mesh, m_tet)
    verts_n = _tet_vertices(basis.mesh, n_tet)
    Vm = tet_volume(verts_m...)
    Vn = tet_volume(verts_n...)
    k0_sq = k0 * k0
    self_pair = (m_tet == n_tet)
    near_cross = !self_pair && _tets_share_nodes(basis.mesh, m_tet, n_tet)

    s = zero(ComplexF64)
    for i in 1:outer_rule.n
        r = bary_to_point(outer_rule.bary[i], verts_m)
        outer_w = outer_rule.weights[i] * Vm
        f_m_r = evaluate(basis, m, r, m_tet)
        div_m = divergence(basis, m, r, m_tet)

        if self_pair
            # Observation point r is inside n_tet: true 1/R singularity.
            # Use subdivision + Duffy around r to cancel the singularity.
            inner_pts, inner_wts = duffy_quadrature_around(verts_n, r, duffy_rule)
            inner = zero(ComplexF64)
            @inbounds for j in eachindex(inner_pts)
                rp = inner_pts[j]
                R = norm(r - rp)
                R == 0 && continue
                G = helmholtz_green(R, k0)
                f_n_rp = evaluate(basis, n, rp, n_tet)
                div_n = divergence(basis, n, rp, n_tet)
                kernel = k0_sq * dot(f_m_r, f_n_rp) - div_m * div_n
                inner += inner_wts[j] * kernel * G
            end
        elseif near_cross
            # Adjacent tet: r is outside n_tet but close to shared feature.
            # Singularity subtraction: split G = G₀ + (G - G₀).
            # - Static part G₀ = 1/(4πR): nearly singular, use Duffy at
            #   each shared vertex and average for robust coverage.
            # - Smooth part (G - G₀): use standard Gauss (G-G₀ is smooth
            #   everywhere, including R→0 where it → ik₀/(4π)).
            shared = _shared_local_vertices(basis.mesh, m_tet, n_tet)

            # Static part: Duffy at each shared vertex, average results
            inner_static = zero(ComplexF64)
            n_shared = length(shared)
            for sv in shared
                sv_pts, sv_wts = duffy_quadrature(verts_n, sv, duffy_rule)
                term = zero(ComplexF64)
                @inbounds for j in eachindex(sv_pts)
                    rp = sv_pts[j]
                    R = norm(r - rp)
                    R == 0 && continue
                    G0 = helmholtz_green_static(R)
                    f_n_rp = evaluate(basis, n, rp, n_tet)
                    div_n = divergence(basis, n, rp, n_tet)
                    kernel = k0_sq * dot(f_m_r, f_n_rp) - div_m * div_n
                    term += sv_wts[j] * kernel * G0
                end
                inner_static += term
            end
            inner_static /= n_shared

            # Smooth part: standard Gauss (G-G₀ is bounded everywhere)
            gauss_pts, gauss_wts = _gauss_pts_wts(outer_rule, verts_n, Vn)
            inner_smooth = zero(ComplexF64)
            @inbounds for j in eachindex(gauss_pts)
                rp = gauss_pts[j]
                R = norm(r - rp)
                if R == 0
                    # lim_{R→0} [G(R)-G₀(R)] = -ik₀/(4π)  (L'Hôpital)
                    dG = -im * k0 * _INV_FOUR_PI
                else
                    dG = helmholtz_green(R, k0) - helmholtz_green_static(R)
                end
                f_n_rp = evaluate(basis, n, rp, n_tet)
                div_n = divergence(basis, n, rp, n_tet)
                kernel = k0_sq * dot(f_m_r, f_n_rp) - div_m * div_n
                inner_smooth += gauss_wts[j] * kernel * dG
            end

            inner = inner_static + inner_smooth
        else
            # Far pair: G(R) is smooth, use standard Gauss.
            inner_pts, inner_wts = _gauss_pts_wts(outer_rule, verts_n, Vn)
            inner = zero(ComplexF64)
            @inbounds for j in eachindex(inner_pts)
                rp = inner_pts[j]
                R = norm(r - rp)
                R == 0 && continue
                G = helmholtz_green(R, k0)
                f_n_rp = evaluate(basis, n, rp, n_tet)
                div_n = divergence(basis, n, rp, n_tet)
                kernel = k0_sq * dot(f_m_r, f_n_rp) - div_m * div_n
                inner += inner_wts[j] * kernel * G
            end
        end
        s += outer_w * inner
    end
    return s
end

# Anisotropic radiation pair: computes
#   ∫_{T_m}∫_{T_n} [k0² Σ_α κ_α (f_m)_α(f_n)_α − κ_avg (∇·f_m)(∇'·f_n)] G dV' dV
@inline function _aniso_kernel(k0_sq::ComplexF64,
                               kappa_v::SVector{3,ComplexF64},
                               kappa_avg::ComplexF64,
                               f_m::Vec3, f_n::Vec3,
                               div_m::Float64, div_n::Float64)
    k0_sq * (kappa_v[1] * f_m[1] * f_n[1] +
             kappa_v[2] * f_m[2] * f_n[2] +
             kappa_v[3] * f_m[3] * f_n[3]) - kappa_avg * div_m * div_n
end

function _radiation_pair_weighted(basis::AbstractDivBasis, m::Int, n::Int,
                                   m_tet::Int, n_tet::Int, k0::ComplexF64,
                                   kappa_v::SVector{3,ComplexF64},
                                   kappa_avg::ComplexF64,
                                   outer_rule::TetQuadRule,
                                   duffy_rule::DuffyQuadRule)
    verts_m = _tet_vertices(basis.mesh, m_tet)
    verts_n = _tet_vertices(basis.mesh, n_tet)
    Vm = tet_volume(verts_m...)
    Vn = tet_volume(verts_n...)
    k0_sq = k0 * k0
    self_pair = (m_tet == n_tet)
    near_cross = !self_pair && _tets_share_nodes(basis.mesh, m_tet, n_tet)

    s = zero(ComplexF64)
    for i in 1:outer_rule.n
        r = bary_to_point(outer_rule.bary[i], verts_m)
        outer_w = outer_rule.weights[i] * Vm
        f_m_r = evaluate(basis, m, r, m_tet)
        div_m = divergence(basis, m, r, m_tet)

        if self_pair
            inner_pts, inner_wts = duffy_quadrature_around(verts_n, r, duffy_rule)
            inner = zero(ComplexF64)
            @inbounds for j in eachindex(inner_pts)
                rp = inner_pts[j]
                R = norm(r - rp)
                R == 0 && continue
                G = helmholtz_green(R, k0)
                f_n_rp = evaluate(basis, n, rp, n_tet)
                div_n = divergence(basis, n, rp, n_tet)
                inner += inner_wts[j] * _aniso_kernel(k0_sq, kappa_v, kappa_avg,
                                                       f_m_r, f_n_rp, div_m, div_n) * G
            end
        elseif near_cross
            shared = _shared_local_vertices(basis.mesh, m_tet, n_tet)
            inner_static = zero(ComplexF64)
            n_shared = length(shared)
            for sv in shared
                sv_pts, sv_wts = duffy_quadrature(verts_n, sv, duffy_rule)
                term = zero(ComplexF64)
                @inbounds for j in eachindex(sv_pts)
                    rp = sv_pts[j]
                    R = norm(r - rp)
                    R == 0 && continue
                    G0 = helmholtz_green_static(R)
                    f_n_rp = evaluate(basis, n, rp, n_tet)
                    div_n = divergence(basis, n, rp, n_tet)
                    term += sv_wts[j] * _aniso_kernel(k0_sq, kappa_v, kappa_avg,
                                                       f_m_r, f_n_rp, div_m, div_n) * G0
                end
                inner_static += term
            end
            inner_static /= n_shared

            gauss_pts, gauss_wts = _gauss_pts_wts(outer_rule, verts_n, Vn)
            inner_smooth = zero(ComplexF64)
            @inbounds for j in eachindex(gauss_pts)
                rp = gauss_pts[j]
                R = norm(r - rp)
                if R == 0
                    dG = -im * k0 * _INV_FOUR_PI
                else
                    dG = helmholtz_green(R, k0) - helmholtz_green_static(R)
                end
                f_n_rp = evaluate(basis, n, rp, n_tet)
                div_n = divergence(basis, n, rp, n_tet)
                inner_smooth += gauss_wts[j] * _aniso_kernel(k0_sq, kappa_v, kappa_avg,
                                                              f_m_r, f_n_rp, div_m, div_n) * dG
            end
            inner = inner_static + inner_smooth
        else
            inner_pts, inner_wts = _gauss_pts_wts(outer_rule, verts_n, Vn)
            inner = zero(ComplexF64)
            @inbounds for j in eachindex(inner_pts)
                rp = inner_pts[j]
                R = norm(r - rp)
                R == 0 && continue
                G = helmholtz_green(R, k0)
                f_n_rp = evaluate(basis, n, rp, n_tet)
                div_n = divergence(basis, n, rp, n_tet)
                inner += inner_wts[j] * _aniso_kernel(k0_sq, kappa_v, kappa_avg,
                                                       f_m_r, f_n_rp, div_m, div_n) * G
            end
        end
        s += outer_w * inner
    end
    return s
end

"""
    _shared_local_vertices(mesh, t1, t2) -> Vector{Int}

Return the local vertex indices (1..4) in tet `t2` that are shared with tet `t1`.
"""
function _shared_local_vertices(mesh::TetMesh, t1::Int, t2::Int)
    n1 = mesh.tets[t1]
    n2 = mesh.tets[t2]
    shared = Int[]
    @inbounds for b in 1:4
        for a in n1
            if n2[b] == a
                push!(shared, b)
                break
            end
        end
    end
    return shared
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
    assemble_impedance_matrix(basis::AbstractDivBasis;
                              k0::Number,
                              eps_p::Number = 1,
                              eps_bg::Number = 1,
                              outer_rule::TetQuadRule = TET_QUAD_5PT,
                              duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                              symmetrize::Bool = false)
        -> Matrix{ComplexF64}

Build the dense `N × N` impedance matrix for `basis` by calling
[`impedance_element`](@ref) on every pair `(m, n)`. The outer loop over
`m` is parallelized with `Threads.@threads`.

Works for any `AbstractDivBasis` (SWGBasis, RT1Basis, etc.).
"""
function assemble_impedance_matrix(basis::AbstractDivBasis;
                                   k0::Number,
                                   eps_p = 1,
                                   eps_bg::Number = 1,
                                   outer_rule::TetQuadRule = TET_QUAD_5PT,
                                   duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                                   tri_rule::TriQuadRule = tri_collapsed_rule(4),
                                   tri_duffy_rule::TriDuffyRule = tri_duffy_reference_rule(6),
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
                                        duffy_rule = duffy_rule,
                                        tri_rule = tri_rule,
                                        tri_duffy_rule = tri_duffy_rule)
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

# ---------------------------------------------------------------------------
# Half-SWG surface correction matrix (Stage 2.7)
#
# Returns the (sparse) N×N matrix
#
#     Z_extra = -(κ/ε_bg) × (K^B + K^C + K^D)
#
# containing only the Stage 2 half-SWG surface terms (K^B, K^C, K^D from
# `.claude/technical_note.md` §9). It is zero on every (m, n) pair where
# neither m nor n is a boundary half-SWG DOF. The base AIM operator covers
# the k0²(f·f') + K^A bulk terms via mass matrix + grid convolution +
# precorrection; adding `Z_extra` as a sparse additive term completes the
# half-SWG-enabled AIM MVP.
#
# Returns a `spzeros` sentinel for interior-only bases, so callers can
# safely add this unconditionally.
# ---------------------------------------------------------------------------
function assemble_half_swg_correction(basis::AbstractDivBasis;
                                        k0::Number,
                                        eps_p = 1,
                                        eps_bg::Number = 1,
                                        outer_rule::TetQuadRule = TET_QUAD_5PT,
                                        duffy_rule::DuffyQuadRule = duffy_reference_rule(5),
                                        tri_rule::TriQuadRule = tri_collapsed_rule(4),
                                        tri_duffy_rule::TriDuffyRule = tri_duffy_reference_rule(6))
    N = n_basis(basis)
    _, eps_bg_c, _, kappa_v, kappa_avg = _aniso_params(eps_p, eps_bg)
    k0_c = ComplexF64(k0)

    if all(iszero, kappa_v) || !(basis isa SWGBasis) || !any(basis.is_boundary)
        return spzeros(ComplexF64, N, N)
    end

    # Use kappa_avg for surface terms (exact for isotropic, O(Δκ/κ) for aniso)
    scale = -(kappa_avg / eps_bg_c)

    # Threaded row-wise assembly: each task writes to its own row buffer
    # only, and the global sparse triple is assembled sequentially from
    # the per-row buffers afterwards. This avoids ConcurrencyViolationErrors
    # that arise from sharing per-thread buffers under task migration.
    #
    # One row at a time: compute row m's non-zeros into a local buffer,
    # then spawn a task per row so the row-buffer appends are thread-safe
    # by construction. Assemble across rows into global (I, J, V) vectors.
    row_buffers = Vector{Tuple{Vector{Int}, Vector{ComplexF64}}}(undef, N)

    Threads.@threads for m in 1:N
        tets_m = support_tets(basis, m)
        m_is_bnd = basis.is_boundary[m]
        Js = Int[]
        Vs = ComplexF64[]
        @inbounds for n in 1:N
            n_is_bnd = basis.is_boundary[n]
            (m_is_bnd || n_is_bnd) || continue
            K_extra = zero(ComplexF64)
            tets_n = support_tets(basis, n)
            if n_is_bnd
                for tm in tets_m
                    tm == 0 && continue
                    K_extra += _bulk_surface_pair(basis, m, n, tm, k0_c,
                                                    outer_rule, tri_rule,
                                                    tri_duffy_rule)
                end
            end
            if m_is_bnd
                for tn in tets_n
                    tn == 0 && continue
                    K_extra += _surface_bulk_pair(basis, m, n, tn, k0_c,
                                                    outer_rule, duffy_rule,
                                                    tri_rule, tri_duffy_rule)
                end
            end
            if m_is_bnd && n_is_bnd
                K_extra += _surface_surface_pair(basis, m, n, k0_c,
                                                   tri_rule, tri_duffy_rule)
            end
            val = scale * K_extra
            if val != 0
                push!(Js, n)
                push!(Vs, val)
            end
        end
        row_buffers[m] = (Js, Vs)
    end

    # Concatenate rows into (I, J, V) triples for sparse().
    total_nnz = sum(length(b[1]) for b in row_buffers)
    I_all = Vector{Int}(undef, total_nnz)
    J_all = Vector{Int}(undef, total_nnz)
    V_all = Vector{ComplexF64}(undef, total_nnz)
    k = 0
    @inbounds for m in 1:N
        Js, Vs = row_buffers[m]
        for idx in eachindex(Js)
            k += 1
            I_all[k] = m
            J_all[k] = Js[idx]
            V_all[k] = Vs[idx]
        end
    end
    return sparse(I_all, J_all, V_all, N, N)
end
