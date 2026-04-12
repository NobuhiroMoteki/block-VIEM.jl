# Surface integrals of the scalar Helmholtz Green's function over a boundary
# face S_n of the half-SWG basis. Needed by the Stage-2 (K^B, K^C, K^D) surface
# correction terms derived in `.claude/technical_note.md` §9.
#
# Central primitive:
#     I_G(r; n) = ∫_{S_n} G(|r - r'|) dS'
#
# where S_n is a planar triangular boundary face and r is a 3D observation
# point. Three regimes are handled:
#
#   :far   — r is far from S_n (e.g., in a non-adjacent tet). Plain triangle
#            Gauss is accurate.
#   :near  — r is in a tet sharing a vertex or edge with S_n but not owning
#            S_n. The 1/R integrand is integrable but near-singular; use
#            vertex-Duffy on S_n with the shared vertex as the singular one
#            (averaged over shared vertices when there is more than one).
#   :self  — r is inside T_n^+ (the tet that owns S_n) and may approach S_n
#            arbitrarily closely. Project r onto the plane of S_n and use
#            `tri_duffy_quadrature_around` with the projected point as the
#            singular vertex.
#
# The caller selects the mode via a keyword; a convenience `:auto` mode picks
# based on the tet indices.

using LinearAlgebra: norm, dot, cross

"""
    boundary_face_vertices(basis::SWGBasis, n::Int) -> NTuple{3,Vec3}

Return the three nodal vertices of boundary face `n` as a tuple. Only valid
for a half-SWG basis function (`basis.is_boundary[n] == true`).
"""
@inline function boundary_face_vertices(basis::SWGBasis, n::Int)
    triple = basis.face_nodes[n]
    nodes = basis.mesh.nodes
    return (nodes[triple[1]], nodes[triple[2]], nodes[triple[3]])
end

"""
    project_point_to_triangle_plane(r::Vec3, verts::NTuple{3,Vec3}) -> Vec3

Orthogonal projection of point `r` onto the plane containing the triangle
`verts`. Used to pick the singular vertex for `tri_duffy_quadrature_around`
when `r` lies in a volume above or below the triangle.
"""
@inline function project_point_to_triangle_plane(r::Vec3, verts::NTuple{3,Vec3})
    v1, v2, v3 = verts
    n_vec = cross(v2 - v1, v3 - v1)
    n2 = dot(n_vec, n_vec)
    n2 == 0 && return r
    t = dot(r - v1, n_vec) / n2
    return r - t * n_vec
end

"""
    surface_integral_G(basis::SWGBasis, n::Int, r::Vec3, k0::Number;
                       mode::Symbol = :auto,
                       m_tet::Int = 0,
                       tri_rule::TriQuadRule = tri_collapsed_rule(4),
                       tri_duffy_rule::TriDuffyRule = tri_duffy_reference_rule(6))
        -> ComplexF64

Evaluate `∫_{S_n} G(|r - r'|) dS'` for the boundary face `n` of a half-SWG
`basis`, using the quadrature strategy indicated by `mode`:

- `:far`   — plain triangle Gauss with `tri_rule`.
- `:near`  — vertex-Duffy on the singular vertex shared with `m_tet`; falls
             back to `:far` when `m_tet` does not share a vertex with face `n`.
             When multiple shared vertices exist, the results from each are
             averaged (consistent with the existing near-singular treatment
             in `_radiation_pair`).
- `:self`  — project `r` onto the plane of face `n` and run
             `tri_duffy_quadrature_around` at that projection.
- `:auto`  — choose `:self` if `m_tet == basis.tet_plus[n]`, `:near` if
             `m_tet` shares a vertex with face `n`, otherwise `:far`.

Returns a `ComplexF64`.
"""
function surface_integral_G(basis::SWGBasis, n::Int, r::Vec3, k0::Number;
                            mode::Symbol = :auto,
                            m_tet::Int = 0,
                            tri_rule::TriQuadRule = tri_collapsed_rule(4),
                            tri_duffy_rule::TriDuffyRule = tri_duffy_reference_rule(6))
    verts = boundary_face_vertices(basis, n)
    k0c = ComplexF64(k0)

    if mode === :auto
        mode = _choose_surface_mode(basis, n, m_tet)
    end

    if mode === :far
        return _surface_G_far(verts, r, k0c, tri_rule)
    elseif mode === :near
        return _surface_G_near(basis, n, verts, r, k0c, m_tet, tri_duffy_rule)
    elseif mode === :self
        return _surface_G_self(verts, r, k0c, tri_duffy_rule)
    else
        throw(ArgumentError("unknown mode $mode — use :far, :near, :self, or :auto"))
    end
end

@inline function _choose_surface_mode(basis::SWGBasis, n::Int, m_tet::Int)
    m_tet == 0 && return :far
    if m_tet == basis.tet_plus[n]
        return :self
    end
    # Check for a shared node between m_tet and face n.
    mesh = basis.mesh
    face_triple = basis.face_nodes[n]
    tet_nodes = mesh.tets[m_tet]
    @inbounds for a in tet_nodes, b in face_triple
        a == b && return :near
    end
    return :far
end

function _surface_G_far(verts::NTuple{3,Vec3}, r::Vec3, k0::ComplexF64,
                        rule::TriQuadRule)
    a = triangle_area(verts...)
    s = zero(ComplexF64)
    @inbounds for i in 1:rule.n
        rp = tri_bary_to_point(rule.bary[i], verts)
        R = norm(r - rp)
        R == 0 && continue
        s += rule.weights[i] * helmholtz_green(R, k0)
    end
    return a * s
end

function _surface_G_near(basis::SWGBasis, n::Int, verts::NTuple{3,Vec3},
                         r::Vec3, k0::ComplexF64, m_tet::Int,
                         rule::TriDuffyRule)
    # Locate the vertices of face n that are shared with m_tet.
    face_triple = basis.face_nodes[n]
    tet_nodes = basis.mesh.tets[m_tet]
    shared_locals = Int[]
    @inbounds for (lf, fn) in enumerate(face_triple)
        for tn in tet_nodes
            if tn == fn
                push!(shared_locals, lf)
                break
            end
        end
    end
    if isempty(shared_locals)
        return _surface_G_far(verts, r, k0,
                              tri_collapsed_rule(rule.n_per_axis))
    end
    accum = zero(ComplexF64)
    for sv_local in shared_locals
        pts, wts = tri_duffy_quadrature(verts, sv_local, rule)
        term = zero(ComplexF64)
        @inbounds for i in eachindex(pts)
            R = norm(r - pts[i])
            R == 0 && continue
            term += wts[i] * helmholtz_green(R, k0)
        end
        accum += term
    end
    return accum / length(shared_locals)
end

function _surface_G_self(verts::NTuple{3,Vec3}, r::Vec3, k0::ComplexF64,
                         rule::TriDuffyRule)
    p_plane = project_point_to_triangle_plane(r, verts)
    pts, wts = tri_duffy_quadrature_around(verts, p_plane, rule)
    isempty(pts) && return zero(ComplexF64)
    s = zero(ComplexF64)
    @inbounds for i in eachindex(pts)
        R = norm(r - pts[i])
        R == 0 && continue
        s += wts[i] * helmholtz_green(R, k0)
    end
    return s
end
