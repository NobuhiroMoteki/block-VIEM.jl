# Shared rendering helpers for the block-VIEM.jl visualisation
# scripts (`visualize_gre.jl`, `visualize_aggregate.jl`).
#
# Provides:
#   • boundary_triangles_oriented(mesh)       — outward-oriented surface triangles
#   • visible_edges(tris, nodes_L, view_dir)  — hidden-edge removal
#       (backface culling + Möller–Trumbore self-occlusion)
#   • camera_direction(azimuth, elevation)    — Makie Axis3 camera unit vector
#   • euler_zyz_matrix(α, β, γ)               — intrinsic ZYZ rotation matrix
#
# Required because CairoMakie has no per-pixel depth test for line
# primitives, so neither `wireframe!` nor a naïve `linesegments!` of
# all boundary edges can render an opaque-solid + mesh-edge view of a
# non-convex closed surface correctly.

using StaticArrays

# ──────────────────────────────────────────────────────────────────────────────
#  Vector helpers (avoid pulling in LinearAlgebra for three multiplies)
# ──────────────────────────────────────────────────────────────────────────────

@inline _cross3(a, b) = (a[2]*b[3] - a[3]*b[2],
                         a[3]*b[1] - a[1]*b[3],
                         a[1]*b[2] - a[2]*b[1])
@inline _dot3(a, b)   = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline _sub3(a, b)   = (a[1]-b[1], a[2]-b[2], a[3]-b[3])

# ──────────────────────────────────────────────────────────────────────────────
#  Boundary-triangle extraction with outward-pointing normals
# ──────────────────────────────────────────────────────────────────────────────

"""
    boundary_triangles_oriented(mesh) -> Vector{NTuple{3,Int}}

Boundary-face triangles (faces shared by exactly one tet) with
vertex order chosen so that `(p2 - p1) × (p3 - p1)` points outward
from the parent tet, i.e. away from the apex vertex.
"""
function boundary_triangles_oriented(mesh)
    count    = Dict{NTuple{3,Int},Int}()
    oriented = Dict{NTuple{3,Int},NTuple{3,Int}}()
    apex     = Dict{NTuple{3,Int},Int}()
    for tet in mesh.tets
        a, b, c, d = tet[1], tet[2], tet[3], tet[4]
        for (face, opp) in (((a, b, c), d), ((a, b, d), c),
                            ((a, c, d), b), ((b, c, d), a))
            key = Tuple(sort!([face...]))::NTuple{3,Int}
            count[key] = get(count, key, 0) + 1
            oriented[key] = face
            apex[key] = opp
        end
    end
    tris = NTuple{3,Int}[]
    sizehint!(tris, length(count) ÷ 8)
    for (k, v) in count
        v == 1 || continue
        face = oriented[k]
        opp  = apex[k]
        p1, p2, p3 = mesh.nodes[face[1]], mesh.nodes[face[2]], mesh.nodes[face[3]]
        n = _cross3(_sub3(p2, p1), _sub3(p3, p1))
        # outward normal must point AWAY from the apex
        if _dot3(n, _sub3(p1, mesh.nodes[opp])) < 0
            push!(tris, (face[1], face[3], face[2]))
        else
            push!(tris, face)
        end
    end
    return tris
end

# ──────────────────────────────────────────────────────────────────────────────
#  Möller–Trumbore ray–triangle intersection
# ──────────────────────────────────────────────────────────────────────────────

"""
    ray_triangle_t(orig, dir, v0, v1, v2, eps_self=1e-12) -> Float64

Return the parametric distance `t` (so the hit point is
`orig + t·dir`) when the ray hits the interior of the triangle at
`t > eps_self`; return `Inf` otherwise.  `dir` need not be unit.
"""
@inline function ray_triangle_t(orig::NTuple{3,Float64},
                                dir::NTuple{3,Float64},
                                v0::NTuple{3,Float64},
                                v1::NTuple{3,Float64},
                                v2::NTuple{3,Float64},
                                eps_self::Float64=1e-12)
    e1 = _sub3(v1, v0)
    e2 = _sub3(v2, v0)
    h  = _cross3(dir, e2)
    a  = _dot3(e1, h)
    abs(a) < 1e-18 && return Inf
    inv_a = 1.0 / a
    s  = _sub3(orig, v0)
    u  = inv_a * _dot3(s, h)
    (u < 0.0 || u > 1.0) && return Inf
    q  = _cross3(s, e1)
    v  = inv_a * _dot3(dir, q)
    (v < 0.0 || u + v > 1.0) && return Inf
    t  = inv_a * _dot3(e2, q)
    return t > eps_self ? t : Inf
end

# ──────────────────────────────────────────────────────────────────────────────
#  Hidden-edge removal
# ──────────────────────────────────────────────────────────────────────────────

"""
    visible_edges(tris_oriented, nodes_L, view_dir; eps_self=1e-9)
        -> Vector{Tuple{Int,Int}}

Hidden-edge removal for an opaque closed surface mesh rendered with
CairoMakie.  Two stages:

1. **Backface culling.** An edge is a candidate only if at least one
   of its two adjacent boundary triangles satisfies
   `n · view_dir > 0` (front-facing).  Silhouette edges are retained
   because they are adjacent to exactly one front-facing triangle.

2. **Self-occlusion test.** For non-convex bodies (doublets, chains,
   clusters, GREs with strong deformation) a front-facing triangle
   on one part of the surface can be physically blocked by another
   front-facing triangle in front of it.  For each candidate edge,
   cast a ray from the edge midpoint toward the camera and reject
   the edge if it is intersected by any other boundary triangle
   (Möller–Trumbore).  Triangles incident to either endpoint of the
   candidate edge are skipped to avoid false positives at shared
   vertices.

This is O(N_edges × N_triangles); for the meshes used by the figure
gallery (~10³–10⁴ triangles) it runs in well under a second.
"""
function visible_edges(tris_oriented::Vector{NTuple{3,Int}},
                       nodes_L::Vector,
                       view_dir::NTuple{3,Float64};
                       eps_self::Float64=1e-9)
    Nt = length(tris_oriented)
    Nn = length(nodes_L)
    front = falses(Nt)
    edge_tris = Dict{Tuple{Int,Int},Vector{Int}}()
    node_tris = [Int[] for _ in 1:Nn]
    @inbounds for (k, (i1, i2, i3)) in enumerate(tris_oriented)
        p1, p2, p3 = nodes_L[i1], nodes_L[i2], nodes_L[i3]
        n = _cross3(_sub3(p2, p1), _sub3(p3, p1))
        front[k] = _dot3(n, view_dir) > 0
        for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
            key = a < b ? (a, b) : (b, a)
            push!(get!(edge_tris, key, Int[]), k)
        end
        push!(node_tris[i1], k)
        push!(node_tris[i2], k)
        push!(node_tris[i3], k)
    end

    verts = Vector{NTuple{3,NTuple{3,Float64}}}(undef, Nt)
    @inbounds for k in 1:Nt
        i1, i2, i3 = tris_oriented[k]
        p1, p2, p3 = nodes_L[i1], nodes_L[i2], nodes_L[i3]
        verts[k] = ((p1[1], p1[2], p1[3]),
                    (p2[1], p2[2], p2[3]),
                    (p3[1], p3[2], p3[3]))
    end

    visible = Tuple{Int,Int}[]
    sizehint!(visible, length(edge_tris))
    skip_mask = falses(Nt)

    @inbounds for (key, adj) in edge_tris
        any_front = false
        for k in adj;  front[k] && (any_front = true; break);  end
        any_front || continue

        a, b = key
        for k in node_tris[a]; skip_mask[k] = true; end
        for k in node_tris[b]; skip_mask[k] = true; end

        pa = nodes_L[a]; pb = nodes_L[b]
        mid = (0.5*(pa[1]+pb[1]), 0.5*(pa[2]+pb[2]), 0.5*(pa[3]+pb[3]))
        occluded = false
        for k in 1:Nt
            skip_mask[k] && continue
            v0, v1, v2 = verts[k]
            t = ray_triangle_t(mid, view_dir, v0, v1, v2, eps_self)
            if t < Inf
                occluded = true
                break
            end
        end

        for k in node_tris[a]; skip_mask[k] = false; end
        for k in node_tris[b]; skip_mask[k] = false; end

        occluded || push!(visible, key)
    end
    return visible
end

# ──────────────────────────────────────────────────────────────────────────────
#  Camera helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    camera_direction(azimuth, elevation) -> NTuple{3,Float64}

Unit vector from the scene origin to the Axis3 camera, matching
Makie's `azimuth`/`elevation` convention (azimuth is rotation about
+z from the +x axis, elevation is angle above the xy-plane).
"""
function camera_direction(azimuth::Real, elevation::Real)
    cα = cos(azimuth);  sα = sin(azimuth)
    cβ = cos(elevation); sβ = sin(elevation)
    return (cα * cβ, sα * cβ, sβ)
end

"""
    euler_zyz_matrix(α, β, γ) -> SMatrix{3,3}

Intrinsic ZYZ Euler rotation matrix (matches
`scipy.spatial.transform.Rotation.from_euler("ZYZ", ...)` and
block-DDA_Py).  `R = Rz(α) · Ry(β) · Rz(γ)` maps particle-frame
coordinates to lab-frame coordinates.
"""
function euler_zyz_matrix(α::Real, β::Real, γ::Real)
    ca, sa = cos(α), sin(α)
    cb, sb = cos(β), sin(β)
    cg, sg = cos(γ), sin(γ)
    Rz_α = @SMatrix [ca  -sa  0.0
                     sa   ca  0.0
                     0.0  0.0 1.0]
    Ry_β = @SMatrix [cb  0.0  sb
                     0.0 1.0  0.0
                    -sb  0.0  cb]
    Rz_γ = @SMatrix [cg  -sg  0.0
                     sg   cg  0.0
                     0.0  0.0 1.0]
    return Rz_α * Ry_β * Rz_γ
end
