# Tetrahedral mesh data structure and geometric helpers.
# Phase 1 of BlockVIEM.jl.

using StaticArrays
using LinearAlgebra: cross, dot, norm

const Vec3 = SVector{3,Float64}
const TetVerts = SVector{4,Int}

"""
    TetMesh

Geometry-only container for a conforming tetrahedral mesh.

# Fields
- `nodes::Vector{Vec3}`           — node coordinates (length `N_nodes`)
- `tets::Vector{TetVerts}`        — tetrahedron node indices (length `N_tets`,
                                     1-based, oriented so `tet_volumes` is positive)
- `tet_volumes::Vector{Float64}`  — signed-positive tetrahedron volumes
- `tet_centroids::Vector{Vec3}`   — tetrahedron centroids
- `tet_phys_tags::Vector{Int}`    — physical group tag of each tet (`0` if none)
"""
struct TetMesh
    nodes::Vector{Vec3}
    tets::Vector{TetVerts}
    tet_volumes::Vector{Float64}
    tet_centroids::Vector{Vec3}
    tet_phys_tags::Vector{Int}
end

"""
    tet_volume(p1, p2, p3, p4) -> Float64

Unsigned volume of a tetrahedron with vertices `p1..p4`.
"""
@inline function tet_volume(p1::Vec3, p2::Vec3, p3::Vec3, p4::Vec3)
    return abs(dot(p2 - p1, cross(p3 - p1, p4 - p1))) / 6
end

"""
    tet_signed_volume(p1, p2, p3, p4) -> Float64

Signed volume; positive iff `(p2-p1, p3-p1, p4-p1)` form a right-handed triple.
"""
@inline function tet_signed_volume(p1::Vec3, p2::Vec3, p3::Vec3, p4::Vec3)
    return dot(p2 - p1, cross(p3 - p1, p4 - p1)) / 6
end

"""
    tet_centroid(p1, p2, p3, p4) -> Vec3
"""
@inline function tet_centroid(p1::Vec3, p2::Vec3, p3::Vec3, p4::Vec3)
    return (p1 + p2 + p3 + p4) / 4
end

"""
    triangle_area(p1, p2, p3) -> Float64
"""
@inline function triangle_area(p1::Vec3, p2::Vec3, p3::Vec3)
    return norm(cross(p2 - p1, p3 - p1)) / 2
end

"""
    TetMesh(nodes, tets; phys_tags=zeros(Int, length(tets)))

Construct a [`TetMesh`](@ref) from raw node coordinates and tetrahedron
connectivity. The orientation of each tet is automatically corrected so that
the stored signed volume is strictly positive (degenerate tets raise an error).
"""
function TetMesh(nodes::AbstractVector{Vec3},
                 tets::AbstractVector{TetVerts};
                 phys_tags::AbstractVector{<:Integer} = zeros(Int, length(tets)))
    @assert length(phys_tags) == length(tets) "phys_tags must match tets length"
    nodes_v = collect(Vec3, nodes)
    tets_v = collect(TetVerts, tets)
    N = length(tets_v)
    volumes = Vector{Float64}(undef, N)
    centroids = Vector{Vec3}(undef, N)
    for i in 1:N
        a, b, c, d = tets_v[i]
        p1, p2, p3, p4 = nodes_v[a], nodes_v[b], nodes_v[c], nodes_v[d]
        sv = tet_signed_volume(p1, p2, p3, p4)
        if sv == 0
            error("Degenerate tetrahedron at index $i")
        elseif sv < 0
            # flip orientation by swapping the last two vertices
            tets_v[i] = TetVerts(a, b, d, c)
            sv = -sv
        end
        volumes[i] = sv
        centroids[i] = tet_centroid(p1, p2, p3, p4)
    end
    return TetMesh(nodes_v, tets_v, volumes, centroids, collect(Int, phys_tags))
end

"""
    n_nodes(mesh::TetMesh) -> Int
"""
@inline n_nodes(mesh::TetMesh) = length(mesh.nodes)

"""
    n_tets(mesh::TetMesh) -> Int
"""
@inline n_tets(mesh::TetMesh) = length(mesh.tets)

"""
    total_volume(mesh::TetMesh) -> Float64
"""
total_volume(mesh::TetMesh) = sum(mesh.tet_volumes)
