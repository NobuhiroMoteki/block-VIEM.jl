# SWG (Schaubert-Wilton-Glisson) basis on a tetrahedral mesh.
# Reference: Schaubert, Wilton, Glisson (1984), Eqs. (10), (12).
# See also `.claude/technical_note.md` §1.

# Local-face convention for a tet `(v1, v2, v3, v4)`:
#   local face k is the face *opposite* vertex k (k = 1..4).
# Hence the "free vertex" of local face k is v_k itself.
const TET_LOCAL_FACES = (
    (2, 3, 4),  # opposite v1
    (1, 3, 4),  # opposite v2
    (1, 2, 4),  # opposite v3
    (1, 2, 3),  # opposite v4
)

"""
    SWGBasis

SWG basis information attached to a [`TetMesh`](@ref). One basis function is
defined per *internal* face (a face shared by exactly two tetrahedra).

Per basis index `n` we store the geometric quantities required by
`f_n(r) = ±(a_n / (3 V_n^±)) (r - p_n^±)` and its divergence
`∇·f_n = ±a_n / V_n^±` (Eqs. (10),(12) of SWG 1984).

# Fields
- `mesh::TetMesh`                       — underlying mesh
- `face_nodes::Vector{SVector{3,Int}}`  — sorted node triple of the shared face
- `face_areas::Vector{Float64}`         — face area `a_n`
- `tet_plus::Vector{Int}`               — index of `T_n^+`
- `tet_minus::Vector{Int}`              — index of `T_n^-`
- `free_vertex_plus::Vector{Int}`       — node index `p_n^+` (free vertex of `T_n^+`)
- `free_vertex_minus::Vector{Int}`      — node index `p_n^-` (free vertex of `T_n^-`)
"""
struct SWGBasis
    mesh::TetMesh
    face_nodes::Vector{SVector{3,Int}}
    face_areas::Vector{Float64}
    tet_plus::Vector{Int}
    tet_minus::Vector{Int}
    free_vertex_plus::Vector{Int}
    free_vertex_minus::Vector{Int}
end

"""
    n_basis(basis::SWGBasis) -> Int
"""
@inline n_basis(basis::SWGBasis) = length(basis.face_areas)

"""
    build_swg_basis(mesh::TetMesh) -> SWGBasis

Construct the SWG basis on `mesh` by enumerating all faces and pairing those
that are shared by two tetrahedra. Boundary faces (only one neighbor) are
discarded since the corresponding SWG basis would lack normal continuity.

The `+` / `-` assignment for each internal face is deterministic: the tet with
the *smaller* global index becomes `T^+`. This guarantees a reproducible basis
ordering across runs and platforms.
"""
function build_swg_basis(mesh::TetMesh)
    # face_key (sorted node triple) -> (tet_idx, local_face_idx)
    face_table = Dict{NTuple{3,Int},Vector{Tuple{Int,Int}}}()
    for (ti, tet) in enumerate(mesh.tets)
        for (lf, idxs) in enumerate(TET_LOCAL_FACES)
            tri = (tet[idxs[1]], tet[idxs[2]], tet[idxs[3]])
            key = _sorted_triple(tri)
            push!(get!(() -> Tuple{Int,Int}[], face_table, key), (ti, lf))
        end
    end

    face_nodes = SVector{3,Int}[]
    face_areas = Float64[]
    tet_plus = Int[]
    tet_minus = Int[]
    free_plus = Int[]
    free_minus = Int[]

    for (key, owners) in face_table
        if length(owners) == 1
            continue                    # boundary face
        elseif length(owners) > 2
            error("Non-manifold face $(key) shared by $(length(owners)) tets")
        end
        (ta, la), (tb, lb) = owners
        # deterministic +/- assignment: smaller tet index is "+"
        if ta < tb
            tp, lp, tm, lm = ta, la, tb, lb
        else
            tp, lp, tm, lm = tb, lb, ta, la
        end
        # free vertex of T± is the local vertex with index lp / lm
        fp = mesh.tets[tp][lp]
        fm = mesh.tets[tm][lm]
        # face area from any of the three triangle nodes
        n1, n2, n3 = key
        a = triangle_area(mesh.nodes[n1], mesh.nodes[n2], mesh.nodes[n3])
        push!(face_nodes, SVector{3,Int}(n1, n2, n3))
        push!(face_areas, a)
        push!(tet_plus, tp)
        push!(tet_minus, tm)
        push!(free_plus, fp)
        push!(free_minus, fm)
    end

    return SWGBasis(mesh, face_nodes, face_areas, tet_plus, tet_minus,
                    free_plus, free_minus)
end

@inline function _sorted_triple(t::NTuple{3,Int})
    a, b, c = t
    if a > b; a, b = b, a; end
    if b > c; b, c = c, b; end
    if a > b; a, b = b, a; end
    return (a, b, c)
end

"""
    evaluate(basis::SWGBasis, n::Int, r::Vec3, tet::Int) -> Vec3

Value of the `n`-th SWG basis function at point `r`, assumed to lie in
tetrahedron index `tet`. Returns a zero vector if `tet` is neither
`T_n^+` nor `T_n^-`.

Implements `f_n = +(a_n / (3 V^+))(r - p^+)` in `T^+` and
`f_n = -(a_n / (3 V^-))(r - p^-)` in `T^-` (SWG 1984 Eq. (10)).
"""
@inline function evaluate(basis::SWGBasis, n::Int, r::Vec3, tet::Int)
    a = basis.face_areas[n]
    if tet == basis.tet_plus[n]
        V = basis.mesh.tet_volumes[tet]
        p = basis.mesh.nodes[basis.free_vertex_plus[n]]
        return (a / (3 * V)) * (r - p)
    elseif tet == basis.tet_minus[n]
        V = basis.mesh.tet_volumes[tet]
        p = basis.mesh.nodes[basis.free_vertex_minus[n]]
        return -(a / (3 * V)) * (r - p)
    else
        return zero(Vec3)
    end
end

"""
    divergence(basis::SWGBasis, n::Int, tet::Int) -> Float64

Divergence of the `n`-th SWG basis function in tetrahedron `tet`. Constant
within each support tetrahedron and zero elsewhere (SWG 1984 Eq. (12)):
`∇·f_n = +a_n/V^+` in `T^+`, `∇·f_n = -a_n/V^-` in `T^-`.
"""
@inline function divergence(basis::SWGBasis, n::Int, tet::Int)
    if tet == basis.tet_plus[n]
        return basis.face_areas[n] / basis.mesh.tet_volumes[tet]
    elseif tet == basis.tet_minus[n]
        return -basis.face_areas[n] / basis.mesh.tet_volumes[tet]
    else
        return 0.0
    end
end
