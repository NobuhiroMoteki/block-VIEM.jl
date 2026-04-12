# SWG (Schaubert-Wilton-Glisson) basis on a tetrahedral mesh.
# Reference: Schaubert, Wilton, Glisson (1984), Eqs. (10), (12).
# See also `.claude/technical_note.md` §1.

# ---------------------------------------------------------------------------
# Abstract parent type for divergence-conforming (H(div)) vector bases.
# All subtypes must implement:
#   n_basis(basis) -> Int
#   evaluate(basis, n, r, tet) -> Vec3
#   divergence(basis, n, r, tet) -> Float64
#   support_tets(basis, n) -> Tuple of tet indices
# ---------------------------------------------------------------------------
abstract type AbstractDivBasis end

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

SWG basis information attached to a [`TetMesh`](@ref).

Following Schaubert-Wilton-Glisson (1984), one basis function is defined per
face of the tetrahedral model, including BOTH internal faces (shared by two
tets) and boundary faces (on the global particle surface ∂Ω). For a boundary
face, `tet_minus[n] = 0` and `free_vertex_minus[n] = 0` as sentinels, so
`f_n` has one-sided support on `T_n^+` only (a "half-SWG" basis function).
This lets `D·n̂` be non-zero on the particle boundary, representing the
surface polarization charge.

Per basis index `n` we store the geometric quantities required by
`f_n(r) = ±(a_n / (3 V_n^±)) (r - p_n^±)` and its bulk divergence
`∇·f_n = ±a_n / V_n^±` (Eqs. (10),(12) of SWG 1984).

# Fields
- `mesh::TetMesh`                       — underlying mesh
- `face_nodes::Vector{SVector{3,Int}}`  — sorted node triple of the face
- `face_areas::Vector{Float64}`         — face area `a_n`
- `tet_plus::Vector{Int}`               — index of `T_n^+`
- `tet_minus::Vector{Int}`              — index of `T_n^-` (0 for boundary faces)
- `free_vertex_plus::Vector{Int}`       — node index `p_n^+` (free vertex of `T_n^+`)
- `free_vertex_minus::Vector{Int}`      — node index `p_n^-` (0 for boundary faces)
- `is_boundary::Vector{Bool}`           — true if face n is on ∂Ω (half-SWG)
"""
struct SWGBasis <: AbstractDivBasis
    mesh::TetMesh
    face_nodes::Vector{SVector{3,Int}}
    face_areas::Vector{Float64}
    tet_plus::Vector{Int}
    tet_minus::Vector{Int}
    free_vertex_plus::Vector{Int}
    free_vertex_minus::Vector{Int}
    is_boundary::Vector{Bool}
end

"""
    n_basis(basis::SWGBasis) -> Int
"""
@inline n_basis(basis::SWGBasis) = length(basis.face_areas)

"""
    support_tets(basis::SWGBasis, n::Int) -> Tuple{Int,Int}

Return the pair of tetrahedra supporting the `n`-th SWG basis function.
"""
@inline support_tets(basis::SWGBasis, n::Int) = (basis.tet_plus[n], basis.tet_minus[n])

"""
    build_tet_to_dofs(basis::AbstractDivBasis) -> Dict{Int,Vector{Int}}

Build a mapping from tetrahedron index to the list of global DOF indices
supported in that tet. Useful for mass matrix and AIM assembly.
Sentinel tets (index 0, from interior DOFs) are skipped.
"""
function build_tet_to_dofs(basis::AbstractDivBasis)
    tet_to_dofs = Dict{Int,Vector{Int}}()
    for n in 1:n_basis(basis)
        for tet in support_tets(basis, n)
            tet == 0 && continue
            push!(get!(Vector{Int}, tet_to_dofs, tet), n)
        end
    end
    return tet_to_dofs
end

"""
    build_swg_basis(mesh::TetMesh) -> SWGBasis

Construct the SWG basis on `mesh` by enumerating all faces.

Faces shared by two tets always become interior SWG basis functions. When
`include_boundary_faces = true`, faces with only one tet (on the global
boundary ∂Ω) additionally become half-SWG basis functions with
`tet_minus = 0`, `free_vertex_minus = 0`. The half-SWG functions let `D·n̂`
be non-zero on ∂Ω, as in the full SWG 1984 formulation.

WARNING: enabling `include_boundary_faces` without also applying the
matching surface-correction terms in the weak form (Eq. 28 of SWG 1984,
the `(κ_+ - κ_-) ∫_{∂_n} G ds'` contribution) breaks charge neutrality
and gives catastrophically wrong cross sections. Only turn this flag on
when the impedance-matrix assembly has been updated to include surface
corrections.

The `+` / `-` assignment for each internal face is deterministic: the tet
with the *smaller* global index becomes `T^+`. This guarantees a
reproducible basis ordering across runs and platforms.
"""
function build_swg_basis(mesh::TetMesh; include_boundary_faces::Bool = false)
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
    is_boundary = Bool[]

    for (key, owners) in face_table
        n1, n2, n3 = key
        a = triangle_area(mesh.nodes[n1], mesh.nodes[n2], mesh.nodes[n3])

        if length(owners) == 2
            (ta, la), (tb, lb) = owners
            if ta < tb
                tp, lp, tm, lm = ta, la, tb, lb
            else
                tp, lp, tm, lm = tb, lb, ta, la
            end
            fp = mesh.tets[tp][lp]
            fm = mesh.tets[tm][lm]
            push!(face_nodes, SVector{3,Int}(n1, n2, n3))
            push!(face_areas, a)
            push!(tet_plus, tp)
            push!(tet_minus, tm)
            push!(free_plus, fp)
            push!(free_minus, fm)
            push!(is_boundary, false)
        elseif length(owners) == 1
            include_boundary_faces || continue
            # Half-SWG on boundary face: one-sided support on T^+ only.
            (ta, la), = owners
            fp = mesh.tets[ta][la]
            push!(face_nodes, SVector{3,Int}(n1, n2, n3))
            push!(face_areas, a)
            push!(tet_plus, ta)
            push!(tet_minus, 0)
            push!(free_plus, fp)
            push!(free_minus, 0)
            push!(is_boundary, true)
        else
            error("Non-manifold face $(key) shared by $(length(owners)) tets")
        end
    end

    return SWGBasis(mesh, face_nodes, face_areas, tet_plus, tet_minus,
                    free_plus, free_minus, is_boundary)
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

"""
    divergence(basis::SWGBasis, n::Int, r::Vec3, tet::Int) -> Float64

4-argument form (uniform calling convention with RT1Basis).
For SWG (RT0), the divergence is constant within each tet, so `r` is ignored.
"""
@inline divergence(basis::SWGBasis, n::Int, ::Vec3, tet::Int) = divergence(basis, n, tet)
