# RT1 (Raviart-Thomas order 1, i.e. 2nd-order divergence-conforming) basis
# on a tetrahedral mesh, using the contravariant Piola transform.
#
# Reference: technical_note.md §8-§11, reference_rt1.jl for the 15
# reference-element basis functions verified against symfem.
#
# DOF structure (per tet on the reference element):
#   Face DOFs:     3 per face × 4 faces = 12  (shared between adjacent tets)
#   Interior DOFs: 3                           (local to each tet)
#   Total per tet: 15
#
# Global DOF count:
#   N_dof = 3 × N_internal_faces  +  3 × N_tets
#
# The Piola transform maps reference basis functions to the physical element:
#   φ^phys(r) = (σ / det J) J φ^ref(F^{-1}(r))
#   div φ^phys(r) = (σ / det J) div^ref φ^ref(F^{-1}(r))
# where σ = +1 for T⁺ (smaller tet index) and σ = -1 for T⁻.

using LinearAlgebra: det, inv
using StaticArrays

# ---------------------------------------------------------------------------
# Reference-face vertex mapping.
# REF_FACE_VERTS[j+1] (0-based ref face j) gives the 1-based local vertex
# indices of the tet that lie ON that reference face.
#
# Reference face numbering (see reference_rt1.jl / technical_note.md §9.1):
#   ref face 0 (z=0):      opposite v3 → vertices v0, v1, v2 → local 1,2,3
#   ref face 1 (y=0):      opposite v2 → vertices v0, v1, v3 → local 1,2,4
#   ref face 2 (x=0):      opposite v1 → vertices v0, v2, v3 → local 1,3,4
#   ref face 3 (x+y+z=1):  opposite v0 → vertices v1, v2, v3 → local 2,3,4
# ---------------------------------------------------------------------------
const REF_FACE_VERTS = (
    (1, 2, 3),  # ref face 0
    (1, 2, 4),  # ref face 1
    (1, 3, 4),  # ref face 2
    (2, 3, 4),  # ref face 3
)

# Mapping: physical local face k (1..4, opposite vertex k) → 0-based ref face
# TET_LOCAL_FACES[k] gives the local vertex indices ON local face k.
# Comparing with REF_FACE_VERTS: local face 4 ↔ ref face 0, ..., local face 1 ↔ ref face 3.
const LOCAL_FACE_TO_REF_FACE = (3, 2, 1, 0)  # index by local face k

"""
    RT1Basis <: AbstractDivBasis

Second-order Raviart-Thomas (RT1) basis on a [`TetMesh`](@ref). Each internal
face contributes 3 face DOFs (one per face vertex), and each tetrahedron
contributes 3 interior (bubble) DOFs.

# Fields
- `mesh::TetMesh` — underlying mesh
- `n_face_dofs::Int` — total number of face DOFs (3 × number of internal faces)
- `face_dof_tet_plus::Vector{Int}` — T⁺ tet for each face DOF
- `face_dof_tet_minus::Vector{Int}` — T⁻ tet for each face DOF
- `face_dof_local_plus::Vector{Int}` — local ref DOF index (1..12) in T⁺
- `face_dof_local_minus::Vector{Int}` — local ref DOF index (1..12) in T⁻
- `tet_J::Vector{SMatrix{3,3,Float64,9}}` — Jacobian matrix per tet
- `tet_Jinv::Vector{SMatrix{3,3,Float64,9}}` — inverse Jacobian per tet
- `tet_detJ::Vector{Float64}` — signed det(J) per tet (= 6 × signed volume)
"""
struct RT1Basis <: AbstractDivBasis
    mesh::TetMesh
    n_face_dofs::Int
    face_dof_tet_plus::Vector{Int}
    face_dof_tet_minus::Vector{Int}
    face_dof_local_plus::Vector{Int}
    face_dof_local_minus::Vector{Int}
    tet_J::Vector{SMatrix{3,3,Float64,9}}
    tet_Jinv::Vector{SMatrix{3,3,Float64,9}}
    tet_detJ::Vector{Float64}
end

"""
    n_basis(basis::RT1Basis) -> Int

Total number of RT1 DOFs: 3 × N_internal_faces + 3 × N_tets.
"""
@inline n_basis(basis::RT1Basis) = basis.n_face_dofs + 3 * length(basis.mesh.tets)

"""
    support_tets(basis::RT1Basis, n::Int)

Return the tetrahedra supporting DOF `n`. Face DOFs return a 2-tuple,
interior DOFs return a 1-tuple.
"""
@inline function support_tets(basis::RT1Basis, n::Int)
    if n <= basis.n_face_dofs
        return (basis.face_dof_tet_plus[n], basis.face_dof_tet_minus[n])
    else
        k = n - basis.n_face_dofs
        return ((k - 1) ÷ 3 + 1,)
    end
end

# ---------------------------------------------------------------------------
# Internal: resolve (global DOF, tet) → (local ref DOF index, sign)
# Returns (0, 0.0) if the DOF is not supported in `tet`.
# ---------------------------------------------------------------------------
@inline function _rt1_local_info(basis::RT1Basis, n::Int, tet::Int)
    if n <= basis.n_face_dofs
        if tet == basis.face_dof_tet_plus[n]
            return basis.face_dof_local_plus[n], 1.0
        elseif tet == basis.face_dof_tet_minus[n]
            return basis.face_dof_local_minus[n], -1.0
        end
    else
        k = n - basis.n_face_dofs
        interior_tet = (k - 1) ÷ 3 + 1
        if tet == interior_tet
            return 13 + (k - 1) % 3, 1.0   # local ref DOF 13, 14, or 15
        end
    end
    return 0, 0.0
end

"""
    evaluate(basis::RT1Basis, n::Int, r::Vec3, tet::Int) -> Vec3

Value of the `n`-th RT1 basis function at point `r` in tetrahedron `tet`.
Uses the contravariant Piola transform: `φ^phys = (σ/det J) J φ^ref(ξ)`.
"""
@inline function evaluate(basis::RT1Basis, n::Int, r::Vec3, tet::Int)
    local_idx, σ = _rt1_local_info(basis, n, tet)
    local_idx == 0 && return zero(Vec3)
    v0 = basis.mesh.nodes[basis.mesh.tets[tet][1]]
    ξ = basis.tet_Jinv[tet] * (r - v0)
    φ_ref = rt1_ref_evaluate(local_idx, ξ[1], ξ[2], ξ[3])
    return (σ / basis.tet_detJ[tet]) * (basis.tet_J[tet] * φ_ref)
end

"""
    divergence(basis::RT1Basis, n::Int, r::Vec3, tet::Int) -> Float64

Divergence of the `n`-th RT1 basis function at point `r` in tetrahedron `tet`.
For RT1 the divergence is a *linear* function of position (unlike the constant
divergence of RT0/SWG).

Uses `div φ^phys = (σ/det J) div^ref φ^ref(ξ)`.
"""
@inline function divergence(basis::RT1Basis, n::Int, r::Vec3, tet::Int)
    local_idx, σ = _rt1_local_info(basis, n, tet)
    local_idx == 0 && return 0.0
    v0 = basis.mesh.nodes[basis.mesh.tets[tet][1]]
    ξ = basis.tet_Jinv[tet] * (r - v0)
    div_ref = rt1_ref_divergence(local_idx, ξ[1], ξ[2], ξ[3])
    return σ * div_ref / basis.tet_detJ[tet]
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    build_rt1_basis(mesh::TetMesh) -> RT1Basis

Construct the RT1 basis on `mesh`. Enumerates internal faces (shared by
exactly two tetrahedra), creates 3 face DOFs per internal face (one per
face vertex, ordered by sorted global node index), and precomputes the
per-tet Jacobian data for the Piola transform.

The T⁺/T⁻ assignment is deterministic: the tet with the smaller global
index becomes T⁺ (σ = +1).
"""
function build_rt1_basis(mesh::TetMesh)
    # --- Step 1: enumerate internal faces (identical logic to build_swg_basis) ---
    face_table = Dict{NTuple{3,Int},Vector{Tuple{Int,Int}}}()
    for (ti, tet) in enumerate(mesh.tets)
        for (lf, idxs) in enumerate(TET_LOCAL_FACES)
            tri = (tet[idxs[1]], tet[idxs[2]], tet[idxs[3]])
            key = _sorted_triple(tri)
            push!(get!(() -> Tuple{Int,Int}[], face_table, key), (ti, lf))
        end
    end

    # --- Step 2: create face DOFs ---
    face_dof_tet_plus = Int[]
    face_dof_tet_minus = Int[]
    face_dof_local_plus = Int[]
    face_dof_local_minus = Int[]

    # Sort face keys for deterministic DOF ordering
    sorted_face_keys = sort(collect(keys(face_table)))

    for key in sorted_face_keys
        owners = face_table[key]
        length(owners) != 2 && continue    # skip boundary / non-manifold

        (ta, la), (tb, lb) = owners
        # Deterministic T⁺/T⁻: smaller tet index = T⁺
        if ta > tb
            ta, la, tb, lb = tb, lb, ta, la
        end

        # Reference face indices (0-based)
        ref_face_a = LOCAL_FACE_TO_REF_FACE[la]
        ref_face_b = LOCAL_FACE_TO_REF_FACE[lb]

        # Physical nodes on each ref face (in Lagrange DOF order)
        nodes_a = _ref_face_phys_nodes(mesh.tets[ta], ref_face_a)
        nodes_b = _ref_face_phys_nodes(mesh.tets[tb], ref_face_b)

        # For each sorted face node, find the matching sub-DOF in each tet
        for sorted_node in key
            sub_a = _find_sub(nodes_a, sorted_node)
            sub_b = _find_sub(nodes_b, sorted_node)
            # 1-based local ref DOF index: ref_face * 3 + sub
            local_a = 3 * ref_face_a + sub_a
            local_b = 3 * ref_face_b + sub_b

            push!(face_dof_tet_plus, ta)
            push!(face_dof_tet_minus, tb)
            push!(face_dof_local_plus, local_a)
            push!(face_dof_local_minus, local_b)
        end
    end

    n_face_dofs = length(face_dof_tet_plus)

    # --- Step 3: precompute per-tet Jacobian data ---
    nt = length(mesh.tets)
    tet_J_arr = Vector{SMatrix{3,3,Float64,9}}(undef, nt)
    tet_Jinv_arr = Vector{SMatrix{3,3,Float64,9}}(undef, nt)
    tet_detJ_arr = Vector{Float64}(undef, nt)

    for t in 1:nt
        v0 = mesh.nodes[mesh.tets[t][1]]
        v1 = mesh.nodes[mesh.tets[t][2]]
        v2 = mesh.nodes[mesh.tets[t][3]]
        v3 = mesh.nodes[mesh.tets[t][4]]
        J = SMatrix{3,3,Float64}(
            v1[1]-v0[1], v1[2]-v0[2], v1[3]-v0[3],  # column 1
            v2[1]-v0[1], v2[2]-v0[2], v2[3]-v0[3],  # column 2
            v3[1]-v0[1], v3[2]-v0[2], v3[3]-v0[3],  # column 3
        )
        tet_J_arr[t] = J
        tet_Jinv_arr[t] = inv(J)
        tet_detJ_arr[t] = det(J)
    end

    return RT1Basis(mesh, n_face_dofs,
                    face_dof_tet_plus, face_dof_tet_minus,
                    face_dof_local_plus, face_dof_local_minus,
                    tet_J_arr, tet_Jinv_arr, tet_detJ_arr)
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Physical node indices on a reference face (in Lagrange DOF order)."""
@inline function _ref_face_phys_nodes(tet_nodes::TetVerts, ref_face::Int)
    verts = REF_FACE_VERTS[ref_face + 1]  # 1-based tuple index
    return (tet_nodes[verts[1]], tet_nodes[verts[2]], tet_nodes[verts[3]])
end

"""Find sub-index (1,2,3) of `node` in a triple of nodes."""
@inline function _find_sub(nodes::NTuple{3,Int}, node::Int)
    nodes[1] == node && return 1
    nodes[2] == node && return 2
    nodes[3] == node && return 3
    error("node $node not found in face nodes $nodes")
end
