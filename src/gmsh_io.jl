# Gmsh `.msh` reader -> TetMesh.
# Uses the Gmsh.jl C-API wrapper.

import Gmsh: gmsh

const GMSH_TET4 = 4   # 4-node linear tetrahedron element type

"""
    read_msh(filename::AbstractString) -> TetMesh

Read a Gmsh `.msh` file (format 2 or 4) and return a [`TetMesh`](@ref).
Only 4-node linear tetrahedra are supported. Physical group tags are
attached per element via `tet_phys_tags`; tetrahedra not assigned to a
3D physical group receive tag `0`.

The function manages Gmsh initialization/finalization internally and is
safe to call from arbitrary contexts.
"""
function read_msh(filename::AbstractString)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(filename)
        return _extract_tetmesh()
    finally
        gmsh.finalize()
    end
end

function _extract_tetmesh()
    # --- nodes -------------------------------------------------------------
    node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
    n_nodes_raw = length(node_tags)
    coords = reshape(node_coords_flat, 3, n_nodes_raw)

    # build a tag -> 1-based index map (Gmsh tags are not necessarily contiguous)
    tag_to_idx = Dict{Int,Int}()
    nodes = Vector{Vec3}(undef, n_nodes_raw)
    for i in 1:n_nodes_raw
        tag_to_idx[Int(node_tags[i])] = i
        nodes[i] = Vec3(coords[1, i], coords[2, i], coords[3, i])
    end

    # --- tetrahedra (all entities, all physical groups) -------------------
    tet_tags_global, tet_node_tags_flat = gmsh.model.mesh.getElementsByType(GMSH_TET4)
    n_tets_raw = length(tet_tags_global)
    tets = Vector{TetVerts}(undef, n_tets_raw)
    conn = reshape(tet_node_tags_flat, 4, n_tets_raw)
    tag_to_local_tet = Dict{Int,Int}()
    for i in 1:n_tets_raw
        a = tag_to_idx[Int(conn[1, i])]
        b = tag_to_idx[Int(conn[2, i])]
        c = tag_to_idx[Int(conn[3, i])]
        d = tag_to_idx[Int(conn[4, i])]
        tets[i] = TetVerts(a, b, c, d)
        tag_to_local_tet[Int(tet_tags_global[i])] = i
    end

    # --- physical-group tags (3D groups only) -----------------------------
    phys_tags = zeros(Int, n_tets_raw)
    for (dim, ptag) in gmsh.model.getPhysicalGroups(3)
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
            etypes, etags, _ = gmsh.model.mesh.getElements(3, entity)
            for (etype_idx, etype) in enumerate(etypes)
                if etype != GMSH_TET4
                    continue
                end
                for global_tag in etags[etype_idx]
                    li = get(tag_to_local_tet, Int(global_tag), 0)
                    if li != 0
                        phys_tags[li] = Int(ptag)
                    end
                end
            end
        end
    end

    return TetMesh(nodes, tets; phys_tags = phys_tags)
end
