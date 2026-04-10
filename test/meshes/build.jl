# Helper to build .msh files from .geo sources on demand.
#
# Usage from a test file:
#   include(joinpath(@__DIR__, "meshes", "build.jl"))
#   path = ensure_msh("sphere")          # builds test/meshes/sphere.msh if missing
#
# .msh files live next to their .geo source and are gitignored.

import Gmsh: gmsh

const _MESH_DIR = @__DIR__

"""
    ensure_msh(name::AbstractString) -> String

Build `<name>.msh` from `<name>.geo` in `test/meshes/` if the .msh is missing
or older than the .geo. Returns the absolute path to the .msh file.
"""
function ensure_msh(name::AbstractString)
    geo = joinpath(_MESH_DIR, name * ".geo")
    msh = joinpath(_MESH_DIR, name * ".msh")
    isfile(geo) || error("missing .geo source: $geo")
    if !isfile(msh) || mtime(msh) < mtime(geo)
        _build(geo, msh)
    end
    return msh
end

function _build(geo::AbstractString, msh::AbstractString)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(geo)
        gmsh.model.mesh.generate(3)
        gmsh.write(msh)
    finally
        gmsh.finalize()
    end
    return msh
end
