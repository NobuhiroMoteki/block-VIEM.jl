"""
    BlockVIEM

Volume Integral Equation Method (VIEM) solver for electromagnetic scattering by
arbitrarily shaped, high-contrast dielectric particles, written in Julia.

The package implements:
- SWG (Schaubert-Wilton-Glisson) basis on tetrahedral meshes
- Duffy-transform-based singular/near-singular volume integration
- AIM (Adaptive Integral Method) FFT acceleration
- Block-Krylov iterative solvers for multi-orientation batch problems
- CAS-v2 compatible scattering observables (mirroring block-DDA_Py)

See `.claude/technical_note.md` for the formulation reference.
"""
module BlockVIEM

# Phase 1: Mesh / SWG basis
include("mesh.jl")
include("swg.jl")
include("gmsh_io.jl")

# Public API (Phase 1)
export Vec3, TetVerts
export TetMesh, n_nodes, n_tets, total_volume
export tet_volume, tet_signed_volume, tet_centroid, triangle_area
export SWGBasis, build_swg_basis, n_basis, evaluate, divergence
export read_msh

# Phase 2: Singular volume integration (Duffy)        — TODO
# Phase 3: AIM (FFT-MVP)                              — TODO
# Phase 4: Block-Krylov solver                        — TODO
# Phase 5: PostProcess (CAS-v2 observables)           — TODO

end # module BlockVIEM
