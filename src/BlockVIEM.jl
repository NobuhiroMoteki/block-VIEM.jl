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

# Phase 2: Singular volume integration (Duffy)
include("quadrature.jl")
include("duffy.jl")
include("green.jl")
include("impedance.jl")

# Phase 3: AIM (FFT-MVP)
include("aim_grid.jl")
include("aim_projection.jl")
include("aim_toeplitz.jl")
include("aim_operator.jl")

# Public API (Phase 1)
export Vec3, TetVerts
export TetMesh, n_nodes, n_tets, total_volume
export tet_volume, tet_signed_volume, tet_centroid, triangle_area
export SWGBasis, build_swg_basis, n_basis, evaluate, divergence
export read_msh

# Public API (Phase 2)
export TetQuadRule, TET_QUAD_1PT, TET_QUAD_4PT, TET_QUAD_5PT
export bary_to_point, integrate, gauss_legendre_unit
export DuffyQuadRule, duffy_reference_rule, duffy_quadrature
export subdivide_around, duffy_quadrature_around
export helmholtz_green, helmholtz_green_static
export impedance_element, assemble_impedance_matrix

# Public API (Phase 3)
export AIMGrid, n_grid_points, grid_point, aim_grid, grid_stencil, grid_point_at_linear
export AIMProjection, build_aim_projection, basis_centroid
export basis_moments, divergence_moments, multi_indices, n_moments
export build_green_toeplitz, precompute_green_fft, fft_convolve, fft_convolve!
export assemble_mass_matrix, near_pairs
export aim_far_mvp, aim_radiation_element, assemble_precorrection
export AIMOperator, build_aim_operator, aim_mvp

# Phase 2 (cont.): full singular pair Z_mn evaluator  — basic scalar API done
# Phase 4: Solver
include("incident.jl")
include("solver.jl")
# Public API (Phase 4)
export project_plane_wave, SolveResult
export solve_direct, solve_iterative

# Phase 5: PostProcess (CAS-v2 observables)           — TODO

end # module BlockVIEM
