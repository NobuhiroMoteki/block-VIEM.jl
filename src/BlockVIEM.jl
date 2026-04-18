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
include("gre_mesh.jl")
include("aggregate_mesh.jl")

# RT1 reference element and physical basis
include("reference_rt1.jl")
include("rt1_basis.jl")

# Phase 2: Singular volume integration (Duffy)
include("quadrature.jl")
include("duffy.jl")
include("triangle_quad.jl")
include("green.jl")
include("surface_integrals.jl")
include("impedance.jl")

# Phase 3: AIM (FFT-MVP)
include("aim_grid.jl")
include("aim_projection.jl")
include("aim_toeplitz.jl")
include("aim_operator.jl")

# Public API (Phase 1)
export Vec3, TetVerts
export TetMesh, n_nodes, n_tets, total_volume, mean_edge_length
export tet_volume, tet_signed_volume, tet_centroid, triangle_area
export AbstractDivBasis, SWGBasis, build_swg_basis, n_basis, evaluate, divergence, support_tets
export RT1Basis, build_rt1_basis, build_tet_to_dofs
export read_msh
export GREParams, gre_semi_axes, adaptive_lc, gre_surface, gre_mesh, gre_mesh_with_field
export SphereAggregate, n_monomers, monomer_volume_sum,
       aggregate_bounding_radius, aggregate_centroid, recenter!
export neck_ratio_to_overlap, overlap_to_neck_ratio, pair_neck_radius
export ptsa_h5_key, load_ptsa_h5, list_ptsa_keys
export make_linear_chain, make_planar_array,
       make_fcc_cluster, make_bcc_cluster, make_hcp_cluster
export adaptive_lc_aggregate, mesh_sphere_aggregate

# Public API (Phase 2)
export TetQuadRule, TET_QUAD_1PT, TET_QUAD_4PT, TET_QUAD_5PT, TET_QUAD_64PT, TET_QUAD_125PT
export tet_collapsed_rule
export bary_to_point, integrate, gauss_legendre_unit
export DuffyQuadRule, duffy_reference_rule, duffy_quadrature
export subdivide_around, duffy_quadrature_around
export TriQuadRule, TRI_QUAD_1PT, TRI_QUAD_3PT, tri_collapsed_rule
export tri_bary_to_point, integrate_tri
export TriDuffyRule, tri_duffy_reference_rule, tri_duffy_quadrature
export subdivide_triangle_around, tri_duffy_quadrature_around
export surface_integral_G, boundary_face_vertices
export helmholtz_green, helmholtz_green_static
export impedance_element, assemble_impedance_matrix, assemble_half_swg_correction

# Public API (Phase 3)
export AIMGrid, n_grid_points, grid_point, aim_grid, grid_stencil, grid_point_at_linear
export AIMProjection, build_aim_projection, basis_centroid
export basis_moments, divergence_moments, multi_indices, n_moments
export build_green_toeplitz, precompute_green_fft, fft_convolve, fft_convolve!
export assemble_mass_matrix, near_pairs, near_pairs_by_distance
export aim_far_mvp, aim_radiation_element, assemble_precorrection
export AIMOperator, build_aim_operator, aim_mvp

# Phase 2 (cont.): full singular pair Z_mn evaluator  — basic scalar API done
# Phase 4: Solver
include("incident.jl")
include("block_krylov.jl")
include("solver.jl")
# Public API (Phase 4)
export project_plane_wave, SolveResult
export solve_direct, solve_iterative, solve_iterative_block
export BlockSolveResult, block_bicgstab, block_gmres

# Phase 5: PostProcess (CAS-v2 observables)
include("postprocess.jl")
include("spheroid_sweep_io.jl")

# Public API (Phase 5)
export far_field_amplitude, ScatteringResult, compute_scattering
export SphericalQuadRule, spherical_product_rule, compute_csca_farfield
export CASOrientation, cas_orientation, CASv2Result, compute_cas_observables
export solve_cas_v2_orientations
export SpheroidSweepGrids, SpheroidSweepData
export write_spheroid_sweep_h5, read_spheroid_sweep_h5
export expand_alpha_from_alpha0

end # module BlockVIEM
