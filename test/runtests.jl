using Test
using BlockVIEM

@testset "BlockVIEM.jl" begin
    include("test_mesh.jl")
    include("test_swg.jl")
    include("test_rt1_basis.jl")
    include("test_gmsh_io.jl")
    include("test_quadrature.jl")
    include("test_duffy.jl")
    include("test_triangle_quad.jl")
    include("test_surface_integrals.jl")
    include("test_half_swg_kb.jl")
    include("test_half_swg_aim.jl")
    include("test_green.jl")
    include("test_impedance.jl")
    include("test_aim_grid.jl")
    include("test_aim_projection.jl")
    include("test_aim_toeplitz.jl")
    include("test_aim_operator.jl")
    include("test_solver.jl")
    include("test_postprocess.jl")
    include("test_mie_validation.jl")
    include("test_cas_v2.jl")
end
