using Test
using BlockVIEM

@testset "BlockVIEM.jl" begin
    include("test_mesh.jl")
    include("test_swg.jl")
    include("test_gmsh_io.jl")
    include("test_quadrature.jl")
    include("test_duffy.jl")
    include("test_green.jl")
end
