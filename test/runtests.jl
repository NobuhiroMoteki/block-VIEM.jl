using Test
using BlockVIEM

@testset "BlockVIEM.jl" begin
    @testset "smoke" begin
        # Phase 0 smoke test: package loads.
        @test isdefined(Main, :BlockVIEM)
    end
end
