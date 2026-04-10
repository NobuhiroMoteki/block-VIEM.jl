using Test
using BlockVIEM
import Gmsh: gmsh

include(joinpath(@__DIR__, "meshes", "build.jl"))

# Inline OCC mesh generator (kept for an end-to-end smoke test that does
# not depend on the .geo files in test/meshes/).
function generate_unit_cube_msh(path::AbstractString; lc::Float64 = 0.5)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("unit_cube_inline")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 42)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
end

@testset "Gmsh I/O" begin
    @testset "inline OCC unit cube" begin
        mktempdir() do dir
            path = joinpath(dir, "unit_cube.msh")
            generate_unit_cube_msh(path; lc = 0.6)
            mesh = read_msh(path)
            @test n_nodes(mesh) >= 8
            @test n_tets(mesh) >= 1
            @test all(>(0), mesh.tet_volumes)
            @test isapprox(total_volume(mesh), 1.0; atol = 1e-12)
            @test all(==(42), mesh.tet_phys_tags)
            basis = build_swg_basis(mesh)
            @test n_basis(basis) >= 1
            @test all(>(0), basis.face_areas)
        end
    end

    @testset "test/meshes/cube.geo" begin
        path = ensure_msh("cube")
        mesh = read_msh(path)
        @test n_tets(mesh) >= 1
        @test isapprox(total_volume(mesh), 1.0; atol = 1e-12)
        @test all(==(42), mesh.tet_phys_tags)
    end

    @testset "test/meshes/sphere.geo" begin
        path = ensure_msh("sphere")
        mesh = read_msh(path)
        @test n_tets(mesh) >= 4
        @test all(>(0), mesh.tet_volumes)
        # Coarse mesh: total volume should approximate (4/3)π but not equal
        # exactly. Allow a generous 20% tolerance for the default lc=0.4.
        @test isapprox(total_volume(mesh), 4π / 3; rtol = 0.2)
        @test all(==(1), mesh.tet_phys_tags)
        # SWG basis must build without errors and contain at least one
        # internal face.
        basis = build_swg_basis(mesh)
        @test n_basis(basis) >= 1
    end
end
