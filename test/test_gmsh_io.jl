using Test
using BlockVIEM
import Gmsh: gmsh

# Build a tiny .msh on the fly via the Gmsh API and read it back through
# BlockVIEM.read_msh, then verify counts and consistency.

function generate_unit_cube_msh(path::AbstractString; lc::Float64 = 0.5)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("unit_cube")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1)
        gmsh.model.occ.synchronize()
        # tag the volume as physical group 42 so we can verify phys_tags
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
    mktempdir() do dir
        path = joinpath(dir, "unit_cube.msh")
        generate_unit_cube_msh(path; lc = 0.6)
        mesh = read_msh(path)

        @test n_nodes(mesh) >= 8           # at least the cube corners
        @test n_tets(mesh) >= 1
        @test all(>(0), mesh.tet_volumes)
        # The total volume of the unit cube must be 1 to within mesh tolerance.
        @test isapprox(total_volume(mesh), 1.0; atol = 1e-12)
        # Every tetrahedron should be tagged with physical group 42.
        @test all(==(42), mesh.tet_phys_tags)

        # And the SWG basis should successfully build on top of it.
        basis = build_swg_basis(mesh)
        @test n_basis(basis) >= 1
        @test all(>(0), basis.face_areas)
    end
end
