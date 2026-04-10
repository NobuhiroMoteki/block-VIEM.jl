using Test
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

@testset "TetMesh" begin
    @testset "single regular tetrahedron" begin
        # Reference tet with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1).
        nodes = Vec3[Vec3(0, 0, 0), Vec3(1, 0, 0),
                     Vec3(0, 1, 0), Vec3(0, 0, 1)]
        tets = TetVerts[TetVerts(1, 2, 3, 4)]
        mesh = TetMesh(nodes, tets)
        @test n_nodes(mesh) == 4
        @test n_tets(mesh) == 1
        @test mesh.tet_volumes[1] ≈ 1 / 6
        @test total_volume(mesh) ≈ 1 / 6
        @test mesh.tet_centroids[1] ≈ Vec3(0.25, 0.25, 0.25)
    end

    @testset "orientation auto-correction" begin
        # Reversed orientation should still yield positive stored volume.
        nodes = Vec3[Vec3(0, 0, 0), Vec3(1, 0, 0),
                     Vec3(0, 1, 0), Vec3(0, 0, 1)]
        tets = TetVerts[TetVerts(1, 2, 4, 3)]   # negative signed volume
        mesh = TetMesh(nodes, tets)
        @test mesh.tet_volumes[1] ≈ 1 / 6
        # The stored connectivity must now be right-handed.
        a, b, c, d = mesh.tets[1]
        sv = tet_signed_volume(mesh.nodes[a], mesh.nodes[b],
                               mesh.nodes[c], mesh.nodes[d])
        @test sv > 0
    end

    @testset "unit cube tessellated into 6 tets" begin
        # Standard 6-tet decomposition of the unit cube.
        nodes = Vec3[
            Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
            Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1),
        ]
        tets = TetVerts[
            TetVerts(1, 2, 3, 7),
            TetVerts(1, 3, 4, 7),
            TetVerts(1, 5, 6, 7),
            TetVerts(1, 6, 2, 7),
            TetVerts(1, 4, 8, 7),
            TetVerts(1, 8, 5, 7),
        ]
        mesh = TetMesh(nodes, tets)
        @test n_tets(mesh) == 6
        @test total_volume(mesh) ≈ 1.0
        @test all(v -> v > 0, mesh.tet_volumes)
    end

    @testset "type stability" begin
        nodes = Vec3[Vec3(0, 0, 0), Vec3(1, 0, 0),
                     Vec3(0, 1, 0), Vec3(0, 0, 1)]
        tets = TetVerts[TetVerts(1, 2, 3, 4)]
        mesh = TetMesh(nodes, tets)
        @inferred total_volume(mesh)
        @inferred tet_volume(nodes[1], nodes[2], nodes[3], nodes[4])
        @inferred triangle_area(nodes[1], nodes[2], nodes[3])
    end
end
