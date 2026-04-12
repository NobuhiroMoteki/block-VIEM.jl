using Test
using StaticArrays
using LinearAlgebra: norm, dot
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

# ----------------------------------------------------------------------------
# Helper: build the canonical 2-tet bipyramid sharing the face {n2, n3, n4}.
#
#   T1 vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
#   T2 vertices: (1,1,1), (1,0,0), (0,1,0), (0,0,1)
#
# Shared face = triangle (1,0,0)-(0,1,0)-(0,0,1) with area sqrt(3)/2.
# Free vertex of T1 = node 1 (origin); free vertex of T2 = node 5 (1,1,1).
# By the smaller-index-is-plus convention: T+ = T1, T- = T2.
# ----------------------------------------------------------------------------
function bipyramid_mesh()
    nodes = Vec3[
        Vec3(0, 0, 0), Vec3(1, 0, 0),
        Vec3(0, 1, 0), Vec3(0, 0, 1),
        Vec3(1, 1, 1),
    ]
    tets = TetVerts[
        TetVerts(1, 2, 3, 4),
        TetVerts(5, 2, 3, 4),
    ]
    return TetMesh(nodes, tets)
end

@testset "SWGBasis" begin
    @testset "topology counts: bipyramid" begin
        mesh = bipyramid_mesh()
        basis = build_swg_basis(mesh)
        @test n_basis(basis) == 1            # exactly one internal face
        @test basis.tet_plus[1] == 1         # smaller index is "+"
        @test basis.tet_minus[1] == 2
        @test basis.free_vertex_plus[1] == 1
        @test basis.free_vertex_minus[1] == 5
        @test basis.face_areas[1] ≈ sqrt(3) / 2
        @test sort(collect(basis.face_nodes[1])) == [2, 3, 4]
        @test basis.is_boundary[1] == false
    end

    @testset "divergence formula (Eq. 12)" begin
        mesh = bipyramid_mesh()
        basis = build_swg_basis(mesh)
        a = basis.face_areas[1]
        V_plus = mesh.tet_volumes[1]
        V_minus = mesh.tet_volumes[2]
        @test V_plus ≈ 1 / 6
        @test V_minus ≈ 1 / 3
        @test divergence(basis, 1, 1) ≈ +a / V_plus
        @test divergence(basis, 1, 2) ≈ -a / V_minus
        @test divergence(basis, 1, 0) == 0.0
    end

    @testset "evaluation at face centroid (Eq. 10)" begin
        mesh = bipyramid_mesh()
        basis = build_swg_basis(mesh)
        face_centroid = (mesh.nodes[2] + mesh.nodes[3] + mesh.nodes[4]) / 3
        v_plus = evaluate(basis, 1, face_centroid, 1)
        v_minus = evaluate(basis, 1, face_centroid, 2)

        # Normal continuity: f_n is single-valued at the shared face,
        # so f^+ and f^- must agree on the entire face.
        @test v_plus ≈ v_minus

        # Normal flux through the face must equal 1 (SWG normalization).
        n_hat = SVector{3,Float64}(1, 1, 1) / sqrt(3)   # outward from T1 to T2
        @test dot(v_plus, n_hat) ≈ 1.0
        @test dot(v_minus, n_hat) ≈ 1.0
    end

    @testset "evaluation outside support" begin
        mesh = bipyramid_mesh()
        basis = build_swg_basis(mesh)
        @test evaluate(basis, 1, Vec3(0.1, 0.1, 0.1), 0) == zero(Vec3)
    end

    @testset "type stability" begin
        mesh = bipyramid_mesh()
        basis = build_swg_basis(mesh)
        r = Vec3(0.3, 0.3, 0.3)
        @inferred evaluate(basis, 1, r, 1)
        @inferred divergence(basis, 1, 1)
    end

    @testset "unit cube: internal face count" begin
        # 6-tet decomposition of the unit cube — see test_mesh.jl.
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
        basis = build_swg_basis(mesh)
        # 6 tets * 4 faces = 24 face-occurrences. The cube surface has
        # 6 squares * 2 tris = 12 boundary triangles. Internal faces:
        # (24 - 12) / 2 = 6.
        @test n_basis(basis) == 6
        for n in 1:n_basis(basis)
            @test basis.face_areas[n] > 0
            @test basis.tet_plus[n] != basis.tet_minus[n]
            @test basis.tet_plus[n] < basis.tet_minus[n]
            @test !(basis.free_vertex_plus[n] in basis.face_nodes[n])
            @test !(basis.free_vertex_minus[n] in basis.face_nodes[n])
            @test basis.is_boundary[n] == false
        end
    end

    @testset "half-SWG opt-in: bipyramid with include_boundary_faces" begin
        mesh = bipyramid_mesh()
        basis = build_swg_basis(mesh; include_boundary_faces = true)
        # 1 internal face + 6 boundary triangles.
        @test n_basis(basis) == 7
        @test sum(basis.is_boundary) == 6
        @test sum(.!basis.is_boundary) == 1
        for n in 1:n_basis(basis)
            if basis.is_boundary[n]
                @test basis.tet_minus[n] == 0
                @test basis.free_vertex_minus[n] == 0
                @test basis.tet_plus[n] in (1, 2)
            end
        end
    end
end
