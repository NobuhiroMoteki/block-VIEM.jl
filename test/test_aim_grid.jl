using Test
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

@testset "AIMGrid" begin
    @testset "explicit construction and indexing" begin
        g = AIMGrid(Vec3(0, 0, 0), 0.5, (5, 4, 3))
        @test n_grid_points(g) == 60
        @test grid_point(g, 1, 1, 1) ≈ Vec3(0, 0, 0)
        @test grid_point(g, 5, 4, 3) ≈ Vec3(2.0, 1.5, 1.0)
        # Linear indexing must round-trip with grid_point.
        L = LinearIndices((5, 4, 3))
        for k in 1:3, j in 1:4, i in 1:5
            lin = L[i, j, k]
            @test grid_point_at_linear(g, lin) ≈ grid_point(g, i, j, k)
        end
    end

    @testset "bounding-box auto-sizing from a tet mesh" begin
        nodes = Vec3[
            Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
            Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1),
        ]
        tets = TetVerts[
            TetVerts(1, 2, 3, 7), TetVerts(1, 3, 4, 7),
            TetVerts(1, 5, 6, 7), TetVerts(1, 6, 2, 7),
            TetVerts(1, 4, 8, 7), TetVerts(1, 8, 5, 7),
        ]
        mesh = TetMesh(nodes, tets)
        g = aim_grid(mesh; pitch = 0.25, padding = 2)
        # Bounding box [0,1]^3 with 2-cell padding on each side at h=0.25
        # gives a [-0.5, 1.5]^3 box → at least 9 points per axis.
        @test g.dims[1] >= 9
        @test g.dims[2] >= 9
        @test g.dims[3] >= 9
        @test g.origin[1] ≤ -0.5 + 1e-12
        @test g.origin[2] ≤ -0.5 + 1e-12
        @test g.origin[3] ≤ -0.5 + 1e-12
        # All mesh nodes must lie strictly inside the grid box.
        max_x = g.origin[1] + g.pitch * (g.dims[1] - 1)
        max_y = g.origin[2] + g.pitch * (g.dims[2] - 1)
        max_z = g.origin[3] + g.pitch * (g.dims[3] - 1)
        for n in nodes
            @test g.origin[1] ≤ n[1] ≤ max_x
            @test g.origin[2] ≤ n[2] ≤ max_y
            @test g.origin[3] ≤ n[3] ≤ max_z
        end
    end

    @testset "stencil lookup (odd and even M)" begin
        g = AIMGrid(Vec3(0, 0, 0), 1.0, (7, 7, 7))
        # Odd stencil M=3 centred on the nearest grid point.
        sten = grid_stencil(g, Vec3(3.4, 3.6, 3.0), 3)   # nearest = (4,5,4) (1-based: 4)
        @test length(sten) == 27
        # Centre of stencil = grid point (4,5,4) (1-based) = (3,4,3) physical.
        L = LinearIndices((7, 7, 7))
        @test L[4, 5, 4] in sten
        @test L[3, 4, 3] in sten
        @test L[5, 6, 5] in sten

        # Even stencil M=2 anchored at the lower-left of the containing cell.
        sten2 = grid_stencil(g, Vec3(2.3, 2.7, 2.5), 2)   # cell starts at (3,3,3) (1-based)
        @test length(sten2) == 8
        @test L[3, 3, 3] in sten2
        @test L[4, 4, 4] in sten2
    end

    @testset "stencil truncation near boundary" begin
        g = AIMGrid(Vec3(0, 0, 0), 1.0, (3, 3, 3))
        # Centre at the corner: half the M=3 stencil falls outside.
        sten = grid_stencil(g, Vec3(0, 0, 0), 3)
        @test length(sten) == 8     # 2x2x2 corner remains
    end

    @testset "argument validation" begin
        nodes = Vec3[Vec3(0, 0, 0), Vec3(1, 0, 0),
                     Vec3(0, 1, 0), Vec3(0, 0, 1)]
        tets = TetVerts[TetVerts(1, 2, 3, 4)]
        mesh = TetMesh(nodes, tets)
        @test_throws ArgumentError aim_grid(mesh; pitch = 0.0)
        @test_throws ArgumentError aim_grid(mesh; pitch = -0.1)
        @test_throws ArgumentError aim_grid(mesh; pitch = 0.5, padding = -1)
        g = AIMGrid(Vec3(0, 0, 0), 1.0, (3, 3, 3))
        @test_throws ArgumentError grid_stencil(g, Vec3(1, 1, 1), 0)
    end
end
