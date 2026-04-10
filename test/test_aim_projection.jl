using Test
using StaticArrays
using LinearAlgebra: norm
using SparseArrays: nnz, findnz
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

function unit_cube_mesh_aim()
    nodes = Vec3[
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
        Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1),
    ]
    tets = TetVerts[
        TetVerts(1, 2, 3, 7), TetVerts(1, 3, 4, 7),
        TetVerts(1, 5, 6, 7), TetVerts(1, 6, 2, 7),
        TetVerts(1, 4, 8, 7), TetVerts(1, 8, 5, 7),
    ]
    return TetMesh(nodes, tets)
end

@testset "AIM moment matching" begin
    @testset "multi-index counts" begin
        @test length(multi_indices(0)) == 1
        @test length(multi_indices(1)) == 4
        @test length(multi_indices(2)) == 10
        @test length(multi_indices(3)) == 20
        @test n_moments(2) == 10
        # Each entry sums to a value in 0:P.
        for P in 0:3
            @test all(t -> 0 <= sum(t) <= P, multi_indices(P))
        end
    end

    @testset "basis_centroid lies inside the basis support" begin
        mesh = unit_cube_mesh_aim()
        basis = build_swg_basis(mesh)
        for n in 1:n_basis(basis)
            c = basis_centroid(basis, n)
            # Centroid should sit somewhere strictly inside the unit cube.
            @test all(0 .≤ c .≤ 1)
        end
    end

    @testset "build_aim_projection on the unit cube (P=2, M=3)" begin
        mesh = unit_cube_mesh_aim()
        basis = build_swg_basis(mesh)
        grid = aim_grid(mesh; pitch = 0.25, padding = 3)
        proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)

        N = n_basis(basis)
        Ng = n_grid_points(grid)
        @test size(proj.Wx) == (N, Ng)
        @test size(proj.Wy) == (N, Ng)
        @test size(proj.Wz) == (N, Ng)
        @test size(proj.Wdiv) == (N, Ng)
        # Each row should hit exactly 27 grid points (full 3³ stencil).
        for n in 1:N
            @test nnz(proj.Wx[n, :]) == 27
            @test nnz(proj.Wdiv[n, :]) == 27
        end
        @test proj.poly_order == 2
        @test proj.stencil == 3
    end

    @testset "moment recovery: weights reproduce target moments" begin
        # By construction, the polynomial system Φ w = M_target is satisfied
        # by the QR-based minimum-norm solution. Verify it numerically: for
        # every basis function and every multi-index, the weighted monomial
        # sum on the stencil must match the analytic target moment.
        mesh = unit_cube_mesh_aim()
        basis = build_swg_basis(mesh)
        grid = aim_grid(mesh; pitch = 0.25, padding = 3)
        P = 2
        rule = TET_QUAD_5PT
        proj = build_aim_projection(basis, grid; poly_order = P, stencil = 3, rule = rule)
        indices = multi_indices(P)

        max_err = 0.0
        for n in 1:n_basis(basis)
            c = basis_centroid(basis, n)
            M_target = basis_moments(basis, n, c, indices, rule)
            row_x = proj.Wx[n, :]
            row_y = proj.Wy[n, :]
            row_z = proj.Wz[n, :]
            for k in eachindex(indices)
                abc = indices[k]
                sx = sy = sz = 0.0
                for (col, wx) in zip(findnz(row_x)...)
                    rj = grid_point_at_linear(grid, col)
                    m = (rj[1] - c[1])^abc[1] *
                        (rj[2] - c[2])^abc[2] *
                        (rj[3] - c[3])^abc[3]
                    sx += wx * m
                end
                for (col, wy) in zip(findnz(row_y)...)
                    rj = grid_point_at_linear(grid, col)
                    m = (rj[1] - c[1])^abc[1] *
                        (rj[2] - c[2])^abc[2] *
                        (rj[3] - c[3])^abc[3]
                    sy += wy * m
                end
                for (col, wz) in zip(findnz(row_z)...)
                    rj = grid_point_at_linear(grid, col)
                    m = (rj[1] - c[1])^abc[1] *
                        (rj[2] - c[2])^abc[2] *
                        (rj[3] - c[3])^abc[3]
                    sz += wz * m
                end
                max_err = max(max_err,
                              abs(sx - M_target[k, 1]),
                              abs(sy - M_target[k, 2]),
                              abs(sz - M_target[k, 3]))
            end
        end
        @test max_err < 1e-10
    end

    @testset "zeroth moment equals total basis integral (P=0)" begin
        # With P = 0 the only moment is the constant; the projection becomes
        # the simple "monopole" approximation. Sum of weights must equal
        # ∫ f_n^α dV exactly (subject to the QR solver tolerance).
        mesh = unit_cube_mesh_aim()
        basis = build_swg_basis(mesh)
        grid = aim_grid(mesh; pitch = 0.25, padding = 3)
        proj = build_aim_projection(basis, grid; poly_order = 0, stencil = 3)
        indices_P0 = multi_indices(0)

        for n in 1:n_basis(basis)
            c = basis_centroid(basis, n)
            M_target = basis_moments(basis, n, c, indices_P0, TET_QUAD_5PT)
            sum_x = sum(proj.Wx[n, :])
            sum_y = sum(proj.Wy[n, :])
            sum_z = sum(proj.Wz[n, :])
            @test isapprox(sum_x, M_target[1, 1]; atol = 1e-12)
            @test isapprox(sum_y, M_target[1, 2]; atol = 1e-12)
            @test isapprox(sum_z, M_target[1, 3]; atol = 1e-12)
        end
    end

    @testset "Wdiv moment recovery" begin
        # Same moment-matching check as for Wx/Wy/Wz, but for the scalar
        # divergence. The divergence is piecewise constant, so its moments
        # are exactly reproduced by the TET_QUAD_5PT rule.
        mesh = unit_cube_mesh_aim()
        basis = build_swg_basis(mesh)
        grid = aim_grid(mesh; pitch = 0.25, padding = 3)
        P = 2
        rule = TET_QUAD_5PT
        proj = build_aim_projection(basis, grid; poly_order = P, stencil = 3, rule = rule)
        indices = multi_indices(P)

        max_err = 0.0
        for n in 1:n_basis(basis)
            c = basis_centroid(basis, n)
            M_div_tgt = divergence_moments(basis, n, c, indices, rule)
            row_div = proj.Wdiv[n, :]
            for k in eachindex(indices)
                abc = indices[k]
                s = 0.0
                for (col, wdiv) in zip(findnz(row_div)...)
                    rj = grid_point_at_linear(grid, col)
                    m = (rj[1] - c[1])^abc[1] *
                        (rj[2] - c[2])^abc[2] *
                        (rj[3] - c[3])^abc[3]
                    s += wdiv * m
                end
                max_err = max(max_err, abs(s - M_div_tgt[k]))
            end
        end
        @test max_err < 1e-10
    end

    @testset "argument validation" begin
        mesh = unit_cube_mesh_aim()
        basis = build_swg_basis(mesh)
        grid = aim_grid(mesh; pitch = 0.25, padding = 3)
        # stencil^3 < #moments must throw.
        @test_throws ArgumentError build_aim_projection(basis, grid;
                                                        poly_order = 3,
                                                        stencil = 2)
        # Negative poly order via multi_indices.
        @test_throws ArgumentError multi_indices(-1)
    end
end
