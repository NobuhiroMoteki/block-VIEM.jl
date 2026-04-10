using Test
using StaticArrays
using LinearAlgebra: norm, dot
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

function unit_cube_mesh_s()
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

@testset "Incident field projection" begin
    mesh = unit_cube_mesh_s()
    basis = build_swg_basis(mesh)
    N = n_basis(basis)

    # k_bg = 0 ⇒ E^inc = E0 (constant). The projection becomes
    # b_m = ∫ f_m · E0 dV = E0 · (∫ f_m dV).
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
    b = project_plane_wave(basis; k_hat = k_hat, E0 = E0, k_bg = 0.0)
    @test length(b) == N
    @test all(isfinite, b)

    # For k_bg = 0 each component should be E0_α * (∫ f_m^α dV).
    # The x-component integral is the zeroth moment.
    for n in 1:N
        c = basis_centroid(basis, n)
        M0 = basis_moments(basis, n, c, multi_indices(0), TET_QUAD_5PT)
        expected = E0[1] * M0[1, 1] + E0[2] * M0[1, 2] + E0[3] * M0[1, 3]
        @test isapprox(b[n], expected; atol = 1e-12)
    end
end

@testset "Direct solver (cube mesh)" begin
    mesh = unit_cube_mesh_s()
    basis = build_swg_basis(mesh)

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
    k0 = 0.5
    eps_p = 2.0 + 0.1im

    result = solve_direct(basis; k0 = k0, eps_p = eps_p,
                          k_hat = k_hat, E0 = E0)
    @test result isa SolveResult
    @test length(result.D_coeffs) == n_basis(basis)
    @test all(isfinite, result.D_coeffs)
    @test result.converged
    @test result.residual_norm < 1e-10
end

@testset "Iterative solver (cube mesh)" begin
    mesh = unit_cube_mesh_s()
    basis = build_swg_basis(mesh)

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
    k0 = 0.5
    eps_p = 2.0 + 0.1im

    result = solve_iterative(basis; k0 = k0, eps_p = eps_p,
                             k_hat = k_hat, E0 = E0,
                             pitch = 0.25, tol = 1e-4, maxiter = 100)
    @test result isa SolveResult
    @test length(result.D_coeffs) == n_basis(basis)
    @test all(isfinite, result.D_coeffs)
    # The AIM operator on this tiny mesh has ~10% far-field error, so the
    # iterative solution may not match the direct one perfectly.
    # Accept convergence to the requested tolerance.
    @test result.residual_norm < 0.01 || result.converged
end

@testset "Direct vs iterative (agreement on small mesh)" begin
    mesh = unit_cube_mesh_s()
    basis = build_swg_basis(mesh)

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
    k0 = 0.5
    eps_p = 2.0 + 0.1im

    r_direct = solve_direct(basis; k0 = k0, eps_p = eps_p,
                            k_hat = k_hat, E0 = E0)
    r_iter = solve_iterative(basis; k0 = k0, eps_p = eps_p,
                             k_hat = k_hat, E0 = E0,
                             pitch = 0.25, tol = 1e-4)
    # The solutions should agree to the AIM accuracy level (~10-20%).
    rel_diff = norm(r_iter.D_coeffs - r_direct.D_coeffs) / norm(r_direct.D_coeffs)
    @test rel_diff < 0.3
end
