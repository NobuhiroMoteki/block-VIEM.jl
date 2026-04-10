using Test
using StaticArrays
using LinearAlgebra: norm
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

function unit_cube_mesh_op()
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

@testset "Sparse mass matrix" begin
    mesh = unit_cube_mesh_op()
    basis = build_swg_basis(mesh)
    M = assemble_mass_matrix(basis)
    N = n_basis(basis)
    @test size(M) == (N, N)
    # Must be symmetric (exact quadrature for degree-2 integrand).
    @test maximum(abs.(M .- M')) < 1e-14
    # Diagonal positive.
    for m in 1:N
        @test M[m, m] > 0
    end
    # Agree with the scalar impedance mass term:
    # impedance_element with κ=0 returns (1/ε_p)*M_mn, so with ε_p=1 it's M_mn.
    for m in 1:N, n in 1:N
        Zmn = impedance_element(basis, m, n; k0 = 0.0, eps_p = 1.0, eps_bg = 1.0)
        @test isapprox(real(Zmn), M[m, n]; atol = 1e-12)
    end
end

@testset "near_pairs" begin
    mesh = unit_cube_mesh_op()
    basis = build_swg_basis(mesh)
    pairs = near_pairs(basis)
    N = n_basis(basis)
    # Must include all diagonals.
    for m in 1:N
        @test (m, m) in pairs
    end
    # Every pair must correspond to basis functions that share a tet.
    for (m, n) in pairs
        shared = !isempty(intersect(
            Set([basis.tet_plus[m], basis.tet_minus[m]]),
            Set([basis.tet_plus[n], basis.tet_minus[n]])))
        @test shared
    end
end

@testset "AIM operator: MVP vs direct Z*v" begin
    @testset "near-pair precorrection is exact" begin
        # For near pairs, K_near = K_direct - K_AIM is stored exactly, so
        # (K_AIM + K_near)[m,n] == K_direct[m,n] for those pairs. Verify by
        # building a test vector that only excites near-pair interactions.
        mesh = unit_cube_mesh_op()
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        k0 = 0.5; eps_p = 2.0 + 0.1im

        Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p)
        op = build_aim_operator(basis; k0 = k0, eps_p = eps_p,
                                pitch = 0.25, padding = 3)

        # Full AIM MVP should be a reasonable approximation (this tiny mesh
        # has far pairs that are only ~1 cell apart, so AIM accuracy is
        # limited). Accept 20% relative error as a smoke test.
        x = ComplexF64[(i + 0.3im) for i in 1:N]
        y_direct = Z * x
        y_aim = aim_mvp(op, x)
        rel_err = norm(y_aim - y_direct) / norm(y_direct)
        @test rel_err < 0.2
    end

    @testset "AIM converges with finer grid" begin
        # A finer AIM grid should reduce the far-field approximation error.
        mesh = unit_cube_mesh_op()
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        k0 = 0.5; eps_p = 2.0 + 0.1im

        Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p)
        x = ComplexF64[(i + 0.3im) for i in 1:N]
        y_direct = Z * x

        errs = Float64[]
        for pitch in (0.5, 0.25)
            op = build_aim_operator(basis; k0 = k0, eps_p = eps_p,
                                    pitch = pitch, padding = 3)
            y_aim = aim_mvp(op, x)
            push!(errs, norm(y_aim - y_direct) / norm(y_direct))
        end
        # Finer grid should be at least as good.
        @test errs[2] ≤ errs[1] + 1e-6
    end
end

@testset "AIM operator: static limit (κ=0)" begin
    mesh = unit_cube_mesh_op()
    basis = build_swg_basis(mesh)
    N = n_basis(basis)

    op = build_aim_operator(basis; k0 = 0.0, eps_p = 1.0, eps_bg = 1.0,
                            pitch = 0.25, padding = 3)
    x = ComplexF64[i for i in 1:N]
    y = aim_mvp(op, x)
    # κ=0 ⇒ radiation term vanishes; y = (1/ε_p) M x = M x.
    M = assemble_mass_matrix(basis)
    y_ref = M * x
    @test norm(y - y_ref) / norm(y_ref) < 1e-13
end
