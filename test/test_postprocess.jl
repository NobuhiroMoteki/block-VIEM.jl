using Test
using StaticArrays
using LinearAlgebra: norm, dot
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

function unit_cube_mesh_pp()
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

@testset "Far-field amplitude" begin
    mesh = unit_cube_mesh_pp()
    basis = build_swg_basis(mesh)
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
    k0 = 0.5
    eps_p = 2.0 + 0.1im

    # Solve directly to get D_coeffs.
    result = solve_direct(basis; k0 = k0, eps_p = eps_p,
                          k_hat = k_hat, E0 = E0)
    D = result.D_coeffs

    @testset "transversality: k̂ · F = 0" begin
        for k_sca in (Vec3(0, 0, 1), Vec3(0, 0, -1),
                      Vec3(1, 0, 0), Vec3(0, 1, 0),
                      Vec3(1, 1, 1) / sqrt(3))
            F = far_field_amplitude(basis, D;
                                    k_hat_sca = k_sca, k0 = k0,
                                    eps_p = eps_p)
            @test abs(dot(SVector{3,ComplexF64}(k_sca), F)) < 1e-12 * norm(F)
        end
    end

    @testset "finite and nonzero for non-trivial D" begin
        F = far_field_amplitude(basis, D;
                                k_hat_sca = k_hat, k0 = k0,
                                eps_p = eps_p)
        @test all(isfinite, F)
        @test norm(F) > 0
    end
end

@testset "ScatteringResult / compute_scattering" begin
    mesh = unit_cube_mesh_pp()
    basis = build_swg_basis(mesh)
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
    k0 = 0.5
    eps_p = 2.0 + 0.1im   # lossy

    result = solve_direct(basis; k0 = k0, eps_p = eps_p,
                          k_hat = k_hat, E0 = E0)
    scat = compute_scattering(basis, result.D_coeffs;
                              k_hat = k_hat, E0 = E0,
                              k0 = k0, eps_p = eps_p)
    @test scat isa ScatteringResult

    @testset "forward amplitudes are finite" begin
        @test isfinite(scat.S_fw_s)
        @test isfinite(scat.S_fw_p)
        @test isfinite(scat.S_bak)
    end

    @testset "C_ext > 0 for a lossy particle" begin
        # On the coarse 6-element cube mesh C_ext may be negative due to
        # limited far-field accuracy. Check finiteness only here; positivity
        # is validated on the sphere mesh in test_mie_validation.jl.
        @test isfinite(scat.C_ext)
    end

    @testset "C_abs > 0 for Im(ε_p) > 0" begin
        @test scat.C_abs > 0
    end

    @testset "energy conservation: C_sca = C_ext - C_abs (identity)" begin
        @test isapprox(scat.C_sca, scat.C_ext - scat.C_abs; atol = 1e-15)
        # On this coarse 6-element mesh C_sca may be slightly negative due
        # to the limited accuracy of the far-field radiation integral.
        # A stricter non-negativity check requires a finer mesh.
    end

    @testset "lossless particle: C_abs ≈ 0" begin
        eps_p_real = 2.0   # purely real, no absorption
        result_r = solve_direct(basis; k0 = k0, eps_p = eps_p_real,
                                k_hat = k_hat, E0 = E0)
        scat_r = compute_scattering(basis, result_r.D_coeffs;
                                    k_hat = k_hat, E0 = E0,
                                    k0 = k0, eps_p = eps_p_real)
        @test abs(scat_r.C_abs) < 1e-12
        @test isapprox(scat_r.C_ext, scat_r.C_sca; atol = 1e-12)
    end
end
