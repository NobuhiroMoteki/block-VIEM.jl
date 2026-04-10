using Test
using StaticArrays
using LinearAlgebra: norm, dot
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

include(joinpath(@__DIR__, "mie_reference.jl"))

function generate_sphere_mesh(radius::Float64, lc::Float64)
    path = joinpath(tempdir(), "sphere_validation_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere_val")
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

@testset "Mie reference self-consistency" begin
    @testset "Rayleigh limit: C_sca ∝ r⁶" begin
        m_p = 1.5 + 0.01im
        wl = 100.0
        r1, r2 = 0.1, 0.2
        mie1 = mie_cross_sections(; wl_0 = wl, m_m = 1.0, r_p = r1, m_p = m_p)
        mie2 = mie_cross_sections(; wl_0 = wl, m_m = 1.0, r_p = r2, m_p = m_p)
        @test isapprox(mie2.C_sca / mie1.C_sca, (r2 / r1)^6; rtol = 0.05)
    end

    @testset "non-absorbing: Q_abs ≈ 0" begin
        mie = mie_cross_sections(; wl_0 = 10.0, m_m = 1.0,
                                   r_p = 1.0, m_p = 1.5 + 0.0im)
        @test abs(mie.Q_abs) < 1e-12
    end

    @testset "lossy sphere: all Q > 0" begin
        mie = mie_cross_sections(; wl_0 = 10.0, m_m = 1.0,
                                   r_p = 1.0, m_p = 1.5 + 0.01im)
        @test mie.Q_ext > 0
        @test mie.Q_abs > 0
        @test mie.Q_sca > 0
    end
end

@testset "VIEM vs Mie: absorption cross section" begin
    # Physical parameters
    radius = 1.0
    wl_0 = 10.0
    m_m = 1.0
    m_p = 1.5 + 0.01im
    eps_bg = m_m^2
    eps_p = m_p^2
    k0 = 2π * m_m / wl_0

    # Mie reference
    mie = mie_cross_sections(; wl_0 = wl_0, m_m = m_m, r_p = radius, m_p = m_p)

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

    @testset "coarse mesh (lc=0.7): C_abs positive and correct order" begin
        path = generate_sphere_mesh(radius, 0.7)
        mesh = read_msh(path)
        basis = build_swg_basis(mesh)
        dr = duffy_reference_rule(7)
        Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p,
                                      eps_bg = eps_bg, duffy_rule = dr,
                                      symmetrize = true)
        k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
        b = project_plane_wave(basis; k_hat = k_hat, E0 = E0, k_bg = k_bg)
        D = Z \ b
        residual = norm(Z * D - b) / norm(b)
        scat = compute_scattering(basis, D;
                                  k_hat = k_hat, E0 = E0,
                                  k0 = k0, eps_p = eps_p, eps_bg = eps_bg)
        @test residual < 1e-8
        @test scat.C_abs > 0
        @test scat.C_abs / mie.C_abs > 0.3
        @test scat.C_abs / mie.C_abs < 3.0
        @info "VIEM coarse" N=n_basis(basis) C_abs_viem=scat.C_abs C_abs_mie=mie.C_abs
    end

    @testset "mesh refinement improves C_abs" begin
        errs = Float64[]
        # Use higher-order Duffy (order 7) for better near-singular accuracy.
        dr = duffy_reference_rule(7)
        for lc in (0.7, 0.5)
            path = generate_sphere_mesh(radius, lc)
            mesh = read_msh(path)
            basis = build_swg_basis(mesh)
            Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p,
                                          eps_bg = eps_bg, duffy_rule = dr,
                                          symmetrize = true)
            k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
            b = project_plane_wave(basis; k_hat = k_hat, E0 = E0, k_bg = k_bg)
            D = Z \ b
            scat = compute_scattering(basis, D;
                                      k_hat = k_hat, E0 = E0,
                                      k0 = k0, eps_p = eps_p, eps_bg = eps_bg)
            push!(errs, abs(scat.C_abs - mie.C_abs) / mie.C_abs)
            @info "VIEM refinement" lc N=n_basis(basis) C_abs=scat.C_abs rel_err=errs[end]
        end
        # Finer mesh should give better or equal C_abs.
        @test errs[2] ≤ errs[1] + 0.01
    end
end
