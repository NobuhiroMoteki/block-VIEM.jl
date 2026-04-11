using Test
using StaticArrays
using LinearAlgebra: norm, dot
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

include(joinpath(@__DIR__, "mie_reference.jl"))

function generate_sphere_mesh(radius::Float64, lc::Float64)
    path = joinpath(tempdir(), "sphere_rt1_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere_rt1")
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

@testset "RT1 vs Mie: absorption cross section" begin
    radius = 1.0
    wl_0 = 10.0
    m_m = 1.0
    m_p = 1.5 + 0.01im
    eps_bg = m_m^2
    eps_p = m_p^2
    k0 = 2π * m_m / wl_0

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

    # Use degree-7 quadrature for RT1 (integrand degree 4 for mass, ≥5 for radiation)
    outer_rule = TET_QUAD_64PT
    dr = duffy_reference_rule(7)

    @testset "RT1 mesh refinement: C_abs converges to Mie" begin
        errs_rt1 = Float64[]
        ndofs_rt1 = Int[]

        for lc in (0.7, 0.5)
            path = generate_sphere_mesh(radius, lc)
            mesh = read_msh(path)
            basis = build_rt1_basis(mesh)
            V_mesh = total_volume(mesh)
            r_ve = (3V_mesh / (4π))^(1 / 3)
            mie_ve = mie_cross_sections(; wl_0 = wl_0, m_m = m_m, r_p = r_ve, m_p = m_p)

            N = n_basis(basis)

            Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p,
                                          eps_bg = eps_bg,
                                          outer_rule = outer_rule,
                                          duffy_rule = dr,
                                          symmetrize = true)
            k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
            b = project_plane_wave(basis; k_hat = k_hat, E0 = E0, k_bg = k_bg,
                                   rule = outer_rule)
            D = Z \ b
            residual = norm(Z * D - b) / norm(b)

            scat = compute_scattering(basis, D;
                                      k_hat = k_hat, E0 = E0,
                                      k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                      rule = outer_rule)

            rel_err = abs(scat.C_abs - mie_ve.C_abs) / mie_ve.C_abs
            push!(errs_rt1, rel_err)
            push!(ndofs_rt1, N)

            @test residual < 1e-6
            @test scat.C_abs > 0
            @info "RT1 VIEM refinement" lc N r_ve C_abs_viem=scat.C_abs C_abs_mie=mie_ve.C_abs rel_err
        end

        # Finer mesh should give better or comparable result
        @test errs_rt1[2] ≤ errs_rt1[1] + 0.01
        # Coarse mesh should be within 50% of Mie
        @test errs_rt1[1] < 0.5
    end

    @testset "RT1 vs RT0 comparison at same mesh" begin
        lc = 0.5
        path = generate_sphere_mesh(radius, lc)
        mesh = read_msh(path)
        V_mesh = total_volume(mesh)
        r_ve = (3V_mesh / (4π))^(1 / 3)
        mie_ve = mie_cross_sections(; wl_0 = wl_0, m_m = m_m, r_p = r_ve, m_p = m_p)

        k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))

        # --- RT0 ---
        basis0 = build_swg_basis(mesh)
        dr0 = duffy_reference_rule(5)
        Z0 = assemble_impedance_matrix(basis0; k0 = k0, eps_p = eps_p,
                                       eps_bg = eps_bg, duffy_rule = dr0,
                                       symmetrize = true)
        b0 = project_plane_wave(basis0; k_hat = k_hat, E0 = E0, k_bg = k_bg)
        D0 = Z0 \ b0
        scat0 = compute_scattering(basis0, D0;
                                   k_hat = k_hat, E0 = E0,
                                   k0 = k0, eps_p = eps_p, eps_bg = eps_bg)
        err_rt0 = abs(scat0.C_abs - mie_ve.C_abs) / mie_ve.C_abs

        # --- RT1 ---
        basis1 = build_rt1_basis(mesh)
        Z1 = assemble_impedance_matrix(basis1; k0 = k0, eps_p = eps_p,
                                       eps_bg = eps_bg,
                                       outer_rule = outer_rule,
                                       duffy_rule = dr,
                                       symmetrize = true)
        b1 = project_plane_wave(basis1; k_hat = k_hat, E0 = E0, k_bg = k_bg,
                                rule = outer_rule)
        D1 = Z1 \ b1
        scat1 = compute_scattering(basis1, D1;
                                   k_hat = k_hat, E0 = E0,
                                   k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                   rule = outer_rule)
        err_rt1 = abs(scat1.C_abs - mie_ve.C_abs) / mie_ve.C_abs

        @info "RT0 vs RT1 at lc=$lc" N_RT0=n_basis(basis0) N_RT1=n_basis(basis1) err_RT0=err_rt0 err_RT1=err_rt1 C_abs_mie=mie_ve.C_abs

        # RT1 should be at least as accurate as RT0 on the same mesh
        # (and ideally significantly better)
        @test err_rt1 < err_rt0 + 0.05
    end
end
