using Test
using Random
using LinearAlgebra
using BlockVIEM
using BlockVIEM: Vec3
using StaticArrays
import Gmsh: gmsh

include(joinpath(@__DIR__, "mie_reference.jl"))

if !@isdefined(generate_sphere_mesh)
    function generate_sphere_mesh(radius::Float64, lc::Float64)
        path = joinpath(tempdir(), "sphere_blk_$(lc).msh")
        gmsh.initialize()
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("sphere_blk")
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
end

@testset "block Krylov on dense random systems" begin
    Random.seed!(20260413)
    N, L = 80, 5
    A = randn(ComplexF64, N, N) + 6I
    B = randn(ComplexF64, N, L)
    Xref = A \ B

    r1 = block_bicgstab(A, B; tol = 1e-10, maxiter = 400)
    @test r1.converged
    @test norm(r1.X - Xref) / norm(Xref) < 1e-8

    r2 = block_gmres(A, B; tol = 1e-12, maxiter = N)
    @test r2.converged
    @test norm(r2.X - Xref) / norm(Xref) < 1e-9
end

@testset "solve_cas_v2_orientations: AIM block Krylov vs dense" begin
    radius = 0.5
    wl_0 = 4.0
    m_m = 1.0
    m_p = 1.5 + 0.01im
    eps_p = m_p^2; eps_bg = m_m^2
    k0 = 2π * m_m / wl_0

    path = generate_sphere_mesh(radius, 0.30)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = true)

    euler_list = [
        (0.0, 0.0, 0.0),
        (0.3, 0.7, 0.5),
        (1.1, 1.9, -0.4),
        (0.0, π/2, 0.0),
    ]

    ref = solve_cas_v2_orientations(basis, euler_list;
                                    k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                    duffy_rule = duffy_reference_rule(7),
                                    method = :dense)

    # AIM pitch matches benchmarks/rt0/v9 — ~4% MVP error vs dense, same as
    # test_half_swg_aim.jl tolerance. AIM-vs-dense agreement is set by the
    # AIM discretization, not by the block Krylov convergence.
    h_bar = mean_edge_length(mesh)
    pitch = 0.5 * h_bar

    local out_bicgstab::Vector{CASv2Result}
    for method in (:aim_bicgstab, :aim_gmres)
        out = solve_cas_v2_orientations(basis, euler_list;
                                        k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                        duffy_rule = duffy_reference_rule(7),
                                        method = method, pitch = pitch,
                                        padding = 4,
                                        tol = 1e-8, maxiter = 400)
        @test length(out) == length(ref)
        # AIM block Krylov vs dense LU: agreement is capped by AIM's
        # ~5% intrinsic MVP approximation error at this mesh resolution.
        for i in eachindex(ref)
            @test isapprox(out[i].S_fw_mean, ref[i].S_fw_mean; rtol = 0.05)
            @test isapprox(out[i].S_bk, ref[i].S_bk; rtol = 0.05)
        end
        # Rotation invariance on the sphere: all orientations agree.
        S_fw_mean_0 = out[1].S_fw_mean
        S_bk_0 = out[1].S_bk
        for r in out[2:end]
            @test isapprox(r.S_fw_mean, S_fw_mean_0; rtol = 0.02)
            @test isapprox(r.S_bk, S_bk_0; rtol = 0.02)
        end
        if method === :aim_bicgstab
            out_bicgstab = out
        else
            # Block BiCGSTAB and block GMRES solve the same AIM system to
            # the same tolerance, so their CAS observables must match
            # to well below the AIM discretization error.
            for i in eachindex(out_bicgstab)
                @test isapprox(out[i].S_fw_mean, out_bicgstab[i].S_fw_mean; rtol = 1e-5)
                @test isapprox(out[i].S_bk, out_bicgstab[i].S_bk; rtol = 1e-5)
            end
        end
    end
end
