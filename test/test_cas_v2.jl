using Test
using StaticArrays
using LinearAlgebra: norm, dot
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

include(joinpath(@__DIR__, "mie_reference.jl"))

# Reuse the sphere mesh helper from test_mie_validation if not already defined.
if !@isdefined(generate_sphere_mesh)
    function generate_sphere_mesh(radius::Float64, lc::Float64)
        path = joinpath(tempdir(), "sphere_cas_$(lc).msh")
        gmsh.initialize()
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("sphere_cas")
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

@testset "CAS-v2 orientation geometry" begin
    @testset "identity orientation" begin
        ori = cas_orientation(0.0, 0.0, 0.0)
        @test ori.u_inc       ≈ Vec3(0, 0, 1)
        @test ori.theta_inc   ≈ Vec3(1, 0, 0)
        @test ori.phi_inc     ≈ Vec3(0, 1, 0)
        @test ori.u_sca_fw    ≈ Vec3(0, 0, 1)
        @test ori.theta_sca_fw ≈ Vec3(1, 0, 0)
        @test ori.phi_sca_fw  ≈ Vec3(0, 1, 0)
        @test ori.u_sca_bk    ≈ Vec3(0, 0, -1)
        @test ori.theta_sca_bk ≈ Vec3(-1, 0, 0)
        @test ori.phi_sca_bk  ≈ Vec3(0, 1, 0)
        # Right-handed: theta × phi = u
        for (t, p, u) in ((ori.theta_inc, ori.phi_inc, ori.u_inc),
                          (ori.theta_sca_fw, ori.phi_sca_fw, ori.u_sca_fw),
                          (ori.theta_sca_bk, ori.phi_sca_bk, ori.u_sca_bk))
            cr = Vec3(t[2]*p[3] - t[3]*p[2],
                      t[3]*p[1] - t[1]*p[3],
                      t[1]*p[2] - t[2]*p[1])
            @test cr ≈ u atol = 1e-12
        end
    end

    @testset "ZYZ rotation: u_inc tilts to particle frame" begin
        # beta = π/2, alpha = γ = 0  → particle frame z-axis is rotated
        # by Ry(π/2) about lab y. Inverse maps lab +z into particle frame.
        ori = cas_orientation(0.0, π/2, 0.0)
        # R_lp = Ry(π/2) sends particle z to lab x ⇒ R_pl sends lab z to particle -x.
        # Numerical check via dot products:
        @test isapprox(norm(ori.u_inc), 1.0; atol=1e-12)
        @test isapprox(ori.u_inc[1], -1.0; atol=1e-12)
        @test isapprox(ori.u_inc[3],  0.0; atol=1e-12)
    end

    @testset "circular polarization vector" begin
        ori = cas_orientation(0.0, 0.0, 0.0)
        e0 = ori.e0_inc
        # |e0|² = 1
        @test real(e0[1]*conj(e0[1]) + e0[2]*conj(e0[2]) + e0[3]*conj(e0[3])) ≈ 1.0 atol = 1e-12
        # e0 ⊥ u_inc
        @test abs(e0[1]*ori.u_inc[1] + e0[2]*ori.u_inc[2] + e0[3]*ori.u_inc[3]) < 1e-12
    end
end

@testset "CAS-v2 vs Mie sphere (S_fw_mean, S_bk)" begin
    radius = 0.5
    wl_0 = 4.0
    m_m = 1.0
    m_p = 1.5 + 0.01im
    eps_bg = m_m^2
    eps_p = m_p^2
    k0 = 2π * m_m / wl_0

    # Build sphere mesh and solve for D-coefficients (dense path).
    lc = 0.18
    path = generate_sphere_mesh(radius, lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    V_mesh = total_volume(mesh)
    r_ve = (3 * V_mesh / (4π))^(1/3)

    dr = duffy_reference_rule(7)
    Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p,
                                  eps_bg = eps_bg, duffy_rule = dr,
                                  symmetrize = true)

    # Reference orientation: alpha=beta=gamma=0
    ori = cas_orientation(0.0, 0.0, 0.0)

    # Build the RHS using the same circular polarization that CAS-v2 expects.
    # k0 is already the background-medium wavenumber (= 2π·m_m/wl_0), so
    # k_bg = k0.
    k_bg = ComplexF64(k0)
    b = project_plane_wave(basis; k_hat = ori.u_inc, E0 = ori.e0_inc, k_bg = k_bg)
    D = Z \ b
    @test norm(Z * D - b) / norm(b) < 1e-8

    cas = compute_cas_observables(basis, D;
                                  orientation = ori,
                                  k0 = k0, eps_p = eps_p, eps_bg = eps_bg)

    mie = mie_cas_observables(; wl_0 = wl_0, m_m = m_m, r_p = r_ve, m_p = m_p)

    @info "CAS-v2 vs Mie (sphere)" lc N=n_basis(basis) r_ve x=mie.x
    @info "  Forward (PCAS)" S_fw_mean_VIEM=cas.S_fw_mean S_fw_mean_Mie=mie.S_fw_mean
    @info "    theta-channel" S_fw_theta=cas.S_fw_theta
    @info "    phi-channel"   S_fw_phi=cas.S_fw_phi
    @info "  Backward (OCBS)" S_bk_VIEM=cas.S_bk S_bk_Mie=mie.S_bk

    # For a sphere with circular illumination at θ=0, S_θ ≈ S_φ.
    @test isapprox(cas.S_fw_theta, cas.S_fw_phi; rtol = 0.01)

    # After the 2026-04-13 physics-convention switch in green.jl/incident.jl/
    # postprocess.jl, both Re and Im of S_fw_mean converge with mesh refinement.
    re_err_fw = abs(real(cas.S_fw_mean) - real(mie.S_fw_mean)) / abs(real(mie.S_fw_mean))
    im_err_fw = abs(imag(cas.S_fw_mean) - imag(mie.S_fw_mean)) / abs(imag(mie.S_fw_mean))
    @info "  S_fw_mean relative error" re_err_fw im_err_fw
    @test re_err_fw < 0.01
    @test im_err_fw < 0.01

    re_err_bk = abs(real(cas.S_bk) - real(mie.S_bk)) / abs(real(mie.S_bk))
    @info "  S_bk Re relative error" re_err_bk
    @test re_err_bk < 0.05

    # Magnitude agreement (sign-convention-independent).
    @test isapprox(abs(cas.S_fw_mean), abs(mie.S_fw_mean); rtol = 0.05)
    @test isapprox(abs(cas.S_bk), abs(mie.S_bk); rtol = 0.05)
end

@testset "CAS-v2 sphere refinement: Im(S_fw_mean) → Mie" begin
    # Demonstrate that Im(S_fw_mean) converges to Mie as the mesh is refined.
    radius = 0.5
    wl_0 = 4.0
    m_m = 1.0
    m_p = 1.5 + 0.01im
    eps_p = m_p^2; eps_bg = m_m^2
    k0 = 2π * m_m / wl_0

    dr = duffy_reference_rule(7)
    ori = cas_orientation(0.0, 0.0, 0.0)

    im_errs = Float64[]
    for lc in (0.30, 0.18, 0.12)
        path = generate_sphere_mesh(radius, lc)
        mesh = read_msh(path)
        basis = build_swg_basis(mesh; include_boundary_faces = true)
        V_mesh = total_volume(mesh)
        r_ve = (3 * V_mesh / (4π))^(1/3)

        Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = eps_p,
                                      eps_bg = eps_bg, duffy_rule = dr,
                                      symmetrize = true)
        b = project_plane_wave(basis; k_hat = ori.u_inc, E0 = ori.e0_inc,
                               k_bg = ComplexF64(k0))
        D = Z \ b
        cas = compute_cas_observables(basis, D; orientation = ori,
                                       k0 = k0, eps_p = eps_p, eps_bg = eps_bg)
        mie = mie_cas_observables(; wl_0 = wl_0, m_m = m_m, r_p = r_ve, m_p = m_p)
        im_err = abs(imag(cas.S_fw_mean) - imag(mie.S_fw_mean)) / abs(imag(mie.S_fw_mean))
        push!(im_errs, im_err)
        @info "refinement" lc N=n_basis(basis) Re_VIEM=real(cas.S_fw_mean) Re_Mie=real(mie.S_fw_mean) Im_VIEM=imag(cas.S_fw_mean) Im_Mie=imag(mie.S_fw_mean) im_err
    end
    @test im_errs[end] < im_errs[1]
end

@testset "solve_cas_v2_orientations: sphere rotation invariance" begin
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
        (-0.5, 2.3, 1.7),
    ]
    results = solve_cas_v2_orientations(basis, euler_list;
                                         k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                         duffy_rule = duffy_reference_rule(7),
                                         method = :dense)
    @test length(results) == length(euler_list)

    # Sphere is rotation-invariant: all S_fw_mean should agree across orientations.
    S_fw_mean_0 = results[1].S_fw_mean
    S_bk_0 = results[1].S_bk
    @info "sphere multi-orientation S_fw_mean" S_fw_mean_0
    # Coarse mesh (lc=0.30, ~400 DOFs) is not perfectly spherically symmetric,
    # so the rotation invariance is limited by mesh asymmetry. ~2% tolerance.
    for r in results[2:end]
        @test isapprox(r.S_fw_mean, S_fw_mean_0; rtol = 0.02)
        @test isapprox(r.S_bk, S_bk_0; rtol = 0.02)
    end

    # block-DDA_Py-compatible API (wl_0, m_m, m_p) must give identical
    # results to the raw (k0, eps_p, eps_bg) form when m_m = 1.
    results_phys = solve_cas_v2_orientations(basis, euler_list;
                                              wl_0 = wl_0, m_m = m_m, m_p = m_p,
                                              duffy_rule = duffy_reference_rule(7),
                                              method = :dense)
    for (r, rp) in zip(results, results_phys)
        @test isapprox(rp.S_fw_mean, r.S_fw_mean; rtol = 1e-12)
        @test isapprox(rp.S_bk, r.S_bk; rtol = 1e-12)
    end

    # Mixing the two input forms should error.
    @test_throws ArgumentError solve_cas_v2_orientations(
        basis, euler_list;
        wl_0 = wl_0, m_m = m_m, m_p = m_p, k0 = k0,
        duffy_rule = duffy_reference_rule(7))
end

@testset "solve_cas_v2_orientations: projection + mass reuse" begin
    # Parameter-sweep reuse path: passing a pre-built AIMProjection and
    # mass matrix to the AIM block-BiCGSTAB solver must produce the
    # same CAS-v2 observables as a fresh build at the same (k0, eps_p).
    radius = 0.5
    path = generate_sphere_mesh(radius, 0.30)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = true)

    wl_0 = 4.0; m_m = 1.0; m_p = 1.5 + 0.01im
    euler_list = [(0.0, 0.0, 0.0), (0.3, 0.7, 0.5), (0.0, π/2, 0.0)]
    pitch = 0.5 * mean_edge_length(mesh)

    grid = aim_grid(mesh; pitch = pitch, padding = 4)
    proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    mass = assemble_mass_matrix(basis)

    res_fresh = solve_cas_v2_orientations(basis, euler_list;
                                           wl_0 = wl_0, m_m = m_m, m_p = m_p,
                                           method = :aim_bicgstab,
                                           pitch = pitch, tol = 1e-8)
    res_reuse = solve_cas_v2_orientations(basis, euler_list;
                                           wl_0 = wl_0, m_m = m_m, m_p = m_p,
                                           method = :aim_bicgstab,
                                           pitch = pitch, tol = 1e-8,
                                           projection = proj, mass = mass)
    for (rf, rr) in zip(res_fresh, res_reuse)
        @test isapprox(rr.S_fw_mean, rf.S_fw_mean; rtol = 1e-8)
        @test isapprox(rr.S_bk,      rf.S_bk;      rtol = 1e-8)
    end
end

@testset "solve_cas_v2_orientations: non-unit m_m (physical API)" begin
    # Cross-check that the physical (wl_0, m_m, m_p) path gives the same
    # physics as the equivalent "vacuum-scaled" raw form when we explicitly
    # set k0 to the background-medium wavenumber. This exercises the
    # corrected `k_bg = k0` convention for m_m ≠ 1.
    radius = 0.5
    wl_0 = 4.0
    m_m = 1.33                      # water-like background
    m_p = 1.5 + 0.01im
    eps_p  = ComplexF64(m_p)^2
    eps_bg = ComplexF64(m_m)^2
    k0_bg  = 2π * m_m / wl_0        # wavenumber in background medium

    path = generate_sphere_mesh(radius, 0.30)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = true)

    euler_list = [(0.0, 0.0, 0.0), (0.3, 0.7, 0.5)]

    # Raw form — user has already pre-computed k0 in the medium.
    res_raw  = solve_cas_v2_orientations(basis, euler_list;
                                         k0 = k0_bg, eps_p = eps_p,
                                         eps_bg = eps_bg,
                                         duffy_rule = duffy_reference_rule(7),
                                         method = :dense)
    # Physical form — internally computes k0 = 2π·m_m/wl_0.
    res_phys = solve_cas_v2_orientations(basis, euler_list;
                                         wl_0 = wl_0, m_m = m_m, m_p = m_p,
                                         duffy_rule = duffy_reference_rule(7),
                                         method = :dense)
    for (rr, rp) in zip(res_raw, res_phys)
        @test isapprox(rp.S_fw_mean, rr.S_fw_mean; rtol = 1e-12)
        @test isapprox(rp.S_bk, rr.S_bk; rtol = 1e-12)
    end

    # Sanity: rotation invariance still holds for non-unit m_m.
    @test isapprox(res_phys[2].S_fw_mean, res_phys[1].S_fw_mean; rtol = 0.02)
    @test isapprox(res_phys[2].S_bk, res_phys[1].S_bk; rtol = 0.02)
end
