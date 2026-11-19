using Test
using BlockVIEM
using BlockVIEM: Vec3, _aniso_params, _eps_p_vec
using StaticArrays
using LinearAlgebra: norm

@testset "Anisotropic ε_p" begin

    # ── helpers ──────────────────────────────────────────────────────────
    @testset "_eps_p_vec / _aniso_params" begin
        # scalar → 3-vector
        v = _eps_p_vec(2.25 + 0.1im)
        @test v == SVector{3,ComplexF64}(2.25+0.1im, 2.25+0.1im, 2.25+0.1im)

        # SVector passthrough
        sv = SVector{3,ComplexF64}(2.0, 2.5, 3.0)
        @test _eps_p_vec(sv) == sv

        # plain vector
        @test _eps_p_vec([2.0+0im, 2.5+0im, 3.0+0im]) == sv

        # _aniso_params
        ev, eb, inv_v, kv, ka = _aniso_params(2.25, 1.0)
        @test ev ≈ SVector{3,ComplexF64}(2.25, 2.25, 2.25)
        @test eb ≈ 1.0
        @test inv_v[1] ≈ 1 / 2.25
        @test kv[1] ≈ (2.25 - 1.0) / 2.25
        @test ka ≈ kv[1]  # isotropic

        # anisotropic
        ev2, _, _, kv2, ka2 = _aniso_params([2.0+0im, 3.0+0im, 4.0+0im], 1.0)
        @test kv2[1] ≈ (2.0 - 1.0) / 2.0
        @test kv2[2] ≈ (3.0 - 1.0) / 3.0
        @test kv2[3] ≈ (4.0 - 1.0) / 4.0
        @test ka2 ≈ (kv2[1] + kv2[2] + kv2[3]) / 3
    end

    # ── isotropic vector == scalar equivalence ───────────────────────────
    @testset "vector m_p = [m,m,m] matches scalar m_p = m" begin
        # Build a small sphere mesh
        import Gmsh: gmsh
        msh_path = tempname() * ".msh"
        gmsh.initialize()
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("aniso_test")
            sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 0.2)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(3, [sph], 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.12)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.12)
            gmsh.model.mesh.generate(3)
            gmsh.write(msh_path)
        finally
            gmsh.finalize()
        end
        mesh = read_msh(msh_path)
        basis = build_swg_basis(mesh; include_boundary_faces=true)

        wl_0 = 0.638
        m_m  = 1.0
        m_p_scalar = 1.5 + 0.01im
        m_p_vector = [m_p_scalar, m_p_scalar, m_p_scalar]

        euler_list = [(0.0, π/4, 0.0)]

        # Scalar solve
        res_s = solve_cas_v2_orientations(basis, euler_list;
                    wl_0=wl_0, m_m=m_m, m_p=m_p_scalar, method=:dense)

        # Vector solve (isotropic)
        res_v = solve_cas_v2_orientations(basis, euler_list;
                    wl_0=wl_0, m_m=m_m, m_p=m_p_vector, method=:dense)

        # Results must be identical (same codepath after _eps_p_vec promotion)
        @test res_s[1].S_fw_theta ≈ res_v[1].S_fw_theta rtol=1e-12
        @test res_s[1].S_fw_phi   ≈ res_v[1].S_fw_phi   rtol=1e-12
        @test res_s[1].S_bk       ≈ res_v[1].S_bk        rtol=1e-12
    end

    # ── anisotropic solve produces plausible results ─────────────────────
    @testset "anisotropic solve runs and gives non-degenerate S" begin
        import Gmsh: gmsh
        msh_path = tempname() * ".msh"
        gmsh.initialize()
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("aniso_test2")
            sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 0.2)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(3, [sph], 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.12)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.12)
            gmsh.model.mesh.generate(3)
            gmsh.write(msh_path)
        finally
            gmsh.finalize()
        end
        mesh = read_msh(msh_path)
        basis = build_swg_basis(mesh; include_boundary_faces=true)

        wl_0 = 0.638
        m_m  = 1.0
        # Birefringent particle: m_p_x ≠ m_p_y ≠ m_p_z
        m_p_aniso = [1.6 + 0.0im, 1.5 + 0.0im, 1.4 + 0.0im]

        euler_list = [(0.0, π/4, 0.0), (0.0, π/2, 0.0)]

        res = solve_cas_v2_orientations(basis, euler_list;
                    wl_0=wl_0, m_m=m_m, m_p=m_p_aniso, method=:dense)

        # Should produce finite, non-zero amplitudes
        for r in res
            @test isfinite(r.S_fw_theta)
            @test isfinite(r.S_fw_phi)
            @test abs(r.S_fw_theta) > 0
            @test abs(r.S_fw_phi)   > 0
        end

        # For an anisotropic sphere, S_theta ≠ S_phi at β=π/2 in general
        # (unlike isotropic sphere where they're equal)
        @test !isapprox(res[2].S_fw_theta, res[2].S_fw_phi; rtol=1e-3)
    end

    # ── anisotropic breaks spheroid alpha-symmetry ───────────────────────
    @testset "anisotropic sphere: S(alpha) ≠ S(0) for alpha ≠ 0" begin
        import Gmsh: gmsh
        msh_path = tempname() * ".msh"
        gmsh.initialize()
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("aniso_alpha")
            sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 0.2)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(3, [sph], 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.12)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.12)
            gmsh.model.mesh.generate(3)
            gmsh.write(msh_path)
        finally
            gmsh.finalize()
        end
        mesh = read_msh(msh_path)
        basis = build_swg_basis(mesh; include_boundary_faces=true)

        m_p_aniso = [1.6 + 0.0im, 1.5 + 0.0im, 1.4 + 0.0im]

        # Two orientations: alpha=0 and alpha=π/3, same beta
        euler_list = [(0.0, π/3, 0.0), (π/3, π/3, 0.0)]

        res = solve_cas_v2_orientations(basis, euler_list;
                    wl_0=0.638, m_m=1.0, m_p=m_p_aniso, method=:dense)

        # For anisotropic particle, rotating in alpha changes the result
        # (unlike isotropic sphere where S(alpha) = const)
        @test !isapprox(res[1].S_fw_theta, res[2].S_fw_theta; rtol=1e-3)
    end

    # ── AIM path: component-weighted mass term ───────────────────────────
    # The mass term of an anisotropic particle is Σ_α (1/ε_α) M_α. Until
    # v0.8.0 the AIM operator used mean(1/ε_α) M instead, which differs from
    # the dense matrix by about 10 % for the tensors below. The dense path was
    # always component-weighted.
    @testset "AIM operator matches the dense matrix for anisotropic ε_p" begin
        mesh = read_msh(joinpath(@__DIR__, "meshes", "sphere.msh"))
        basis = build_swg_basis(mesh; include_boundary_faces=true)
        N = n_basis(basis)
        k0 = 1.2
        pitch = 0.5 * mean_edge_length(mesh)
        relerr(a, b) = norm(a .- b) / norm(b)
        X = ComplexF64[sin(0.7i + 1.3j) + im * cos(0.4i - 0.9j) for i in 1:N, j in 1:3]

        # Weighted mass matrix: equal weights reduce to the unweighted one,
        # and the three components add up to it.
        M = assemble_mass_matrix(basis)
        w = SVector{3,ComplexF64}(0.4 + 0.1im, 0.4 + 0.1im, 0.4 + 0.1im)
        @test relerr(assemble_weighted_mass_matrix(basis, w), (0.4 + 0.1im) .* M) < 1e-13
        e(a) = SVector{3,ComplexF64}(a == 1, a == 2, a == 3)
        Mc = [assemble_weighted_mass_matrix(basis, e(a)) for a in 1:3]
        @test relerr(Mc[1] + Mc[2] + Mc[3], M) < 1e-13
        @test all(A -> relerr(A, transpose(A)) < 1e-13, Mc)
        @test relerr(Mc[1], Mc[3]) > 0.1          # the components differ

        err_iso = 0.0
        for (iso, eps) in ((true,  ComplexF64[2.25, 2.25, 2.25]),
                           (false, ComplexF64[2.25, 2.25, 4.0]),
                           (false, ComplexF64[2.25 + 0.1im, 3.0 + 0.3im, 4.0 + 0.05im]))
            Z = assemble_impedance_matrix(basis; k0=k0, eps_p=eps, eps_bg=1.0,
                                          symmetrize=true)
            op = build_aim_operator(basis; k0=k0, eps_p=eps, eps_bg=1.0,
                                    pitch=pitch, padding=4)
            @test (op.mass_weighted === nothing) == iso
            err = relerr(aim_mvp(op, X), Z * X)
            iso && (err_iso = err)
            # AIM approximation error only, the same size for all three tensors.
            @test err < 5e-3
            @test err < 3 * err_iso
            @test relerr(aim_mvp(op, X[:, 2]), Z * X[:, 2]) < 5e-3
            @test aim_mvp(op, X)[:, 1] ≈ aim_mvp(op, X[:, 1]) rtol = 1e-12
            if !iso
                inv_eps = 1 ./ eps
                w_eps = SVector{3,ComplexF64}(inv_eps...)
                @test relerr(op.mass_weighted,
                             assemble_weighted_mass_matrix(basis, w_eps)) < 1e-14
                # The former mean-weight mass term is far from the dense one.
                @test relerr((sum(inv_eps) / 3) .* (M * X), op.mass_weighted * X) > 0.05
            end
        end
    end

    @testset "anisotropic observables: :aim_gmres against :dense" begin
        mesh = read_msh(joinpath(@__DIR__, "meshes", "sphere.msh"))
        basis = build_swg_basis(mesh; include_boundary_faces=true)
        euler_list = [(0.0, 0.0, 0.0), (0.3, 1.1, 0.0), (2.0, 2.4, 0.7)]
        fields(r) = (r.S_fw_theta, r.S_fw_phi, r.S_bk)
        for m_p in ([1.5 + 0.0im, 1.5 + 0.0im, 2.0 + 0.0im],
                    [1.5 + 0.02im, 1.73 + 0.0im, 2.0 + 0.05im])
            dense = solve_cas_v2_orientations(basis, euler_list; wl_0=2π/1.2, m_m=1.0,
                                              m_p=m_p, method=:dense)
            aim = solve_cas_v2_orientations(basis, euler_list; wl_0=2π/1.2, m_m=1.0,
                                            m_p=m_p, method=:aim_gmres,
                                            tol=1e-8, maxiter=300)
            for i in eachindex(euler_list)
                scale = maximum(abs, fields(dense[i]))
                diff = maximum(abs.(collect(fields(aim[i])) .- collect(fields(dense[i]))))
                # 0.25 and 0.16 with the former mass term.
                @test diff / scale < 0.02
            end
        end
    end
end
