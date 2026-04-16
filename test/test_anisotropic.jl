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
end
