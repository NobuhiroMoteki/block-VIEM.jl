using Test
using Random
using LinearAlgebra
using SparseArrays
using BlockVIEM
using StaticArrays

include(joinpath(@__DIR__, "mie_reference.jl"))

# Phase 2 of the shifted refractive-index sweep:
# `solve_cas_v2_orientations_shifted` against the existing solvers, and the
# opt-in mass-matrix right preconditioning of `solve_cas_v2_orientations`.

_cas_fields(r::CASv2Result) = (r.S_fw_theta, r.S_fw_phi, r.S_fw_mean,
                               r.S_bk_theta, r.S_bk_phi, r.S_bk)

# Largest difference between two results, relative to the largest amplitude
# of the reference. Components that vanish by symmetry are thus measured
# against the size of the non-vanishing ones.
function _cas_diff(a::CASv2Result, b::CASv2Result)
    scale = maximum(abs, _cas_fields(b))
    return maximum(abs.(_cas_fields(a) .- _cas_fields(b))) / scale
end

@testset "shifted refractive-index sweep" begin
    mesh = read_msh(joinpath(@__DIR__, "meshes", "sphere.msh"))
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N = n_basis(basis)
    r_ve = cbrt(3 * total_volume(mesh) / (4π))                # volume-equivalent radius
    wl_0 = 2π / 1.2                                           # size parameter ~ 1.2
    m_m = 1.0
    k0 = 2π * m_m / wl_0
    eps_bg = ComplexF64(m_m)^2
    m_list = ComplexF64[1.5 + 0.01im, 1.75, 2.0, 2.5 + 0.2im]
    eulers = [(0.0, 0.0, 0.0), (0.3, 1.1, 0.0), (2.0, 2.4, 0.7)]
    oris = [cas_orientation(e...) for e in eulers]
    L = length(eulers)

    pitch = 0.5 * mean_edge_length(mesh)
    grid = aim_grid(mesh; pitch = pitch, padding = 4)
    proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    mass = assemble_mass_matrix(basis)

    # ── T5: the shift formulation against dense direct solves ─────────────
    # The dense operator has no AIM error, so this checks the formulation
    # itself: K0 from one dense Z, the shifts, the back-substitution and the
    # observables, against an LU solve of a separately assembled Z per index.
    @testset "T5: dense operator, shifted vs direct solve" begin
        Zs = [assemble_impedance_matrix(basis; k0 = k0, eps_p = m^2, eps_bg = eps_bg,
                                        symmetrize = true) for m in m_list]
        Md = Matrix(mass)
        kap = [(m^2 - eps_bg) / m^2 for m in m_list]
        tau = [eps_bg / (m^2 - eps_bg) for m in m_list]
        K0s = [(eps_bg / kap[s]) .* (Md ./ m_list[s]^2 .- Zs[s]) for s in eachindex(m_list)]
        # kappa factors out of the radiation operator.
        for s in 2:length(m_list)
            @test norm(K0s[s] .- K0s[1]) / norm(K0s[1]) < 1e-10
        end

        B = Matrix{ComplexF64}(undef, N, L)                   # shape: (N, L)
        for (j, o) in enumerate(oris)
            B[:, j] = project_plane_wave(basis; k_hat = o.u_inc, E0 = o.e0_inc, k_bg = k0)
        end
        C = K0s[end] / Md                                     # K0 M^{-1}, shape: (N, N)
        r = block_gmres_shifted(C, B, tau; tol = 1e-10, maxiter = N ÷ L)
        @test all(r.converged)
        for s in eachindex(m_list)
            D = (eps_bg / kap[s]) .* (Md \ r.X[s])
            Dref = Zs[s] \ B
            @test norm(D .- Dref) / norm(Dref) < 1e-7
            for i in 1:L
                obs(Dc) = compute_cas_observables(basis, Dc; orientation = oris[i],
                                                  k0 = k0, eps_p = m_list[s]^2,
                                                  eps_bg = eps_bg)
                @test _cas_diff(obs(view(D, :, i)), obs(view(Dref, :, i))) < 1e-6
            end
            # Cross sections from the same solution block.
            sc(Dc) = compute_scattering(basis, Dc; k_hat = oris[1].u_inc, E0 = oris[1].e0_inc,
                                        k0 = k0, eps_p = m_list[s]^2, eps_bg = eps_bg,
                                        csca_method = :optical_theorem)
            a, b = sc(D[:, 1]), sc(Dref[:, 1])
            @test isapprox(a.C_ext, b.C_ext; rtol = 1e-6)
            @test isapprox(a.C_abs, b.C_abs; rtol = 1e-6, atol = 1e-12 * abs(b.C_ext))
        end
    end

    # Independent AIM solves, one operator per refractive index.
    independent(m; tol, kwargs...) = solve_cas_v2_orientations(
        basis, eulers; wl_0 = wl_0, m_m = m_m, m_p = m, method = :aim_gmres,
        tol = tol, maxiter = 250, projection = proj, mass = mass,
        return_D = true, return_solve_info = true, kwargs...)

    tol = 1e-8
    res, D, info = solve_cas_v2_orientations_shifted(
        basis, eulers; wl_0 = wl_0, m_m = m_m, m_p_list = m_list,
        tol = tol, maxiter = 250, verify = :all, return_D = true,
        projection = proj, mass = mass)
    refs = [independent(m; tol = tol) for m in m_list]

    # ── T6: against the existing AIM path on the same grid ────────────────
    @testset "T6: shifted vs independent :aim_gmres" begin
        @test size(res) == (length(m_list), L)
        @test length(D) == length(m_list) && all(d -> size(d) == (N, L), D)
        @test all(info.converged)
        for s in eachindex(m_list)
            ref_res, ref_D, ref_info = refs[s]
            @test ref_info.converged
            for i in 1:L
                @test _cas_diff(res[s, i], ref_res[i]) < 1e-6
            end
            @test norm(D[s] .- ref_D) / norm(ref_D) < 1e-5
            # One Arnoldi process, fewer iterations than each independent solve.
            @test info.iterations[s] < ref_info.iterations
        end
        @test info.n_block_iterations == maximum(info.iterations)
        @test size(info.residual_history) == (info.n_block_iterations, length(m_list))
        @test info.reference_index == argmax([abs((m^2 - eps_bg) / m^2) for m in m_list])
        @test all(t -> t >= 0, values(info.timing))
    end

    # ── T7: the reported residual is the residual of the original system ──
    @testset "T7: true residual" begin
        @test all(info.true_residual .< tol)
        ratio = info.true_residual ./ info.residual_norm
        @test all(0.5 .< ratio .< 2.0)
        # Residual with an operator built for the refractive index itself.
        B = Matrix{ComplexF64}(undef, N, L)
        for (j, o) in enumerate(oris)
            B[:, j] = project_plane_wave(basis; k_hat = o.u_inc, E0 = o.e0_inc, k_bg = k0)
        end
        for s in (1, length(m_list))
            op = build_aim_operator(basis; k0 = k0, eps_p = m_list[s]^2, eps_bg = eps_bg,
                                    projection = proj, mass = mass)
            @test norm(B .- aim_mvp(op, D[s])) / norm(B) < tol
        end

        # verify = :hardest checks one index, :none checks nothing.
        _, info_h = solve_cas_v2_orientations_shifted(
            basis, eulers; wl_0 = wl_0, m_m = m_m, m_p_list = m_list,
            tol = 1e-5, maxiter = 250, projection = proj, mass = mass)
        @test count(!isnan, info_h.true_residual) == 1
        @test !isnan(info_h.true_residual[argmax(info_h.iterations)])
        @test all(info_h.converged)
        _, info_n = solve_cas_v2_orientations_shifted(
            basis, eulers; wl_0 = wl_0, m_m = m_m, m_p_list = m_list,
            tol = 1e-5, maxiter = 250, verify = :none, projection = proj, mass = mass)
        @test all(isnan, info_n.true_residual)
        @test info_n.iterations == info_h.iterations
    end

    # ── T8: error against Mie theory equals that of the existing solver ──
    @testset "T8: Mie sphere" begin
        for s in eachindex(m_list)
            mie = mie_cas_observables(; wl_0 = wl_0, m_m = m_m, r_p = r_ve, m_p = m_list[s])
            err_shift = abs(res[s, 1].S_fw_mean - mie.S_fw_mean) / abs(mie.S_fw_mean)
            err_indep = abs(refs[s][1][1].S_fw_mean - mie.S_fw_mean) / abs(mie.S_fw_mean)
            @test err_shift < 0.1                              # coarse 333-tet mesh
            @test abs(err_shift - err_indep) < 1e-6
            err_bk_shift = abs(res[s, 1].S_bk - mie.S_bk) / abs(mie.S_bk)
            err_bk_indep = abs(refs[s][1][1].S_bk - mie.S_bk) / abs(mie.S_bk)
            @test abs(err_bk_shift - err_bk_indep) < 1e-5
        end
    end

    # ── T9: inputs the shifted sweep does not cover ───────────────────────
    @testset "T9: argument checks" begin
        call(list; kwargs...) = solve_cas_v2_orientations_shifted(
            basis, eulers; wl_0 = wl_0, m_m = m_m, m_p_list = list,
            projection = proj, mass = mass, kwargs...)
        @test_throws ArgumentError call(Any[1.5, [1.5, 1.5, 1.7]])
        @test_throws ArgumentError call([1.5 + 0.0im, m_m + 0.0im])
        @test_throws ArgumentError call(ComplexF64[])
        @test_throws ArgumentError call([1.5]; verify = :some)
        @test_throws ArgumentError call([1.5]; maxiter = 0)
        @test_throws ArgumentError solve_cas_v2_orientations_shifted(
            basis, NTuple{3,Float64}[]; wl_0 = wl_0, m_m = m_m, m_p_list = [1.5])
    end

    # ── T10: one refractive index, defaults of both functions ─────────────
    @testset "T10: single index vs solve_cas_v2_orientations" begin
        m = 1.75 + 0.05im
        ref = solve_cas_v2_orientations(basis, eulers; wl_0 = wl_0, m_m = 1.33, m_p = m,
                                        tol = 1e-8, maxiter = 250)
        out, inf1 = solve_cas_v2_orientations_shifted(
            basis, eulers; wl_0 = wl_0, m_m = 1.33, m_p_list = [m],
            tol = 1e-8, maxiter = 250)
        @test size(out) == (1, L)
        @test inf1.converged == [true]
        @test inf1.true_residual[1] < 1e-8
        for i in 1:L
            @test _cas_diff(out[1, i], ref[i]) < 1e-6
        end
    end

    # ── Unconverged indices are flagged, converged ones are unaffected ────
    @testset "maxiter cut-off" begin
        out, inf2 = solve_cas_v2_orientations_shifted(
            basis, eulers; wl_0 = wl_0, m_m = m_m, m_p_list = m_list,
            tol = tol, maxiter = info.iterations[1], verify = :all,
            projection = proj, mass = mass)
        @test inf2.converged[1]
        @test !all(inf2.converged)
        @test all(inf2.iterations .<= info.iterations[1])
        for s in eachindex(m_list)
            # Converged or not, the reported residual is the true one.
            @test isapprox(inf2.true_residual[s], inf2.residual_norm[s]; rtol = 1e-3)
            inf2.converged[s] || @test inf2.residual_norm[s] >= tol
        end
        for i in 1:L
            @test _cas_diff(out[1, i], res[1, i]) < 1e-12
        end
    end

    # ── Opt-in mass-matrix right preconditioning of the ordinary solve ────
    @testset "solve_cas_v2_orientations: precond = :mass_right" begin
        for (s, m) in enumerate(m_list[[1, end]])
            ref_res, ref_D, ref_info = refs[[1, length(m_list)][s]]
            for method in (:aim_gmres, :aim_bicgstab)
                out, Dp, infp = solve_cas_v2_orientations(
                    basis, eulers; wl_0 = wl_0, m_m = m_m, m_p = m, method = method,
                    tol = tol, maxiter = 250, projection = proj, mass = mass,
                    precond = :mass_right, return_D = true, return_solve_info = true)
                @test infp.converged
                for i in 1:L
                    @test _cas_diff(out[i], ref_res[i]) < 1e-6
                end
                @test norm(Dp .- ref_D) / norm(ref_D) < 1e-5
                method === :aim_gmres && @test infp.iterations < ref_info.iterations
            end
        end
        # The default is the unpreconditioned solve, bit for bit.
        a = solve_cas_v2_orientations(basis, eulers; wl_0 = wl_0, m_m = m_m, m_p = m_list[1],
                                      tol = tol, maxiter = 250, projection = proj, mass = mass,
                                      precond = :none)
        @test all(_cas_fields(a[i]) == _cas_fields(refs[1][1][i]) for i in 1:L)
        @test_throws ArgumentError solve_cas_v2_orientations(
            basis, eulers; wl_0 = wl_0, m_m = m_m, m_p = 1.5, precond = :jacobi,
            projection = proj, mass = mass)
        @test_throws ArgumentError solve_cas_v2_orientations(
            basis, eulers; wl_0 = wl_0, m_m = m_m, m_p = 1.5, method = :dense,
            precond = :mass_right)
    end
end
