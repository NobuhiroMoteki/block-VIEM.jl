using Test
using Random
using LinearAlgebra
using SparseArrays
using BlockVIEM
using StaticArrays

# Phase 1 of the shifted refractive-index sweep: the shifted block GMRES
# solver, the K0 product of the AIM operator and the right-preconditioned
# operator C = K0 M^{-1}.

_relerr(a, b) = norm(a .- b) / norm(b)

@testset "shifted block GMRES" begin

    # ── T1: every shift agrees with a direct solve ────────────────────────
    @testset "T1: dense random C, 5 shifts vs direct solve" begin
        rng = MersenneTwister(20261117)
        N, L = 200, 4
        C = randn(rng, ComplexF64, N, N) ./ sqrt(N)          # spectral radius ~ 1
        B = randn(rng, ComplexF64, N, L)                      # shape: (N, L)
        taus = ComplexF64[2.5, -3.0 + 0.5im, 0.8 + 2.4im, 4.0 - 1.0im, -1.5 - 2.5im]

        r = block_gmres_shifted(C, B, taus; tol = 1e-12, maxiter = N ÷ L)
        @test r isa ShiftedBlockSolveResult
        @test all(r.converged)
        @test size(r.residual_history) == (maximum(r.iterations), length(taus))
        for (s, tau) in enumerate(taus)
            A = tau * I - C
            @test _relerr(r.X[s], A \ B) < 1e-10
            # The reported residual is the true residual of the shifted system.
            @test norm(B - A * r.X[s]) / norm(B) ≈ r.residual_norm[s] rtol = 1e-3
            @test r.residual_history[r.iterations[s], s] == r.residual_norm[s]
            @test all(isnan, r.residual_history[r.iterations[s]+1:end, s])
            @test !any(isnan, r.residual_history[1:r.iterations[s], s])
        end
        # Shifts far from the spectrum need fewer iterations than near ones.
        @test length(unique(r.iterations)) > 1
    end

    # ── T2: one shift reproduces block_gmres on (tau I − C) ───────────────
    @testset "T2: single shift vs block_gmres" begin
        rng = MersenneTwister(7)
        N, L = 200, 4
        C = randn(rng, ComplexF64, N, N) ./ sqrt(N)
        B = randn(rng, ComplexF64, N, L)
        for tau in ComplexF64[3.0, 1.2 + 1.8im, -2.0 + 0.3im]
            A = tau * I - C
            for tol in (1e-5, 1e-10)
                ref = block_gmres(A, B; tol = tol, maxiter = N ÷ L)
                r = block_gmres_shifted(C, B, [tau]; tol = tol, maxiter = N ÷ L)
                @test ref.converged && r.converged[1]
                @test r.iterations[1] == ref.iterations
                @test _relerr(r.X[1], ref.X) < 1e-10
                @test r.residual_history[:, 1] ≈ ref.residual_history rtol = 1e-8
            end
        end
    end

    # ── T3: maxiter cut-off is reported per shift ─────────────────────────
    @testset "T3: unconverged shifts at maxiter" begin
        rng = MersenneTwister(11)
        N, L = 200, 4
        C = randn(rng, ComplexF64, N, N) ./ sqrt(N)
        B = randn(rng, ComplexF64, N, L)
        taus = ComplexF64[12.0, 0.1 + 0.1im, -10.0im]         # easy, hard, easy
        maxiter = 6
        r = block_gmres_shifted(C, B, taus; tol = 1e-5, maxiter = maxiter)
        @test r.converged == [true, false, true]
        @test r.iterations[1] < maxiter && r.iterations[3] < maxiter
        @test r.iterations[2] == maxiter
        @test size(r.residual_history, 1) == maxiter
        @test r.residual_norm[2] >= 1e-5
        for (s, tau) in enumerate(taus)
            A = tau * I - C
            # Converged or not, the returned iterate has the reported residual.
            @test norm(B - A * r.X[s]) / norm(B) ≈ r.residual_norm[s] rtol = 1e-6
        end
        # The unconverged iterate is the minimum-residual one of block_gmres.
        ref = block_gmres(taus[2] * I - C, B; tol = 1e-5, maxiter = maxiter)
        @test !ref.converged
        @test _relerr(r.X[2], ref.X) < 1e-10
    end

    @testset "edge cases" begin
        rng = MersenneTwister(3)
        N, L = 40, 3
        C = randn(rng, ComplexF64, N, N) ./ sqrt(N)
        B = randn(rng, ComplexF64, N, L)

        # Zero right-hand side.
        r0 = block_gmres_shifted(C, zeros(ComplexF64, N, L), ComplexF64[2.0, 3.0])
        @test all(r0.converged) && all(iszero, r0.iterations)
        @test all(X -> size(X) == (N, L) && all(iszero, X), r0.X)

        # No shifts.
        re = block_gmres_shifted(C, B, ComplexF64[])
        @test isempty(re.X) && isempty(re.converged)

        # Real inputs and repeated shifts.
        Cr = randn(rng, N, N) ./ sqrt(N)
        Br = randn(rng, N, L)
        rr = block_gmres_shifted(Cr, Br, [3.0, 3.0]; tol = 1e-12, maxiter = N)
        @test rr.X[1] == rr.X[2]
        @test _relerr(rr.X[1], (3.0 * I - Cr) \ Br) < 1e-10

        # An operator that only implements `*`.
        struct _OnlyMul; A::Matrix{ComplexF64}; end
        Base.:*(o::_OnlyMul, X::AbstractMatrix) = o.A * X
        ro = block_gmres_shifted(_OnlyMul(C), B, ComplexF64[2.5 + 1im];
                                 tol = 1e-12, maxiter = N)
        @test _relerr(ro.X[1], ((2.5 + 1im) * I - C) \ B) < 1e-10

        @test_throws ArgumentError block_gmres_shifted(C, B, [2.0]; maxiter = 0)
        @test_throws ArgumentError block_gmres_shifted(
            randn(2, 2), randn(2, 3), [2.0])
    end

    # ── Type stability of the solver body ─────────────────────────────────
    @testset "type stability" begin
        argtypes = (Matrix{ComplexF64}, Matrix{ComplexF64}, Vector{ComplexF64},
                    Float64, Int, Bool)
        ci, rt = only(Base.code_typed(BlockVIEM._block_gmres_shifted_core, argtypes;
                                      optimize = false))
        @test rt === BlockVIEM._ShiftedKrylov
        @test !any(T -> T === Any, ci.slottypes)
        # No value that the body uses is inferred as `Any`. Statements whose
        # result is discarded (branches, `setindex!`, …) are reported as `Any`
        # by the unoptimised IR and are skipped.
        used = Set{Int}()
        scan(x) = x isa Core.SSAValue ? push!(used, x.id) :
                  x isa Expr ? foreach(scan, x.args) :
                  x isa Core.GotoIfNot ? scan(x.cond) :
                  x isa Core.PiNode ? scan(x.val) :
                  x isa Core.ReturnNode ? (isdefined(x, :val) && scan(x.val)) : nothing
        foreach(scan, ci.code)
        @test !isempty(used)
        @test !any(i -> ci.ssavaluetypes[i] === Any, used)

        rng = MersenneTwister(5)
        C = randn(rng, ComplexF64, 30, 30) ./ sqrt(30)
        B = randn(rng, ComplexF64, 30, 2)
        @test (@inferred block_gmres_shifted(C, B, ComplexF64[2.0, 3.0im])) isa
              ShiftedBlockSolveResult
    end
end

@testset "aim_K0_mvp and the right-preconditioned operator" begin
    mesh = read_msh(joinpath(@__DIR__, "meshes", "sphere.msh"))
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N = n_basis(basis)
    k0 = 2.0                                                  # unit sphere, x = 2
    eps_bg = ComplexF64(1.33^2)
    pitch = 0.5 * mean_edge_length(mesh)
    grid = aim_grid(mesh; pitch = pitch, padding = 4)
    proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    mass = assemble_mass_matrix(basis)
    build(eps_p) = build_aim_operator(basis; k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                      projection = proj, mass = mass)
    eps_list = ComplexF64[2.25 + 0.03im, 9.0 + 1.0im, -12.1 + 1.22im]
    ops = [build(e) for e in eps_list]
    rng = MersenneTwister(42)
    X = randn(rng, ComplexF64, N, 3)                          # shape: (N, 3)

    # Method b of the plan: K0 X = (eps_bg/kappa) [(1/eps_p) M X − Z X].
    k0_by_difference(op, X) =
        (op.eps_bg / op.kappa_v[1]) .* (op.inv_eps_v[1] .* (op.mass * X) .- aim_mvp(op, X))

    # ── T4: method a against method b, and independence of eps_p ──────────
    @testset "T4: K0 product" begin
        Y = [aim_K0_mvp(op, X) for op in ops]
        for (op, Yi) in zip(ops, Y)
            @test _relerr(Yi, k0_by_difference(op, X)) < 1e-9
        end
        @test _relerr(Y[2], Y[1]) < 1e-10
        @test _relerr(Y[3], Y[1]) < 1e-10

        # Vector method equals the block method column by column.
        for j in 1:size(X, 2)
            @test _relerr(aim_K0_mvp(ops[1], X[:, j]), Y[1][:, j]) < 1e-13
        end

        # Z X rebuilt from M and K0 for a permittivity the operator was not built for.
        op_ref = ops[2]
        for (e, op) in zip(eps_list, ops)
            kappa = (e - eps_bg) / e
            ZX = (1 / e) .* (mass * X) .- (kappa / eps_bg) .* aim_K0_mvp(op_ref, X)
            @test _relerr(ZX, aim_mvp(op, X)) < 1e-10
        end
    end

    @testset "T4: half-SWG surface term stored separately" begin
        # build_aim_operator leaves half_swg_extra empty. Fill it by hand to
        # cover operators that carry the surface terms outside the precorrection.
        op = ops[1]
        extra = sparse(triu(sprandn(MersenneTwister(1), ComplexF64, N, N, 0.01)))
        op2 = AIMOperator(op.projection, op.G_hat, op.mass, op.precorrection,
                          op.diag_precorrection, extra, Vector{ComplexF64}(diag(extra)),
                          op.k0, op.eps_bg, op.kappa_v, op.kappa_avg, op.inv_eps_v)
        @test _relerr(aim_K0_mvp(op2, X), k0_by_difference(op2, X)) < 1e-9
        @test _relerr(aim_K0_mvp(op2, X[:, 1]), k0_by_difference(op2, X)[:, 1]) < 1e-9
        @test _relerr(aim_K0_mvp(op2, X), aim_K0_mvp(op, X)) > 1e-3
    end

    @testset "T4: unsupported operators" begin
        op_aniso = build(ComplexF64[2.25, 2.25, 4.0])
        @test_throws ArgumentError aim_K0_mvp(op_aniso, X)
        @test_throws ArgumentError aim_K0_mvp(op_aniso, X[:, 1])
        op_null = build(eps_bg)                               # kappa = 0
        @test_throws ArgumentError aim_K0_mvp(op_null, X)
    end

    @testset "mass solve" begin
        Mi = BlockVIEM._MassCholesky(mass)
        Y = BlockVIEM._mass_solve(Mi, X)
        @test Y isa Matrix{ComplexF64}
        @test _relerr(mass * Y, X) < 1e-12
        C = BlockVIEM._K0MinvOp(ops[1])
        @test _relerr(C * X, aim_K0_mvp(ops[1], Y)) < 1e-13
        @test (@inferred C * X) isa Matrix{ComplexF64}
    end

    # ── One Arnoldi process on K0 M^{-1} solves Z(eps_p) X = B for all eps_p ──
    @testset "shifted solve of Z(eps_p) X = B" begin
        # Plane-wave right-hand sides of three orientations, shape (N, 3).
        eulers = [(0.0, 0.0, 0.0), (0.3, 1.1, 0.0), (2.0, 2.4, 0.7)]
        B = Matrix{ComplexF64}(undef, N, length(eulers))
        for (j, eu) in enumerate(eulers)
            ori = cas_orientation(eu...)
            B[:, j] = project_plane_wave(basis; k_hat = ori.u_inc, E0 = ori.e0_inc,
                                         k_bg = k0)
        end
        C = BlockVIEM._K0MinvOp(ops[2])
        Mi = C.minv
        taus = [BlockVIEM._shift_tau(e, eps_bg) for e in eps_list]
        tol = 1e-8
        r = block_gmres_shifted(C, B, taus; tol = tol, maxiter = N ÷ 3)
        @test all(r.converged)
        # The dielectric shifts are frozen long before the metallic one.
        @test r.iterations[1] < r.iterations[2] < r.iterations[3]
        Id = Matrix{ComplexF64}(I, N, N)
        for (s, (e, op)) in enumerate(zip(eps_list, ops))
            kappa = BlockVIEM._shift_kappa(e, eps_bg)
            Xs = (eps_bg / kappa) .* BlockVIEM._mass_solve(Mi, r.X[s])
            # True residual of the original system, with the operator built for
            # eps_p itself. Right preconditioning makes it the GMRES residual.
            true_res = norm(B .- aim_mvp(op, Xs)) / norm(B)
            @test true_res < tol
            @test true_res ≈ r.residual_norm[s] rtol = 1e-3
            # Direct solve with the same AIM operator in dense form.
            Zs = aim_mvp(op, Id)                              # shape: (N, N)
            @test _relerr(Xs, Zs \ B) < 1e-5
        end
    end
end
