# Phase 0, item 4: cost of the M^{-1} action against one block MVP.
#
# For a homogeneous sphere meshed to a target N_DOF this script measures
#   - sparse Cholesky of the mass matrix: factorisation time, factor size,
#     solve time for an (N, L) complex block;
#   - Jacobi-preconditioned CG to a relative residual of 1e-13: iterations, time;
#   - with --mvp: the AIM set-up split (projection, mass, operator = Green FFT +
#     precorrection) and the time of one block MVP, giving
#         r_M = t(M^{-1} block) / t(block MVP).
#
# Run: julia --project=. -t 8 benchmarks/shifted/p0_minv_cost.jl [--mvp] [--tag NAME] N_DOF...

include(joinpath(@__DIR__, "_common.jl"))

const WITH_MVP = "--mvp" in ARGS
const TAG = let i = findfirst(==("--tag"), ARGS)
    i === nothing ? "local" : ARGS[i+1]
end
const TARGETS = let rest = String[]
    skip = false
    for a in ARGS
        if skip; skip = false; continue; end
        a == "--mvp" && continue
        if a == "--tag"; skip = true; continue; end
        push!(rest, a)
    end
    isempty(rest) ? [10_000] : [round(Int, parse(Float64, a)) for a in rest]
end
const L_LIST = (5, 13, 64)
const A_EQ = 0.4          # sphere radius [um]
const M_P = 2.0 + 0.0im   # only scales kappa; timings do not depend on it

function peak_rss_gb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Float64, split(line)[2]) / 1024^2
    end
    return NaN
end

"Number of stored numerical values in a CHOLMOD factor."
function factor_nvals(F)
    try
        st = unsafe_load(pointer(F))
        return Int(st.is_super != 0 ? st.xsize : st.nzmax)
    catch
        return nnz(sparse(F.L))
    end
end

"Best of `n` timings of `f()` [s]."
function best_time(f, n::Int = 3)
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

function one_size(n_dof_target::Int)
    # Empirical gmsh density for these ellipsoids: V_tet ~ 0.21 lc^3, N_DOF ~ 2.07 N_tet.
    V = 4π / 3 * A_EQ^3
    lc = cbrt(V / (0.21 * n_dof_target / 2.07))
    t_mesh = @elapsed ((mesh, basis) = build_shape(:sphere, A_EQ; lc = lc))
    N = n_basis(basis)
    t_mass = @elapsed M = assemble_mass_matrix(basis)
    @printf("\n== N_DOF target %d: lc = %.5f um, N_tet = %d, N_DOF = %d, nnz(M) = %d (%.1f per row)\n",
            n_dof_target, lc, n_tets(mesh), N, nnz(M), nnz(M) / N)
    @printf("   mesh + basis %.1f s, mass %.2f s\n", t_mesh, t_mass)
    flush(stdout)

    t_fact = @elapsed F = cholesky(Symmetric(M))
    nvals = factor_nvals(F)
    minv = MinvChol(F)
    @printf("   Cholesky: %.2f s, factor values %d (%.1f x nnz(tril M), %.2f GB)\n",
            t_fact, nvals, nvals / ((nnz(M) + N) / 2), nvals * 8 / 1024^3)

    rows = NamedTuple[]
    for L in L_LIST
        B = randn(MersenneTwister(L), ComplexF64, N, L)        # shape: (N, L)
        t_chol = best_time(() -> minv(B))
        Xc = minv(B)
        e_chol = norm(M * Xc .- B) / norm(B)
        Xp = Matrix{ComplexF64}(undef, N, L)
        it = pcg_jacobi!(Xp, M, B; rtol = 1e-13)
        t_pcg = best_time(() -> pcg_jacobi!(Xp, M, B; rtol = 1e-13))
        e_pcg = norm(M * Xp .- B) / norm(B)
        push!(rows, (L = L, t_chol = t_chol, err_chol = e_chol,
                     t_pcg = t_pcg, pcg_iters = maximum(it), err_pcg = e_pcg,
                     chol_vs_pcg = relerr(Xc, Xp)))
        @printf("   L=%3d  Cholesky solve %.3f s (res %.1e) | PCG %.3f s, %d iters (res %.1e) | X diff %.1e\n",
                L, t_chol, e_chol, t_pcg, maximum(it), e_pcg, relerr(Xc, Xp))
        flush(stdout)
    end

    mvp = nothing
    if WITH_MVP
        pitch = 0.5 * mean_edge_length(mesh)
        grid = aim_grid(mesh; pitch = pitch, padding = 4)
        t_proj = @elapsed proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
        t_op = @elapsed op = build_aim_operator(basis; k0 = K_BG, eps_p = M_P^2,
                                                eps_bg = EPS_BG, projection = proj, mass = M)
        @printf("   AIM grid %s, projection %.1f s, operator (Green FFT + precorrection) %.1f s, nnz(precorr) = %d\n",
                string(grid.dims), t_proj, t_op, nnz(op.precorrection))
        flush(stdout)
        mrows = NamedTuple[]
        for (i, L) in enumerate(L_LIST)
            X = randn(MersenneTwister(100 + L), ComplexF64, N, L)
            aim_mvp(op, X)                                   # warm-up (FFT plans)
            t_z = best_time(() -> aim_mvp(op, X), 2)
            t_k0 = best_time(() -> k0_mvp(op, X), 2)
            r = rows[i]
            push!(mrows, (L = L, t_mvp = t_z, t_k0 = t_k0,
                          rM_chol = r.t_chol / t_k0, rM_pcg = r.t_pcg / t_k0))
            @printf("   L=%3d  block MVP Z %.3f s, K0 %.3f s | r_M: Cholesky %.3f, PCG %.3f\n",
                    L, t_z, t_k0, r.t_chol / t_k0, r.t_pcg / t_k0)
            flush(stdout)
        end
        mvp = (grid_dims = collect(grid.dims), t_proj = t_proj, t_op = t_op,
               nnz_precorr = nnz(op.precorrection), rows = mrows)
    end
    @printf("   peak RSS so far %.1f GB\n", peak_rss_gb())
    return (n_dof_target = n_dof_target, lc = lc, n_tet = n_tets(mesh), n_dof = N,
            nnz_M = nnz(M), t_mesh = t_mesh, t_mass = t_mass, t_fact = t_fact,
            factor_values = nvals, minv = rows, mvp = mvp, peak_rss_gb = peak_rss_gb())
end

function main()
    @printf("threads: julia %d, BLAS %d | sizes %s | mvp %s\n",
            Threads.nthreads(), BLAS.get_num_threads(), string(TARGETS), WITH_MVP)
    one_size(2_000)                                          # compile
    results = [one_size(n) for n in TARGETS]
    save_json("p0_minv_cost_$(TAG).json",
              (threads = Threads.nthreads(), blas_threads = BLAS.get_num_threads(),
               host = gethostname(), with_mvp = WITH_MVP, a_eq = A_EQ, sizes = results))
end

main()
