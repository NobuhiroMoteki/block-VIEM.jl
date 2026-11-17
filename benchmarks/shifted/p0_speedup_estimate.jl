# Phase 0, item 5: wall-time model of a refractive-index sweep at fixed (D_ve, AR).
#
# The existing /target/cost/ fields cannot be used for this split: t_solve_s
# contains the operator build (Green FFT + precorrection) together with the
# Krylov solve. The model below therefore uses the split measured by
# p0_minv_cost.jl (same machine, same thread count) and the iteration counts
# measured by p0_precond_convergence.jl.
#
# Per (D_ve, AR), with N_m refractive indices and one block of L orientations:
#
#   current   = T_sh + N_m (t_op + K_Z t_mvp + t_post)
#       iron-oxide driver today: mesh, projection, mass shared; operator rebuilt per m
#   reuse     = T_sh + t_op + N_m (K_Z t_mvp + t_post)
#       K0 scaling only: one precorrection serves every m; unpreconditioned solves
#   precond   = T_sh + t_op + t_fact + N_m (K_R (t_mvp + t_minv) + t_minv + t_post)
#       K0 scaling + right M^{-1} preconditioning; independent solves
#   shifted   = T_sh + t_op + t_fact + K_max (t_mvp + t_minv) + N_m (t_minv + t_post)
#       one block Arnoldi process for the whole batch
#
# T_sh = t_mesh + t_proj + t_mass. Orthogonalisation and the small least-squares
# problems are left out (same order in every variant; < 3 % in the prototype).
#
# Run: julia --project=. benchmarks/shifted/p0_speedup_estimate.jl [TAG]

include(joinpath(@__DIR__, "_common.jl"))

const TAG = isempty(ARGS) ? "supermicro" : ARGS[1]
const L_REF = 13                       # iron-oxide polar grid
const NM_LIST = (3, 10, 30)

readjson(name) = JSON3.read(read(joinpath(RESULTS_DIR, name), String))

function main()
    cost = readjson("p0_minv_cost_$(TAG).json")
    conv = readjson("p0_precond_convergence.json")
    proto = readjson("p0_shifted_prototype.json")

    # Post-processing cost per (DOF x orientation), from the prototype runs.
    post_coeff = maximum(sum(c.t_ind_post) / length(c.t_ind_post) / (c.n_dof * c.L)
                         for c in proto.cases)

    # Iteration scenarios (K_Z, K_R mean, K_R max over a batch).
    function scen(labels)
        cs = [c for c in conv.cases if any(occursin(l, c.label) for l in labels) && c.conv_Z]
        kz = sum(c.iters_Z for c in cs) / length(cs)
        kr = sum(c.iters_right for c in cs) / length(cs)
        return kz, kr
    end
    kz_d, kr_d = scen(("n15", "n20"))
    kz_h, kr_h = scen(("hem",))
    h27 = only(c for c in proto.cases if c.label == "oblate_hematite27")
    d7 = only(c for c in proto.cases if c.label == "sphere_dielectric7")
    spread(c) = maximum(c.iters_shift) / (sum(c.iters_shift) / length(c.iters_shift))
    scenarios = [("dielectric (n15, n20)", kz_d, kr_d, kr_d * spread(d7)),
                 ("iron oxide (2.7-3.1 + 0.05-0.2i)", kz_h, kr_h, kr_h * spread(h27))]

    @printf("cost data: %s, %d threads | L = %d | post-processing %.2e s per DOF per orientation\n",
            cost.host, cost.threads, L_REF, post_coeff)
    out = Any[]
    for (name, kz, kr, kmax) in scenarios
        @printf("\n%s: K_Z = %.1f, K_R = %.1f, K_max = %.1f\n", name, kz, kr, kmax)
        println("   N_DOF    t_op  t_mvp  t_minv |  N_m | current   reuse  precond  shifted [s] | speedup: reuse precond shifted")
        for s in cost.sizes
            s.mvp === nothing && continue
            mrow = only(r for r in s.mvp.rows if r.L == L_REF)
            irow = only(r for r in s.minv if r.L == L_REF)
            t_sh = s.t_mesh + s.mvp.t_proj + s.t_mass
            t_op, t_mvp, t_minv, t_fact = s.mvp.t_op, mrow.t_k0, irow.t_chol, s.t_fact
            t_post = post_coeff * s.n_dof * L_REF
            for nm in NM_LIST
                cur = t_sh + nm * (t_op + kz * t_mvp + t_post)
                reu = t_sh + t_op + nm * (kz * t_mvp + t_post)
                pre = t_sh + t_op + t_fact + nm * (kr * (t_mvp + t_minv) + t_minv + t_post)
                shf = t_sh + t_op + t_fact + kmax * (t_mvp + t_minv) + nm * (t_minv + t_post)
                @printf("  %7d %6.1f %6.3f %6.3f | %4d | %7.0f %7.0f %7.0f %7.0f     | %12.1f %7.1f %7.1f\n",
                        s.n_dof, t_op, t_mvp, t_minv, nm, cur, reu, pre, shf,
                        cur / reu, cur / pre, cur / shf)
                push!(out, (scenario = name, n_dof = s.n_dof, n_m = nm, K_Z = kz, K_R = kr,
                            K_max = kmax, t_shared = t_sh, t_op = t_op, t_mvp = t_mvp,
                            t_minv = t_minv, t_fact = t_fact, t_post = t_post,
                            current = cur, reuse = reu, precond = pre, shifted = shf,
                            S_reuse = cur / reu, S_precond = cur / pre, S_shifted = cur / shf))
            end
        end
    end
    save_json("p0_speedup_estimate_$(TAG).json", (L = L_REF, post_coeff = post_coeff, rows = out))
end

main()
