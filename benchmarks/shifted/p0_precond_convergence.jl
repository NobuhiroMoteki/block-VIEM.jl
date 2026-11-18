# Phase 0, item 3: how does M^{-1} preconditioning change block-GMRES convergence?
#
# For each paper slot the same right-hand sides are solved three ways with the
# existing `block_gmres`:
#
#   Z      : Z X = B                       (current solver)
#   left   : M^{-1} Z X = M^{-1} B         (plan §2.2; minimises the M^{-1} residual)
#   right  : Z M^{-1} Y = B, X = M^{-1} Y  (minimises the true residual B - Z X)
#
# Z M^{-1} = (kappa/eps_bg)(tau I - K0 M^{-1}), and GMRES iteration counts do not
# depend on a scalar factor, so `right` is exactly the convergence of the shifted
# system for that refractive index.
#
# Item 3b: a shifted block-GMRES prototype (one Arnoldi process on K0 M^{-1} for a
# whole refractive-index batch) checked against independent solves.
#
# Run: julia --project=. -t 10 benchmarks/shifted/p0_precond_convergence.jl [conv|shift|all]

include(joinpath(@__DIR__, "_common.jl"))

const TOL = 1e-5
const PADDING = 4

function setup_shared(basis)
    pitch = 0.5 * mean_edge_length(basis.mesh)
    grid = aim_grid(basis.mesh; pitch = pitch, padding = PADDING)
    t_proj = @elapsed proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    t_mass = @elapsed mass = assemble_mass_matrix(basis)
    return proj, mass, t_proj, t_mass
end

build_op(basis, m_p, proj, mass) =
    build_aim_operator(basis; k0 = K_BG, eps_p = ComplexF64(m_p)^2, eps_bg = EPS_BG,
                       projection = proj, mass = mass)

true_res(op, X, B) = norm(B .- aim_mvp(op, X)) / norm(B)

# ── Item 3 ───────────────────────────────────────────────────────────────────

function convergence_case(label, shape, a_eq, m_p, eulers; maxiter)
    mesh, basis = build_shape(shape, a_eq; m_p_max = abs(m_p))
    N = n_basis(basis)
    L = length(eulers)
    proj, mass, _, _ = setup_shared(basis)
    op = build_op(basis, m_p, proj, mass)
    minv = MinvChol(mass)
    B = rhs_block(basis, eulers)

    t_z = @elapsed rz = block_gmres(ZOp(op), B; tol = TOL, maxiter = maxiter)
    t_l = @elapsed rl = block_gmres(LeftOp(op, minv), minv(B); tol = TOL, maxiter = maxiter)
    t_r = @elapsed rr = block_gmres(RightOp(op, minv), B; tol = TOL, maxiter = maxiter)
    Xr = minv(rr.X)

    out = (label = label, shape = String(shape), a_eq = a_eq,
           m_p = (real(m_p), imag(m_p)), tau = (real(tau_of(m_p^2)), imag(tau_of(m_p^2))),
           n_tet = n_tets(mesh), n_dof = N, L = L, maxiter = maxiter,
           iters_Z = rz.iterations, iters_left = rl.iterations, iters_right = rr.iterations,
           conv_Z = rz.converged, conv_left = rl.converged, conv_right = rr.converged,
           res_Z = rz.residual_norm, res_left_Minv = rl.residual_norm,
           res_right = rr.residual_norm,
           true_res_Z = true_res(op, rz.X, B),
           true_res_left = true_res(op, rl.X, B),
           true_res_right = true_res(op, Xr, B),
           it1e3_Z = iters_to(rz.residual_history, 1e-3),
           it1e3_right = iters_to(rr.residual_history, 1e-3),
           it1e3_left_Minv = iters_to(rl.residual_history, 1e-3),
           X_right_vs_Z = relerr(Xr, rz.X), X_left_vs_Z = relerr(rl.X, rz.X),
           t_Z = t_z, t_left = t_l, t_right = t_r,
           hist_Z = rz.residual_history, hist_left = rl.residual_history,
           hist_right = rr.residual_history)
    @printf("%-22s N=%6d L=%3d | iters Z %3d  left %3d  right %3d | true res Z %.1e  left %.1e (M^-1 %.1e)  right %.1e | to 1e-3: Z %3d right %3d | t %.0f/%.0f/%.0f s\n",
            label, N, L, out.iters_Z, out.iters_left, out.iters_right,
            out.true_res_Z, out.true_res_left, out.res_left_Minv, out.true_res_right,
            out.it1e3_Z, out.it1e3_right, t_z, t_l, t_r)
    flush(stdout)
    return out
end

function run_convergence()
    n15 = 1.5 + 0.01im
    n20 = 2.0 + 0.0im
    hem = 2.9 + 0.1im
    au  = 0.17525 + 3.4830im
    e5 = euler_cosbeta5()
    cases = Any[]
    for (mname, m) in (("n15", n15), ("n20", n20)), shape in (:sphere, :oblate), a in (0.1, 0.2)
        push!(cases, ("$(shape)_$(mname)_a$(a)", shape, a, m, e5, 200))
    end
    for shape in (:sphere, :oblate)
        push!(cases, ("$(shape)_hem_a0.1", shape, 0.1, hem, e5, 200))
    end
    push!(cases, ("sphere_Au_a0.1_L1", :sphere, 0.1, au, euler_random(1), 200))
    push!(cases, ("sphere_Au_a0.1_L64", :sphere, 0.1, au, euler_random(64), 120))
    results = Any[]
    for c in cases
        push!(results, convergence_case(c[1], c[2], c[3], c[4], c[5]; maxiter = c[6]))
        GC.gc()
    end
    save_json("p0_precond_convergence.json", (tol = TOL, cases = results))
end

# ── Item 3b ──────────────────────────────────────────────────────────────────

function observables(basis, X, eulers, m_p)
    out = Vector{ComplexF64}(undef, 3 * length(eulers))
    for (i, ea) in enumerate(eulers)
        r = compute_cas_observables(basis, view(X, :, i);
                                    orientation = cas_orientation(ea...),
                                    k0 = K_BG, eps_p = ComplexF64(m_p)^2, eps_bg = EPS_BG)
        out[3i-2] = r.S_fw_theta
        out[3i-1] = r.S_fw_phi
        out[3i]   = r.S_bk
    end
    return out
end

function shifted_case(label, shape, a_eq, m_list, eulers; tol, maxiter, ref_idx = eachindex(m_list))
    m_max = maximum(abs, m_list)
    mesh, basis = build_shape(shape, a_eq; m_p_max = m_max)
    N = n_basis(basis)
    L = length(eulers)
    S = length(m_list)
    proj, mass, t_proj, t_mass = setup_shared(basis)
    B = rhs_block(basis, eulers)

    # Independent solves, as the iron-oxide driver does them today
    # (mesh, projection and mass shared; operator rebuilt for every m).
    ind = Dict{Int,Any}()
    for s in ref_idx
        t_op = @elapsed op = build_op(basis, m_list[s], proj, mass)
        t_sv = @elapsed r = block_gmres(ZOp(op), B; tol = tol, maxiter = maxiter)
        t_post = @elapsed obs = observables(basis, r.X, eulers, m_list[s])
        ind[s] = (X = r.X, iters = r.iterations, res = r.residual_norm, obs = obs,
                  t_op = t_op, t_solve = t_sv, t_post = t_post)
        GC.gc()
    end

    # Shifted solve: one operator (largest |kappa|), one Arnoldi process.
    i_ref = argmax(abs.(kappa_of.(ComplexF64.(m_list) .^ 2)))
    t_op_ref = @elapsed op_ref = build_op(basis, m_list[i_ref], proj, mass)
    t_chol = @elapsed minv = MinvChol(mass)
    taus = ComplexF64[tau_of(ComplexF64(m)^2) for m in m_list]
    cmul = V -> k0_mvp(op_ref, minv(V))
    t_shift = @elapsed begin
        Y, iters, res, k_total, t_small = shifted_block_gmres_right(cmul, B, taus;
                                                                    tol = tol, maxiter = maxiter)
    end
    t_recon = @elapsed Xs = [(EPS_BG / kappa_of(ComplexF64(m_list[s])^2)) .* minv(Y[s])
                             for s in 1:S]
    t_post_sh = @elapsed obs_sh = [observables(basis, Xs[s], eulers, m_list[s]) for s in 1:S]

    # True residual of Z_s X_s = B, evaluated through M and K0 only.
    true_r = map(1:S) do s
        e = ComplexF64(m_list[s])^2
        ZX = (mass * Xs[s]) ./ e .- (kappa_of(e) / EPS_BG) .* k0_mvp(op_ref, Xs[s])
        norm(B .- ZX) / norm(B)
    end

    refs = sort(collect(keys(ind)))
    dX   = [relerr(Xs[s], ind[s].X) for s in refs]
    dObs = [maximum(abs.(obs_sh[s] .- ind[s].obs) ./ abs.(ind[s].obs)) for s in refs]
    t_ind_per_m = sum(ind[s].t_op + ind[s].t_solve + ind[s].t_post for s in refs) / length(refs)
    t_shift_tot = t_op_ref + t_chol + t_shift + t_recon + t_post_sh
    out = (label = label, shape = String(shape), a_eq = a_eq, tol = tol,
           n_tet = n_tets(mesh), n_dof = N, L = L, n_m = S,
           m_list = [(real(m), imag(m)) for m in m_list],
           taus = [(real(t), imag(t)) for t in taus],
           iters_shift = iters, k_total = k_total, res_shift = res, true_res_shift = true_r,
           ref_idx = refs, iters_ind = [ind[s].iters for s in refs],
           X_diff = dX, obs_diff = dObs,
           t_proj = t_proj, t_mass = t_mass,
           t_ind_op = [ind[s].t_op for s in refs], t_ind_solve = [ind[s].t_solve for s in refs],
           t_ind_post = [ind[s].t_post for s in refs],
           t_op_ref = t_op_ref, t_chol = t_chol, t_shift_solve = t_shift, t_small_ls = t_small,
           t_recon = t_recon, t_post_shift = t_post_sh,
           t_ind_per_m = t_ind_per_m, t_shift_total = t_shift_tot,
           speedup_excl_shared = S * t_ind_per_m / t_shift_tot)
    @printf("%-26s N=%6d L=%2d N_m=%2d tol=%.0e | shifted iters %s (total %d) | independent %s\n",
            label, N, L, S, tol, string(iters), k_total, string(out.iters_ind))
    @printf("    max |res - true res| %.1e, max true res %.2e | X diff max %.1e | observable diff max %.1e\n",
            maximum(abs.(res .- true_r)), maximum(true_r), maximum(dX), maximum(dObs))
    @printf("    independent per m: op %.1f + solve %.1f + post %.1f s | shifted: op %.1f + chol %.2f + solve %.1f (small LS %.2f) + recon %.2f + post %.1f s | speedup %.2f (N_m = %d)\n",
            sum(out.t_ind_op) / length(refs), sum(out.t_ind_solve) / length(refs),
            sum(out.t_ind_post) / length(refs), t_op_ref, t_chol, t_shift, t_small,
            t_recon, t_post_sh, out.speedup_excl_shared, S)
    flush(stdout)
    return out
end

function run_shifted()
    e5 = euler_cosbeta5()
    results = Any[]
    diel = ComplexF64[re + 0.01im for re in 1.40:0.05:1.70]
    hem3 = ComplexF64[re + 0.1im for re in (2.7, 2.9, 3.1)]
    hem27 = ComplexF64[re + im_ * im for im_ in (0.05, 0.1, 0.2)
                       for re in range(2.7, 3.1, length = 9)]
    push!(results, shifted_case("sphere_dielectric7", :sphere, 0.1, diel, e5;
                                tol = 1e-5, maxiter = 200))
    push!(results, shifted_case("oblate_hematite3", :oblate, 0.1, hem3, e5;
                                tol = 1e-5, maxiter = 200))
    push!(results, shifted_case("oblate_hematite3_tight", :oblate, 0.1, hem3, e5;
                                tol = 1e-10, maxiter = 200))
    push!(results, shifted_case("oblate_hematite27", :oblate, 0.1, hem27, e5;
                                tol = 1e-5, maxiter = 200, ref_idx = [1, 14, 27]))
    save_json("p0_shifted_prototype.json", (cases = results,))
end

mode = isempty(ARGS) ? "all" : ARGS[1]
mode in ("conv", "all") && run_convergence()
mode in ("shift", "all") && run_shifted()
