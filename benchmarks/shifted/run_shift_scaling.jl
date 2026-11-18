# Scaling benchmark of the shifted block GMRES for refractive-index sweeps.
#
# For one oblate spheroid (D_ve, b/c) the refractive index is swept over N_m
# values on a common mesh. Three ways of solving the sweep are timed.
#
#   B  one existing block-GMRES solve per m (`solve_cas_v2_orientations`,
#      `:aim_gmres`), with the mesh, the AIM projection and the mass matrix
#      shared. This is what `benchmarks/cas_v2/ironoxide_sweep_h5.jl` does
#      without `--shifted-ri`. The AIM operator is rebuilt for every m.
#   P  as B, with `precond = :mass_right`. Shows what the preconditioner alone
#      gives. Run only for the N_m listed in --p-n-m.
#   C  ONE shifted block-GMRES process for all m
#      (`solve_cas_v2_orientations_shifted`). This is `--shifted-ri`.
#
# B and C solve the same linear systems on the same mesh and AIM grid, so
# T_B / T_C is the gain of `--shifted-ri` over the present production path.
# The mesh generation, the projection and the mass matrix are shared by both
# and are reported apart (t_shared); the totals T_B and T_C include them.
#
# For N_m > 1 the refractive indices are equispaced between the two ends of
# the family. For N_m = 1 the single index is the upper end. Solves of B and P
# are done once per distinct m and reused for every N_m that contains it.
# Orientations follow the spheroid sweeps: alpha = gamma = 0 and cos(beta)
# equidistant on [0, 1].
#
# The mesh follows the production rule
#     lc = min(lambda_0 / (|m|_max N_pw), 0.3 c),   N_pw = 10,
# with c the short semi-axis and |m|_max the largest |m| of the family.
#
# Lengths in micrometres, times in seconds, memory in bytes.
#
# Run (from the repository root):
#   julia --project=. -t 8 benchmarks/shifted/run_shift_scaling.jl \
#       --family ironoxide --d-ve 0.35 --L 13 --tag NAME
# Options:
#   --family ironoxide | dielectric     refractive-index family (see FAMILIES)
#   --d-ve X          volume-equivalent diameter [um]            (default 0.35)
#   --bc X            axis ratio b/c of the oblate spheroid      (default 2)
#   --L n             number of orientations                     (default 13)
#   --n-m a,b,...     sweep sizes for C                          (default 1,3,10,30)
#   --b-n-m a,b,...   sweep sizes that also get method B         (default: all of --n-m)
#   --p-n-m a,b,...   sweep sizes that also get method P         (default 3)
#   --tol X           solver tolerance                           (default 1e-5)
#   --maxiter n       cap on block iterations                    (default 200)
#   --tag NAME        output file results/shift_scaling_NAME.json

include(joinpath(@__DIR__, "_common.jl"))
include(joinpath(@__DIR__, "..", "..", "viem_results", "rss_monitor.jl"))
using .RSSMonitor
using Dates

function cli(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i+1]
end
ints(s) = isempty(s) ? Int[] : parse.(Int, split(s, ","))

# (lower end, upper end) of the swept complex refractive index.
const FAMILIES = Dict(
    "ironoxide"  => (2.7 + 0.1im, 3.1 + 0.1im),     # hematite-like, as in Phase 0
    "dielectric" => (1.35 + 0.0im, 1.70 + 0.0im),   # the range of the block-DDA_Py benchmark
)

const FAMILY  = cli("--family", "ironoxide")
const D_VE    = parse(Float64, cli("--d-ve", "0.35"))
const BC      = parse(Float64, cli("--bc", "2"))
const L_ORI   = parse(Int, cli("--L", "13"))
const NM_LIST = ints(cli("--n-m", "1,3,10,30"))
const B_NM    = ints(cli("--b-n-m", join(NM_LIST, ",")))
const P_NM    = ints(cli("--p-n-m", "3"))
const TOL     = parse(Float64, cli("--tol", "1e-5"))
const MAXITER = parse(Int, cli("--maxiter", "200"))
const TAG     = cli("--tag", "$(FAMILY)_D$(D_VE)_L$(L_ORI)")
const N_PW    = 10

haskey(FAMILIES, FAMILY) || error("unknown family $FAMILY")
issubset(B_NM, NM_LIST) && issubset(P_NM, NM_LIST) ||
    error("--b-n-m and --p-n-m must be subsets of --n-m")

"Refractive indices of a sweep of size `n`."
function m_grid(n::Int)
    lo, hi = FAMILIES[FAMILY]
    return n == 1 ? ComplexF64[hi] : ComplexF64[lo + (hi - lo) * (i - 1) / (n - 1) for i in 1:n]
end

"Largest difference of the forward amplitudes, relative to the larger of the two."
function fw_diff(a::CASv2Result, b::CASv2Result)
    scale = max(abs(b.S_fw_theta), abs(b.S_fw_phi))
    return max(abs(a.S_fw_theta - b.S_fw_theta), abs(a.S_fw_phi - b.S_fw_phi)) / scale
end

"Position of `m` in `ms` (the grids are built from the same end points)."
function find_m(ms::Vector{ComplexF64}, m::ComplexF64)
    i = findfirst(x -> abs(x - m) <= 1e-12 * abs(m), ms)
    i === nothing && error("refractive index $m not in the list")
    return i
end

function main()
    mon = RSSMonitor.Monitor()
    RSSMonitor.start!(mon)
    lo, hi = FAMILIES[FAMILY]
    m_abs_max = max(abs(lo), abs(hi))
    @printf("%s  family %s (%s .. %s)  D_ve %.3f um  b/c %.2f  L %d  tol %.0e  threads %d\n",
            Dates.format(now(), "yyyy-mm-dd HH:MM"), FAMILY, lo, hi, D_VE, BC, L_ORI, TOL,
            Threads.nthreads())

    # ── Shared set-up: mesh, basis, AIM projection, mass matrix ──
    p = GREParams(D_VE / 2, BC, 1.0, 0.0)
    c_min = gre_semi_axes(p)[3]
    lc = min(WL_0 / (m_abs_max * N_PW), 0.3 * c_min)
    t_mesh = @elapsed begin
        mesh, _ = gre_mesh(p, MersenneTwister(20260417); lc = lc)
        basis = build_swg_basis(mesh; include_boundary_faces = true)
    end
    N = n_basis(basis)
    pitch = 0.5 * mean_edge_length(mesh)
    grid = aim_grid(mesh; pitch = pitch, padding = 4)
    t_proj = @elapsed proj = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    t_mass = @elapsed mass = assemble_mass_matrix(basis)
    t_shared = t_mesh + t_proj + t_mass
    @printf("mesh: lc %.5f um, N_tet %d, N_DOF %d, AIM grid %s | mesh %.1f s, projection %.1f s, mass %.2f s\n",
            lc, n_tets(mesh), N, string(grid.dims), t_mesh, t_proj, t_mass)
    flush(stdout)
    eulers = [(0.0, acos(u), 0.0) for u in range(0.0, 1.0, length = L_ORI)]

    # Compile every path on a small problem before timing.
    let pw = GREParams(0.05, BC, 1.0, 0.0)
        mw, _ = gre_mesh(pw, MersenneTwister(1); lc = 0.3 * gre_semi_axes(pw)[3])
        bw = build_swg_basis(mw; include_boundary_faces = true)
        solve_cas_v2_orientations(bw, eulers[1:2]; wl_0 = WL_0, m_m = M_M, m_p = hi,
                                  return_solve_info = true)
        solve_cas_v2_orientations(bw, eulers[1:2]; wl_0 = WL_0, m_m = M_M, m_p = hi,
                                  precond = :mass_right, return_solve_info = true)
        solve_cas_v2_orientations_shifted(bw, eulers[1:2]; wl_0 = WL_0, m_m = M_M,
                                          m_p_list = [lo, hi], verify = :all)
    end

    # ── Methods B and P: one solve per distinct refractive index ──
    function independent(ms::Vector{ComplexF64}, precond::Symbol)
        out = Dict{Int,Any}()
        for (i, m) in enumerate(ms)
            RSSMonitor.reset!(mon)
            t = @elapsed res, info = solve_cas_v2_orientations(
                basis, eulers; wl_0 = WL_0, m_m = M_M, m_p = m, method = :aim_gmres,
                tol = TOL, maxiter = MAXITER, pitch = pitch, padding = 4,
                projection = proj, mass = mass, precond = precond,
                return_solve_info = true)
            out[i] = (res = res, t = t, iters = info.iterations, converged = info.converged,
                      residual = info.residual_norm, peak_rss = RSSMonitor.peak_bytes(mon))
            @printf("  %s m = %.4f%+.4fi  %7.1f s  %3d it%s\n", precond === :none ? "B" : "P",
                    real(m), imag(m), t, info.iterations, info.converged ? "" : "  NOT CONVERGED")
            flush(stdout)
        end
        return out
    end
    distinct(nms) = unique(x -> round(x; digits = 12), reduce(vcat, [m_grid(n) for n in nms];
                                                               init = ComplexF64[]))
    m_B = distinct(B_NM)
    m_P = distinct(P_NM)
    println("method B: ", length(m_B), " distinct refractive indices")
    res_B = independent(m_B, :none)
    println("method P: ", length(m_P), " distinct refractive indices")
    res_P = independent(m_P, :mass_right)

    # ── Method C: one shifted process per sweep size ──
    rows = Any[]
    for nm in NM_LIST
        ms = m_grid(nm)
        RSSMonitor.reset!(mon)
        t_C = @elapsed res_C, info = solve_cas_v2_orientations_shifted(
            basis, eulers; wl_0 = WL_0, m_m = M_M, m_p_list = ms,
            tol = TOL, maxiter = MAXITER, pitch = pitch, padding = 4,
            projection = proj, mass = mass, verify = :hardest)
        peak_C = RSSMonitor.peak_bytes(mon)
        tm = info.timing
        row = Dict{String,Any}(
            "n_m" => nm,
            "m_re" => real.(ms), "m_im" => imag.(ms),
            "C_t_total" => t_C, "C_t_rhs" => tm.rhs, "C_t_operator" => tm.operator,
            "C_t_factor" => tm.factor, "C_t_krylov" => tm.krylov,
            "C_t_verify" => tm.verify, "C_t_post" => tm.post,
            "C_iters" => info.iterations, "C_converged" => info.converged,
            "C_residual" => info.residual_norm,
            "C_true_residual_max" => maximum(filter(!isnan, info.true_residual)),
            "C_n_block_iterations" => info.n_block_iterations,
            "C_peak_rss" => peak_C)
        @printf("C N_m = %2d  total %7.1f s (operator %.1f, Cholesky %.2f, Krylov %.1f, verify %.1f, post %.1f)  iterations %d..%d  peak RSS %.2f GB\n",
                nm, t_C, tm.operator, tm.factor, tm.krylov, tm.verify, tm.post,
                minimum(info.iterations), maximum(info.iterations), peak_C / 1024^3)
        for (label, nms, m_ref, ref) in (("B", B_NM, m_B, res_B), ("P", P_NM, m_P, res_P))
            nm in nms || continue
            idx = [find_m(m_ref, m) for m in ms]
            t_ref = sum(ref[i].t for i in idx)
            its = [ref[i].iters for i in idx]
            diff = maximum(fw_diff(res_C[s, j], ref[idx[s]].res[j])
                           for s in 1:nm, j in 1:L_ORI)
            row["$(label)_t_total"] = t_ref
            row["$(label)_iters"] = its
            row["$(label)_converged"] = [ref[i].converged for i in idx]
            row["$(label)_peak_rss"] = maximum(ref[i].peak_rss for i in idx)
            row["$(label)_vs_C_max_fw_diff"] = diff
            row["T_$(label)_over_T_C"] = (t_shared + t_ref) / (t_shared + t_C)
            @printf("  %s N_m = %2d  total %7.1f s  iterations %d..%d (sum %d)  | (T_shared + T_%s)/(T_shared + T_C) = %.2f  | max forward-amplitude difference %.2e\n",
                    label, nm, t_ref, minimum(its), maximum(its), sum(its), label,
                    (t_shared + t_ref) / (t_shared + t_C), diff)
        end
        push!(rows, row)
        flush(stdout)
    end
    RSSMonitor.stop!(mon)

    save_json("shift_scaling_$(TAG).json",
              (host = gethostname(), threads = Threads.nthreads(),
               blas_threads = BLAS.get_num_threads(), date = string(now()),
               family = FAMILY, m_lo = (real(lo), imag(lo)), m_hi = (real(hi), imag(hi)),
               wl_0 = WL_0, m_m = M_M, d_ve = D_VE, bc_ratio = BC, L = L_ORI,
               tol = TOL, maxiter = MAXITER, lc = lc, n_tet = n_tets(mesh), n_dof = N,
               aim_grid = collect(grid.dims), t_mesh = t_mesh, t_proj = t_proj,
               t_mass = t_mass, t_shared = t_shared, rows = rows))
end

main()
