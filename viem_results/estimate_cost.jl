# Pre-run resource estimator for block-VIEM.jl paper sweeps.
#
# Reads the same HDF5 file consumed by run_viem.jl (created by create_h5.jl
# or one of the per-paper variants under viem_results/paper/) and prints
# estimated N_DOF, peak RSS, setup time, and end-to-end wall time per
# shape slot, plus the sweep totals. Warns if any single condition exceeds
# 24 h of wall time or if peak RSS exceeds the machine's available memory.
#
# How it works:
#   For each shape slot (i_rv, i_bc, i_ab, i_bt) we actually build the
#   tetrahedral mesh with the worst-case (smallest λ₀, largest |m_p|)
#   adaptive_lc — the mesh build is fast (~1 s for moderate sizes) but
#   pins down N_DOF exactly.  Time and memory are then projected from
#   per-DOF empirical constants calibrated on the README phase-A doublet
#   benchmark (Au, R = 0.030 μm, lc = R/5, N_DOF = 11501, t_setup ≈ 105 s,
#   RSS peak ≈ 1.2 GB).
#
# Usage:
#   julia --project=. -t auto viem_results/estimate_cost.jl <sweep.hdf5>
#
# Tunable env vars (optional):
#   N_ITER_EST       assumed BiCGSTAB iterations per RHS block  (default 200)
#   T_SETUP_MS_DOF   setup time per DOF, milliseconds            (default  9.0)
#   T_ITER_MS_DOF    per-iteration cost per DOF, milliseconds    (default  0.20)
#   RSS_KB_PER_DOF   peak RSS per DOF, kilobytes                 (default 100.0)
#   MEM_LIMIT_GB     machine memory limit override               (default auto)

using BlockVIEM
using BlockVIEM: GREParams, gre_mesh, build_swg_basis, n_basis, n_tets,
                 SphereAggregate, mesh_sphere_aggregate, total_volume
using HDF5
using Printf
using Random

# ──────────────────────────────────────────────────────────────────────
#  Tunable constants (env-overridable)
# ──────────────────────────────────────────────────────────────────────
const N_ITER_EST     = parse(Int,     get(ENV, "N_ITER_EST",     "200"))
# Defaults calibrated from the 2026-04-21 pilot:
# sphere_n317 a_eq=0.1 μm, N_DOF=5644, multi-threaded (julia -t auto).
# Same constants apply to n20 (paper "high" since v0.7.5):
#   t_setup ≈ 0.9 s   → 0.16 ms/DOF  (set to 0.3 with ~2× safety margin)
#   t_solve ≈ 16.9 s for L=5         → 0.01 ms/DOF/iter (assuming ~100 iters,
#                                       4× safety margin)
# Override via env vars when calibrating against a heavier slot.
const T_SETUP_MS_DOF = parse(Float64, get(ENV, "T_SETUP_MS_DOF", "0.3"))
const T_ITER_MS_DOF  = parse(Float64, get(ENV, "T_ITER_MS_DOF",  "0.01"))
const RSS_KB_PER_DOF = parse(Float64, get(ENV, "RSS_KB_PER_DOF", "100.0"))

# Match run_viem.jl mesh construction
const RNG_SEED = 12345
const N_PW     = 10

const ESCALATION_HOURS = 24.0   # warn if any single slot exceeds this

# ──────────────────────────────────────────────────────────────────────
#  Machine memory
# ──────────────────────────────────────────────────────────────────────
function machine_mem_gb()
    if haskey(ENV, "MEM_LIMIT_GB")
        return parse(Float64, ENV["MEM_LIMIT_GB"])
    end
    try
        for line in eachline("/proc/meminfo")
            if startswith(line, "MemAvailable:")
                kb = parse(Float64, split(line)[2])
                return kb / 1024 / 1024
            end
        end
    catch
        # fall through
    end
    return NaN
end

# ──────────────────────────────────────────────────────────────────────
#  Doublet aggregate aligned along particle z (matches run_viem.jl)
# ──────────────────────────────────────────────────────────────────────
function _doublet_along_z(R::Real)
    gap = 0.1 * R
    step = 2.0 * R + gap
    centers = zeros(Float64, 3, 2)
    centers[3, 1] = -step / 2
    centers[3, 2] = +step / 2
    radii = Float64[R, R]
    return SphereAggregate(centers, radii)
end

# ──────────────────────────────────────────────────────────────────────
#  Build mesh for one shape slot (worst-case wl_0 / m_p)
# ──────────────────────────────────────────────────────────────────────
function build_shape_mesh(shape_kind, r_v_base, bc_ratio, ab_ratio, gre_beta,
                          wl_0_min, m_p_max_global)
    rng = Random.MersenneTwister(RNG_SEED)
    if shape_kind == "doublet"
        R = Float64(r_v_base) / 2.0^(1/3)
        agg = _doublet_along_z(R)
        mesh, _msh = mesh_sphere_aggregate(agg;
                                           wl_0=wl_0_min,
                                           m_p_max=Float64(m_p_max_global),
                                           N_pw=N_PW)
        V = total_volume(mesh)
        r_ve = (3 * V / (4π))^(1/3)
    else
        p   = GREParams(r_v_base, bc_ratio, ab_ratio, gre_beta)
        mesh, r_ve = gre_mesh(p, rng;
                              wl_0=wl_0_min,
                              m_p_max=Float64(m_p_max_global),
                              N_pw=N_PW)
    end
    basis = build_swg_basis(mesh; include_boundary_faces=true)
    return mesh, basis, r_ve
end

# ──────────────────────────────────────────────────────────────────────
#  Cost model
# ──────────────────────────────────────────────────────────────────────
struct SlotEstimate
    shape_idx::NTuple{4,Int}
    r_v::Float64
    bc::Float64
    ab::Float64
    beta::Float64
    n_tet::Int
    n_dof::Int
    n_orient::Int
    spheroid_mode::Bool
    n_solve_orient::Int   # spheroid mode solves only N_beta
    rss_gb::Float64
    t_setup_s::Float64
    t_solve_s::Float64    # all orientations for this slot
    t_per_orient_s::Float64
    t_total_s::Float64
    n_inner_loop::Int     # N_pairs × N_m_p (sweeps share the same shape mesh)
end

function estimate_slot(shape_kind, shape_idx, r_v, bc, ab, beta,
                       n_tet, n_dof,
                       n_alpha, n_beta_ori, n_gamma,
                       n_inner_loop)
    rss_gb     = n_dof * RSS_KB_PER_DOF / 1024 / 1024
    t_setup_s  = n_dof * T_SETUP_MS_DOF / 1000
    spheroid_mode  = shape_kind == "doublet" ||
                     ((ab == 1.0) && (beta == 0.0))
    n_orient_total = n_alpha * n_beta_ori * n_gamma
    n_solve_orient = spheroid_mode ? n_beta_ori : n_orient_total
    # block-Krylov: per-iteration cost ≈ T_ITER_MS_DOF × N_DOF, scaled by
    # the number of RHS via FFT batching (sub-linear in practice).  We
    # use a conservative L^0.7 scaling for the iteration cost.
    L_block        = n_solve_orient
    block_factor   = max(1.0, L_block ^ 0.7)
    t_solve_s      = N_ITER_EST * T_ITER_MS_DOF * n_dof / 1000 * block_factor
    t_per_orient_s = t_solve_s / max(1, n_solve_orient)
    # The shape mesh + projection is built once; the inner (i_pair, i_mp)
    # loop reuses it but each entry still re-runs the solve.
    t_total_s      = t_setup_s + n_inner_loop * t_solve_s
    return SlotEstimate(shape_idx, r_v, bc, ab, beta,
                        n_tet, n_dof, n_orient_total,
                        spheroid_mode, n_solve_orient,
                        rss_gb, t_setup_s, t_solve_s,
                        t_per_orient_s, t_total_s, n_inner_loop)
end

# ──────────────────────────────────────────────────────────────────────
#  Pretty printers
# ──────────────────────────────────────────────────────────────────────
fmt_time(s) = s < 60   ? @sprintf("%.1fs",  s)        :
              s < 3600 ? @sprintf("%.1fm", s/60)      :
              s < 86400 ? @sprintf("%.2fh", s/3600)   :
                          @sprintf("%.2fd", s/86400)
fmt_gb(g)   = g < 1.0  ? @sprintf("%.0f MB", g*1024)  : @sprintf("%.2f GB", g)

function print_header()
    println("─"^110)
    @printf("%-13s %8s %8s %8s %8s %7s %7s %8s %10s %10s %10s\n",
        "shape_idx", "r_v(μm)", "bc", "ab", "β_gre",
        "N_tet", "N_DOF", "RSS",
        "t_setup", "t_per_ori", "t_total")
    println("─"^110)
end

function print_row(s::SlotEstimate; warn_total::Bool=false, warn_rss::Bool=false)
    flag = (warn_total || warn_rss) ? " ⚠️" : ""
    @printf("%-13s %8.4f %8.2f %8.2f %8.2f %7d %7d %8s %10s %10s %10s%s\n",
        string(s.shape_idx), s.r_v, s.bc, s.ab, s.beta,
        s.n_tet, s.n_dof, fmt_gb(s.rss_gb),
        fmt_time(s.t_setup_s), fmt_time(s.t_per_orient_s),
        fmt_time(s.t_total_s), flag)
end

# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
function main()
    length(ARGS) >= 1 || error("Usage: julia --project=. " *
                               "viem_results/estimate_cost.jl <sweep.hdf5>")
    h5path = ARGS[1]
    isfile(h5path) || error("HDF5 file not found: $h5path")

    mem_gb = machine_mem_gb()

    println("="^110)
    println("block-VIEM.jl  pre-run estimator")
    println("file       : $h5path")
    println("calib (envs): N_ITER_EST=$N_ITER_EST  T_SETUP_MS_DOF=$T_SETUP_MS_DOF  " *
            "T_ITER_MS_DOF=$T_ITER_MS_DOF  RSS_KB_PER_DOF=$RSS_KB_PER_DOF")
    println("machine    : MemAvailable ≈ $(isnan(mem_gb) ? "?" : @sprintf("%.1f GB", mem_gb))")
    println("escalation : any slot t_total > $(ESCALATION_HOURS) h or " *
            "RSS > $(@sprintf("%.0f%%", 100*0.9)) of MemAvailable → flagged ⚠️")
    println("="^110)

    h5open(h5path, "r") do f
        t = f["target"]

        n_alpha = Int(read_attribute(t, "N_alpha_ori"))
        n_beta  = Int(read_attribute(t, "N_beta_ori"))
        n_gamma = Int(read_attribute(t, "N_gamma_ori"))
        shape_kind = haskey(attributes(t), "shape_kind") ?
            String(read_attribute(t, "shape_kind")) : "gre"

        wl_pairs = read(t["wl_m_m_pairs"])
        m_p_list = read(t["m_p_xyz_list"])
        r_v_list = read(t["r_v_base_list"])
        bc_list  = read(t["bc_ratio_list"])
        ab_list  = read(t["ab_ratio_list"])
        bt_list  = read(t["gre_beta_list"])

        n_pairs = size(wl_pairs, 1)
        n_mp    = size(m_p_list, 1)
        n_rv    = length(r_v_list)
        n_bc    = length(bc_list)
        n_ab    = length(ab_list)
        n_bt    = length(bt_list)

        wl_min  = minimum(wl_pairs[:, 1])
        m_p_max = maximum(abs.(vec(m_p_list)))
        n_inner = n_pairs * n_mp

        println("sweep      : shape_kind=$shape_kind  $n_pairs wl-pairs × " *
                "$n_mp m_p × $n_rv r_v × $n_bc bc × $n_ab ab × $n_bt β")
        println("orientations: N_α=$n_alpha N_β=$n_beta N_γ=$n_gamma " *
                "(L=$(n_alpha*n_beta*n_gamma))")
        println("worst-case : wl_0_min=$(@sprintf("%.4f", wl_min)) μm  " *
                "|m_p|_max=$(@sprintf("%.4f", m_p_max))")
        println()

        print_header()

        slots = SlotEstimate[]
        any_warn = false

        for i_rv in 1:n_rv, i_bc in 1:n_bc, i_ab in 1:n_ab, i_bt in 1:n_bt
            r_v  = r_v_list[i_rv]
            bc   = bc_list[i_bc]
            ab   = ab_list[i_ab]
            beta = bt_list[i_bt]

            mesh, basis, _ = build_shape_mesh(shape_kind, r_v, bc, ab, beta,
                                              wl_min, m_p_max)
            ntet = n_tets(mesh)
            ndof = n_basis(basis)

            s = estimate_slot(shape_kind, (i_rv,i_bc,i_ab,i_bt),
                              r_v, bc, ab, beta,
                              ntet, ndof,
                              n_alpha, n_beta, n_gamma,
                              n_inner)

            warn_total = s.t_total_s > ESCALATION_HOURS * 3600
            warn_rss   = !isnan(mem_gb) && s.rss_gb > 0.9 * mem_gb
            any_warn   = any_warn || warn_total || warn_rss

            push!(slots, s)
            print_row(s; warn_total=warn_total, warn_rss=warn_rss)
        end

        println("─"^110)
        # Sweep totals
        # Wall time: setup happens once per shape slot in run_viem.jl, then
        # the inner (i_pair, i_mp) loop iterates n_inner times per shape.
        # Total wall time = sum(t_total_s) over slots.  Peak RSS = max of slots.
        t_sweep = sum(s.t_total_s for s in slots)
        rss_peak = maximum(s.rss_gb for s in slots)
        println("SWEEP TOTAL: $(length(slots)) shape slots × " *
                "$n_inner (wl × m_p) inner pts each")
        @printf("  estimated wall time      = %s\n", fmt_time(t_sweep))
        mem_str = isnan(mem_gb) ? "?" : @sprintf("%.1f GB", mem_gb)
        @printf("  estimated peak RSS       = %s  (machine: %s)\n",
                fmt_gb(rss_peak), mem_str)

        if any_warn
            println()
            println("⚠️  Some slots exceed the escalation threshold " *
                    "($(ESCALATION_HOURS) h or 90% of MemAvailable).")
            println("   Confirm with the user before launching run_viem.jl.")
        else
            println()
            println("✓ All slots within escalation thresholds.")
        end
        println("="^110)
    end
end

main()
