# RHS-scaling diagnostic for block-Krylov (CLAUDE.md §6).
#
# For every shape slot in a paper sweep HDF5, run BOTH block-BiCGSTAB
# and block-GMRES against L ∈ L_LIST distinct orientations and record
# the iteration count + wall time.  Results are written under
# /target/rhs_scaling/ as
#
#   L_values:        (nL,)                                Int   # [1, 2, 4, 8, 16, 32, 64, 128]
#   n_dof:           (N_rv, N_bc, N_ab, N_bt)             Int   # shared across methods
#   n_tet:           (N_rv, N_bc, N_ab, N_bt)             Int   # shared across methods
#   bicgstab/
#     iters:                  (nL, N_rv, N_bc, N_ab, N_bt)   Int
#     converged:              (nL, N_rv, N_bc, N_ab, N_bt)   Int  (0/1)
#     t_total_s:              (nL, N_rv, N_bc, N_ab, N_bt)   Float64
#     t_end2end_per_orient_s: (nL, N_rv, N_bc, N_ab, N_bt)   Float64
#   gmres/
#     (same datasets as bicgstab/)
#
# This is a *diagnostic* — it does NOT change any of the production data
# slots (C_ext, S_fw, etc.).  Run on the same HDF5 used for the main
# sweep (or a copy) any time after the sweep completes.
#
# Usage:
#   julia --project=. -t auto viem_results/paper/run_rhs_scaling.jl <sweep.hdf5>

using BlockVIEM
using BlockVIEM: GREParams, gre_mesh, build_swg_basis, n_basis, n_tets,
                 SphereAggregate, mesh_sphere_aggregate, total_volume,
                 mean_edge_length, aim_grid, build_aim_projection,
                 assemble_mass_matrix, solve_cas_v2_orientations,
                 duffy_reference_rule
using HDF5
using Printf
using Random
using Dates

include(joinpath(dirname(@__DIR__), "rss_monitor.jl"))
using .RSSMonitor

# ──────────────────────────────────────────────────────────────────────
#  Settings (mirror viem_results/run_viem.jl)
# ──────────────────────────────────────────────────────────────────────
const RNG_SEED        = 12345
const SOLVER_TOL      = 1e-5
const MAXITER         = 100
const N_PW            = 10
const DUFFY_ORDER     = 5
const AIM_PITCH_RATIO = 0.5
const AIM_PADDING     = 4
const L_LIST          = [1, 2, 4, 8, 16, 32, 64, 128]   # CLAUDE.md §6
# Per-method (subgroup name, solver-method symbol)
const METHODS         = ((:bicgstab, :aim_bicgstab), (:gmres, :aim_gmres))

# ──────────────────────────────────────────────────────────────────────
#  Helpers (kept in sync with viem_results/run_viem.jl::build_particle)
# ──────────────────────────────────────────────────────────────────────
function _doublet_along_z(R::Real)
    gap = 0.1 * R
    step = 2.0 * R + gap
    centers = zeros(Float64, 3, 2)
    centers[3, 1] = -step / 2
    centers[3, 2] = +step / 2
    return SphereAggregate(centers, Float64[R, R])
end

function build_particle(shape_kind, r_v_base, bc_ratio, ab_ratio, gre_beta,
                        wl_0, m_p_xyz)
    rng = Random.MersenneTwister(RNG_SEED)
    m_p_max = maximum(abs.(m_p_xyz))
    if shape_kind == "doublet"
        R = Float64(r_v_base) / 2.0^(1/3)
        agg = _doublet_along_z(R)
        mesh, _msh = mesh_sphere_aggregate(agg;
                                           wl_0=wl_0, m_p_max=m_p_max, N_pw=N_PW)
        V = total_volume(mesh)
        r_ve = (3 * V / (4π))^(1/3)
    else
        p = GREParams(r_v_base, bc_ratio, ab_ratio, gre_beta)
        mesh, r_ve = gre_mesh(p, rng; wl_0=wl_0, m_p_max=m_p_max, N_pw=N_PW)
    end
    basis = build_swg_basis(mesh; include_boundary_faces=true)
    return mesh, basis, r_ve
end

# ──────────────────────────────────────────────────────────────────────
#  Quasi-random orientations (deterministic via fixed RNG seed; nested
#  so larger L includes the smaller L's orientations as a prefix)
# ──────────────────────────────────────────────────────────────────────
function pick_orientations(L::Int)
    rng = Random.MersenneTwister(RNG_SEED)
    eulers = NTuple{3,Float64}[]
    for _ in 1:L
        α = 2π * rand(rng)
        β = acos(2 * rand(rng) - 1)            # uniform on sphere
        γ = 2π * rand(rng)
        push!(eulers, (α, β, γ))
    end
    return eulers
end

# ──────────────────────────────────────────────────────────────────────
#  Per-(shape, method, L) measurement
# ──────────────────────────────────────────────────────────────────────
function measure_one(basis, mesh, projection, mass, method_sym, L,
                     wl_0, m_m, m_p_xyz, mon::RSSMonitor.Monitor)
    eulers = pick_orientations(L)
    pitch  = AIM_PITCH_RATIO * mean_edge_length(mesh)

    RSSMonitor.reset!(mon)
    t0 = time_ns()
    _results, solve_info = solve_cas_v2_orientations(
        basis, eulers;
        wl_0=wl_0, m_m=m_m, m_p=m_p_xyz,
        method=method_sym,
        tol=SOLVER_TOL, maxiter=MAXITER,
        verbose=false,
        return_solve_info=true,
        projection=projection, mass=mass,
        pitch=pitch, padding=AIM_PADDING,
        duffy_rule=duffy_reference_rule(DUFFY_ORDER),
        symmetrize=true)
    t_total  = (time_ns() - t0) / 1e9
    peak_rss = RSSMonitor.peak_bytes(mon)

    return (
        iters            = solve_info.iterations,
        converged        = solve_info.converged,
        t_total          = t_total,
        t_per_orient     = t_total / L,
        peak_rss         = peak_rss,
        residual_history = Vector{Float64}(solve_info.residual_history),
    )
end

# ──────────────────────────────────────────────────────────────────────
#  HDF5 writer (replaces /target/rhs_scaling/ if present)
# ──────────────────────────────────────────────────────────────────────
function write_rhs_scaling(t, per_method, n_dof_arr, n_tet_arr)
    if haskey(t, "rhs_scaling")
        delete_object(t, "rhs_scaling")
    end
    g = create_group(t, "rhs_scaling")
    attrs(g)["description"] =
        "Block-Krylov scaling diagnostic over L ∈ L_values, per method " *
        "(bicgstab, gmres). Per-orientation sets are nested " *
        "(L=1 ⊂ L=2 ⊂ … ⊂ L=128), drawn from a " *
        "fixed-seed uniform-sphere Euler sequence."
    attrs(g)["units"]           = "t:[s]"
    attrs(g)["solver_tol"]      = SOLVER_TOL
    attrs(g)["solver_maxiter"]  = MAXITER
    attrs(g)["aim_pitch_ratio"] = AIM_PITCH_RATIO
    attrs(g)["aim_padding"]     = AIM_PADDING

    write_dataset(g, "L_values", Int.(L_LIST))
    write_dataset(g, "n_dof",    n_dof_arr)
    write_dataset(g, "n_tet",    n_tet_arr)

    for (subgroup_name, m) in per_method
        sg = create_group(g, String(subgroup_name))
        attrs(sg)["solver_method"] = String(m.method_sym)
        write_dataset(sg, "iters",                   m.iters_arr)
        write_dataset(sg, "converged",               m.conv_arr)
        write_dataset(sg, "t_total_s",               m.t_total_arr)
        write_dataset(sg, "t_end2end_per_orient_s",  m.t_per_arr)
        write_dataset(sg, "peak_rss_bytes",          m.peak_rss_arr)
        d_hist = create_dataset(sg, "residual_history", Float64, size(m.hist_arr))
        attrs(d_hist)["definition"] =
            "per-iteration relative residual, shape " *
            "(nL, N_rv, N_bc, N_ab, N_bt, MAXITER). " *
            "NaN-padded beyond iter_fin.  Enables convergence-profile " *
            "figures symmetric with block-DDA_Py rhs_scaling output."
        write(d_hist, m.hist_arr)
    end
end

# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
_log(msg) = println("[$(Dates.format(now(), "HH:MM:SS"))] $msg")

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=. " *
                               "viem_results/paper/run_rhs_scaling.jl <sweep.hdf5>")
    h5path = ARGS[1]
    isfile(h5path) || error("HDF5 not found: $h5path")

    _log("Opening $h5path  (rhs-scaling diagnostic)")
    h5open(h5path, "r+") do f
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

        n_rv = length(r_v_list); n_bc = length(bc_list)
        n_ab = length(ab_list); n_bt = length(bt_list)

        # Worst-case mesh (matches viem_results/run_viem.jl reuse mode)
        wl_min  = minimum(wl_pairs[:, 1])
        m_p_max = maximum(abs.(vec(m_p_list)))
        m_p_worst = fill(ComplexF64(m_p_max), 3)

        # Use first (wl_0, m_m, m_p) entry as the physical operating point
        # for the diagnostic (paper HDF5s have a single point anyway)
        wl_0_op = wl_pairs[1, 1]
        m_m_op  = wl_pairs[1, 2]
        m_p_op  = ComplexF64.(m_p_list[1, :])

        _log("shape_kind=$shape_kind  worst-case wl_0=$(round(wl_min, digits=4)) " *
             "|m_p|_max=$(round(m_p_max, digits=4))")
        _log("operating point: wl_0=$(round(wl_0_op, digits=4))  " *
             "m_m=$(round(m_m_op, digits=4))  m_p=$m_p_op")
        _log("L_LIST = $L_LIST  methods = $([Symbol(s) for (s,_) in METHODS])  " *
             "tol=$SOLVER_TOL  maxiter=$MAXITER")

        nL = length(L_LIST)
        n_dof_arr = zeros(Int, n_rv, n_bc, n_ab, n_bt)
        n_tet_arr = zeros(Int, n_rv, n_bc, n_ab, n_bt)

        # Per-method storage (one tuple of arrays per (subgroup_name, method_sym))
        per_method = [(name, (
            method_sym    = sym,
            iters_arr     = zeros(Int,     nL, n_rv, n_bc, n_ab, n_bt),
            conv_arr      = zeros(Int,     nL, n_rv, n_bc, n_ab, n_bt),
            t_total_arr   = zeros(Float64, nL, n_rv, n_bc, n_ab, n_bt),
            t_per_arr     = zeros(Float64, nL, n_rv, n_bc, n_ab, n_bt),
            peak_rss_arr  = zeros(Int64,   nL, n_rv, n_bc, n_ab, n_bt),
            hist_arr      = fill(NaN,      nL, n_rv, n_bc, n_ab, n_bt, MAXITER),
        )) for (name, sym) in METHODS]

        # Peak-RSS sampler shared across all (shape, method, L) measurements.
        mon = RSSMonitor.Monitor(0.2)
        RSSMonitor.start!(mon)

        for i_rv in 1:n_rv, i_bc in 1:n_bc, i_ab in 1:n_ab, i_bt in 1:n_bt
            r_v = r_v_list[i_rv]
            bc  = bc_list[i_bc]
            ab  = ab_list[i_ab]
            bt  = bt_list[i_bt]

            _log("─"^70)
            _log("shape=($i_rv,$i_bc,$i_ab,$i_bt)  r_v=$r_v  bc=$bc  ab=$ab  β=$bt")

            # Build worst-case mesh, projection, mass (shared across all L and methods)
            t_mesh = @elapsed begin
                mesh, basis, r_ve = build_particle(
                    shape_kind, r_v, bc, ab, bt, wl_min, m_p_worst)
                pitch = AIM_PITCH_RATIO * mean_edge_length(mesh)
                grid       = aim_grid(basis.mesh; pitch=pitch, padding=AIM_PADDING)
                projection = build_aim_projection(basis, grid; poly_order=2, stencil=3)
                mass       = assemble_mass_matrix(basis)
            end
            n_dof_arr[i_rv,i_bc,i_ab,i_bt] = n_basis(basis)
            n_tet_arr[i_rv,i_bc,i_ab,i_bt] = n_tets(mesh)
            _log("  mesh+projection+mass  N_tet=$(n_tets(mesh))  " *
                 "N_DOF=$(n_basis(basis))  ($(round(t_mesh, digits=2))s)")

            for (subgroup_name, m_arrays) in per_method
                _log("  method = $subgroup_name")
                for (iL, L) in enumerate(L_LIST)
                    m = measure_one(basis, mesh, projection, mass,
                                    m_arrays.method_sym,
                                    L, wl_0_op, m_m_op, m_p_op, mon)
                    m_arrays.iters_arr[iL,i_rv,i_bc,i_ab,i_bt]    = m.iters
                    m_arrays.conv_arr[iL,i_rv,i_bc,i_ab,i_bt]     = m.converged ? 1 : 0
                    m_arrays.t_total_arr[iL,i_rv,i_bc,i_ab,i_bt]  = m.t_total
                    m_arrays.t_per_arr[iL,i_rv,i_bc,i_ab,i_bt]    = m.t_per_orient
                    m_arrays.peak_rss_arr[iL,i_rv,i_bc,i_ab,i_bt] = Int64(m.peak_rss)
                    # residual_history, NaN-padded to MAXITER length
                    eh = m.residual_history
                    m_eh = min(length(eh), MAXITER)
                    @views m_arrays.hist_arr[iL,i_rv,i_bc,i_ab,i_bt, 1:m_eh] .= eh[1:m_eh]
                    conv_str = m.converged ? "✓" : "✗"
                    _log(@sprintf("    L=%-3d  iters=%-4d  %s  t_total=%7.2fs  t/ori=%7.3fs  peak=%6.2f GB",
                                  L, m.iters, conv_str, m.t_total, m.t_per_orient,
                                  m.peak_rss / 2^30))
                    flush(stdout)
                end
            end
        end
        RSSMonitor.stop!(mon)

        write_rhs_scaling(t, per_method, n_dof_arr, n_tet_arr)
        _log("─"^70)
        _log("Wrote /target/rhs_scaling/ to $h5path")
    end
end

main()
