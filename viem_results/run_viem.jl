# Production parameter-sweep script for block-VIEM.jl.
#
# Julia equivalent of block-DDA_Py/run_dda.py. Reads an HDF5 file created
# by create_h5.jl and fills every condition slot with VIEM CAS-v2
# observables plus the Mie reference for the volume-equivalent sphere.
#
# Features:
#   - AIM FFT-MVP + block-BiCGSTAB by default (O(N log N) per iteration)
#   - Automatic spheroid-mode detection (ab_ratio==1 && gre_beta==0):
#     solve only N_beta orientations at α=0, fill the full grid analytically
#   - On solver failure, fills NaN and continues (matches block-DDA_Py)
#   - Resume from partially filled HDF5 (skips already-computed slots)
#   - Per-condition Mie reference calculation
#
# Usage:
#     julia --project=. -t auto viem_results/run_viem.jl [filename]
#
# The -t auto flag is recommended to enable multi-threaded assembly.

using BlockVIEM
using BlockVIEM: Vec3, SphereAggregate, mesh_sphere_aggregate
using StaticArrays
using LinearAlgebra
using HDF5
using Printf
using Random
using Dates

# Include Mie reference (test utility, not part of the package)
include(joinpath(dirname(@__DIR__), "test", "mie_reference.jl"))

# Per-slot peak-RSS sampler (daemon task reads /proc/self/status)
include(joinpath(@__DIR__, "rss_monitor.jl"))
using .RSSMonitor

# ══════════════════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════════════════
const RNG_SEED      = 12345       # GRE shape random seed (matches run_dda.py)
const SOLVER_TOL    = 1e-5        # solver tolerance (paper-production default)
const SOLVER_METHOD = :aim_gmres     # :aim_gmres (default, v0.7.1+) | :aim_bicgstab | :dense
const MAXITER       = 100         # max Krylov iterations per try
const N_PW          = 10          # mesh: points per wavelength
const DUFFY_ORDER   = 5           # Duffy quadrature order
const AIM_PITCH_RATIO = 0.5      # AIM grid pitch = ratio × mean edge length
const AIM_PADDING   = 4           # AIM grid padding (number of pitch cells)
# When true, a single worst-case mesh (built with the shortest wl_0 and
# largest |m_p| in the sweep) is used across every (wl_0, m_p) entry
# within the same shape_idx, and the k0-independent projection +
# mass matrices are built once per shape and reused.  Trades slightly
# over-refined meshes at long-wl / small-|m_p| points for a large
# setup speed-up across the inner (i_pair, i_mp) loop.  Set false to
# restore the original per-(wl_0, m_p) mesh sizing.
const REUSE_PROJECTION_PER_SHAPE = true
const OUTPUT_FILE   = length(ARGS) >= 1 ? ARGS[1] :
                      joinpath(@__DIR__, "pcas_ocbs_simulated_data.hdf5")

# ══════════════════════════════════════════════════════════════════════════════
#  Logging utility
# ══════════════════════════════════════════════════════════════════════════════
function _log(msg::AbstractString)
    println("[$(Dates.format(now(), "HH:MM:SS"))] $msg")
end

# ══════════════════════════════════════════════════════════════════════════════
#  Euler angle grid (deterministic, matches run_dda.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
function generate_euler_grid(N_alpha::Int, N_beta::Int, N_gamma::Int)
    alpha = range(0, 2π, length=N_alpha+1)[1:end-1]  # [0, 2π) endpoint=false
    cos_beta = range(1 - 1/N_beta, -1 + 1/N_beta, length=N_beta)
    beta = acos.(cos_beta)
    gamma = range(0, 2π, length=N_gamma+1)[1:end-1]

    L = N_alpha * N_beta * N_gamma
    euler = Vector{NTuple{3,Float64}}(undef, L)
    idx = 0
    for ia in 1:N_alpha
        for ib in 1:N_beta
            for ig in 1:N_gamma
                idx += 1
                euler[idx] = (alpha[ia], beta[ib], gamma[ig])
            end
        end
    end
    return euler
end

# ══════════════════════════════════════════════════════════════════════════════
#  Spheroid (axial-symmetry) detection
# ══════════════════════════════════════════════════════════════════════════════
# The α-expansion in `run_viem_spheroid` works for any particle that is
# cylindrically symmetric about its z-axis: GRE family with ab=1, β=0 OR
# a doublet placed along particle z (CLAUDE.md §2).
_is_spheroid(shape_kind, ab_ratio, gre_beta) =
    shape_kind == "doublet" || (ab_ratio == 1.0 && gre_beta == 0.0)

# ══════════════════════════════════════════════════════════════════════════════
#  Doublet aggregate aligned along particle z-axis
#  Equal-radius monomers, gap = 0.1 R between surfaces (CLAUDE.md §2),
#  doublet axis = particle z so the spheroid α-expansion applies.
# ══════════════════════════════════════════════════════════════════════════════
function _doublet_along_z(R::Real)
    gap = 0.1 * R                   # surface-to-surface gap (CLAUDE.md)
    step = 2.0 * R + gap
    centers = zeros(Float64, 3, 2)
    centers[3, 1] = -step / 2
    centers[3, 2] = +step / 2
    radii = Float64[R, R]
    metadata = Dict{String,Any}(
        "source"  => "run_viem.jl::_doublet_along_z",
        "axis"    => "z",
        "R"       => Float64(R),
        "gap"     => Float64(gap),
    )
    return SphereAggregate(centers, radii; metadata=metadata)
end

# ══════════════════════════════════════════════════════════════════════════════
#  Build mesh + SWG basis (dispatches on shape_kind)
# ══════════════════════════════════════════════════════════════════════════════
function build_particle(shape_kind, r_v_base, bc_ratio, ab_ratio, gre_beta,
                        wl_0, m_p_xyz)
    rng = Random.MersenneTwister(RNG_SEED)
    m_p_max = maximum(abs.(m_p_xyz))
    if shape_kind == "doublet"
        # a_eq = r_v_base; monomer radius R = a_eq / 2^(1/3) (CLAUDE.md §2)
        R = Float64(r_v_base) / 2.0^(1/3)
        agg = _doublet_along_z(R)
        mesh, _msh = mesh_sphere_aggregate(agg;
                                           wl_0=wl_0, m_p_max=m_p_max, N_pw=N_PW)
        # r_ve from actual discretised volume (small deviation from a_eq
        # due to mesh truncation of the sphere surfaces)
        V = total_volume(mesh)
        r_ve = (3 * V / (4π))^(1/3)
    else
        # GRE family
        p = GREParams(r_v_base, bc_ratio, ab_ratio, gre_beta)
        mesh, r_ve = gre_mesh(p, rng; wl_0=wl_0, m_p_max=m_p_max, N_pw=N_PW)
    end
    basis = build_swg_basis(mesh; include_boundary_faces=true)
    return mesh, basis, r_ve
end

# ══════════════════════════════════════════════════════════════════════════════
#  Common solve core: calls solve_cas_v2_orientations with return_D=true,
#  then computes cross sections from D_block.
# ══════════════════════════════════════════════════════════════════════════════
function _solve_and_postprocess(basis, euler_list, wl_0, m_m, m_p_xyz, mesh;
                                projection = nothing, mass = nothing)
    k0 = ComplexF64(2π * m_m / wl_0)
    eps_p = _to_eps(m_p_xyz)
    eps_bg = ComplexF64(m_m)^2
    num_ori = length(euler_list)

    # AIM pitch from mesh geometry (ignored when method == :dense or
    # when a pre-built `projection` is supplied — its grid pitch wins).
    h_bar = mean_edge_length(mesh)
    pitch = AIM_PITCH_RATIO * h_bar

    method_kw = Dict{Symbol,Any}(
        :method            => SOLVER_METHOD,
        :tol               => SOLVER_TOL,
        :maxiter           => MAXITER,
        :verbose           => false,
        :return_D          => true,
        :return_solve_info => true,
        :duffy_rule        => duffy_reference_rule(DUFFY_ORDER),
        :symmetrize        => true,
    )
    if SOLVER_METHOD !== :dense
        method_kw[:pitch]      = pitch
        method_kw[:padding]    = AIM_PADDING
        method_kw[:projection] = projection
        method_kw[:mass]       = mass
    end

    cas_results, D_block, solve_info = solve_cas_v2_orientations(
        basis, euler_list;
        wl_0=wl_0, m_m=m_m, m_p=m_p_xyz,
        method_kw...)

    # Extract CAS-v2 observables
    S_fw_th = [r.S_fw_theta for r in cas_results]
    S_fw_ph = [r.S_fw_phi   for r in cas_results]
    S_bk_v  = [r.S_bk       for r in cas_results]

    # Cross sections from D_block (farfield integration)
    orientations = [cas_orientation(ea...) for ea in euler_list]
    C_abs = Vector{Float64}(undef, num_ori)
    C_ext = Vector{Float64}(undef, num_ori)
    for i in 1:num_ori
        ori = orientations[i]
        sr = compute_scattering(basis, @view(D_block[:, i]);
                                k_hat=ori.u_inc, E0=ori.e0_inc,
                                k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                csca_method=:farfield)
        C_abs[i] = sr.C_abs
        C_ext[i] = sr.C_ext
    end

    return C_abs, C_ext, S_fw_th, S_fw_ph, S_bk_v, solve_info
end

# ══════════════════════════════════════════════════════════════════════════════
#  Solve: general particle (all orientations)
# ══════════════════════════════════════════════════════════════════════════════
function run_viem_general(basis, mesh, wl_0, m_m, m_p_xyz, euler_angles;
                          projection = nothing, mass = nothing)
    C_abs, C_ext, S_fw_th, S_fw_ph, S_bk, solve_info =
        _solve_and_postprocess(basis, euler_angles, wl_0, m_m, m_p_xyz, mesh;
                               projection = projection, mass = mass)
    return C_abs, C_ext, S_fw_th, S_fw_ph, S_bk, solve_info
end

# ══════════════════════════════════════════════════════════════════════════════
#  Solve: spheroid (α-expansion, solve only N_beta orientations)
# ══════════════════════════════════════════════════════════════════════════════
function run_viem_spheroid(basis, mesh, wl_0, m_m, m_p_xyz,
                           N_alpha, N_beta, N_gamma;
                           projection = nothing, mass = nothing)
    num_full = N_alpha * N_beta * N_gamma

    # Beta-only grid for solve (α=0, γ=0)
    cos_beta = range(1 - 1/N_beta, -1 + 1/N_beta, length=N_beta)
    beta_vals = acos.(cos_beta)
    euler_beta_only = [(0.0, β, 0.0) for β in beta_vals]

    C_abs_0, C_ext_0, S_s_0, S_p_0, S_bk_0, solve_info =
        _solve_and_postprocess(basis, euler_beta_only, wl_0, m_m, m_p_xyz, mesh;
                               projection = projection, mass = mass)

    _log("    [spheroid, L=$N_beta]: " *
         (solve_info.converged ? "converged ✓" : "NOT converged ✗") *
         "  iters=$(solve_info.iterations)")

    # Analytical α-expansion to full grid
    alpha_vals = collect(range(0, 2π, length=N_alpha+1)[1:end-1])

    C_abs   = Vector{Float64}(undef, num_full)
    C_ext   = Vector{Float64}(undef, num_full)
    S_fw_th = Vector{ComplexF64}(undef, num_full)
    S_fw_ph = Vector{ComplexF64}(undef, num_full)
    S_bk    = Vector{ComplexF64}(undef, num_full)

    idx = 0
    for ia in 1:N_alpha
        α = alpha_vals[ia]
        e2a = exp(2im * α)
        for ib in 1:N_beta
            A_fw = (S_s_0[ib] + S_p_0[ib]) / 2
            B_fw = (S_s_0[ib] - S_p_0[ib]) / 2
            s_fw_th_val = A_fw + B_fw * e2a
            s_fw_ph_val = A_fw - B_fw * e2a
            s_bk_val    = S_bk_0[ib] * e2a
            for _ in 1:N_gamma
                idx += 1
                C_abs[idx]   = C_abs_0[ib]
                C_ext[idx]   = C_ext_0[ib]
                S_fw_th[idx] = s_fw_th_val
                S_fw_ph[idx] = s_fw_ph_val
                S_bk[idx]    = s_bk_val
            end
        end
    end

    return C_abs, C_ext, S_fw_th, S_fw_ph, S_bk, solve_info
end

# ══════════════════════════════════════════════════════════════════════════════
#  Mie reference for volume-equivalent sphere
# ══════════════════════════════════════════════════════════════════════════════
function compute_mie_reference(wl_0, m_m, r_v_base, m_p_xyz)
    m_p_avg = ComplexF64(sum(m_p_xyz) / length(m_p_xyz))
    cs = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_v_base, m_p=m_p_avg)
    cas = mie_cas_observables(; wl_0=wl_0, m_m=m_m, r_p=r_v_base, m_p=m_p_avg)
    return cs.C_abs, cs.C_ext, cas.S_fw_mean, cas.S_bk
end

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
function _to_eps(m_p_xyz)
    if m_p_xyz isa Number
        return ComplexF64(m_p_xyz)^2
    else
        return ComplexF64.(m_p_xyz) .^ 2
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#  Main sweep
# ══════════════════════════════════════════════════════════════════════════════
function main()
    isfile(OUTPUT_FILE) || error("HDF5 file not found: $OUTPUT_FILE\n" *
        "Create it first with: julia --project=. viem_results/create_h5.jl")

    _log("Opening $OUTPUT_FILE")
    _log("Solver: $SOLVER_METHOD  tol=$SOLVER_TOL  maxiter=$MAXITER  " *
         "pitch_ratio=$AIM_PITCH_RATIO  padding=$AIM_PADDING")

    h5open(OUTPUT_FILE, "r+") do f
        t  = f["target"]
        sd = t["simulated_data"]
        # The /target/cost/ group is mandatory since v0.7.3 (symmetric
        # with block-DDA_Py).  Refuse to run on HDF5 files produced by
        # an older create_paper_h5.jl — they must be regenerated.
        haskey(t, "cost") || error(
            "HDF5 is missing the /target/cost/ group — regenerate via " *
            "viem_results/paper/<shape>_<material>.jl (v0.7.3+).")
        cost = t["cost"]

        # Peak-RSS sampler (daemon task); reset before each slot so the
        # recorded peak reflects just that slot's allocations.
        mon = RSSMonitor.Monitor(0.2)
        RSSMonitor.start!(mon)

        N_alpha = Int(read_attribute(t, "N_alpha_ori"))
        N_beta  = Int(read_attribute(t, "N_beta_ori"))
        N_gamma = Int(read_attribute(t, "N_gamma_ori"))
        num_orientations = N_alpha * N_beta * N_gamma
        # shape_kind: "gre" (default for legacy files) or "doublet"
        shape_kind = haskey(attributes(t), "shape_kind") ?
            String(read_attribute(t, "shape_kind")) : "gre"

        wl_m_m_pairs  = read(t["wl_m_m_pairs"])     # (N_pairs, 2) in Julia
        m_p_xyz_list  = read(t["m_p_xyz_list"])      # (N_m_p, 3)
        r_v_base_list = read(t["r_v_base_list"])
        bc_ratio_list = read(t["bc_ratio_list"])
        ab_ratio_list = read(t["ab_ratio_list"])
        gre_beta_list = read(t["gre_beta_list"])

        N_pairs = size(wl_m_m_pairs, 1)
        N_m_p   = size(m_p_xyz_list, 1)
        N_rv    = length(r_v_base_list)
        N_bc    = length(bc_ratio_list)
        N_ab    = length(ab_ratio_list)
        N_bt    = length(gre_beta_list)

        # Full Euler angle grid
        euler_angles = generate_euler_grid(N_alpha, N_beta, N_gamma)

        _log("Sweep: shape_kind=$shape_kind  $(N_pairs) wl-pairs × " *
             "$(N_m_p) m_p × $(N_rv) r_v × $(N_bc) bc × $(N_ab) ab × $(N_bt) β")
        _log("Orientation grid: N_α=$N_alpha, N_β=$N_beta, N_γ=$N_gamma " *
             "(L=$num_orientations)")

        n_done = 0
        n_skip = 0
        n_total = N_pairs * N_m_p * N_rv * N_bc * N_ab * N_bt

        # Worst-case mesh sizing for projection/mass reuse across the
        # inner (i_pair, i_mp) loop: gre_mesh's adaptive lc is
        # monotonically decreasing in `wl_0 / max|m_p|`, so using the
        # smallest `wl_0` and the largest `|m_p|` anywhere in the sweep
        # yields a mesh that resolves every (wl_0, m_p) slot.
        wl_0_min_sweep = minimum(wl_m_m_pairs[:, 1])
        m_p_max_global = maximum(abs.(vec(m_p_xyz_list)))
        if REUSE_PROJECTION_PER_SHAPE
            _log("Reuse mode: 1 mesh + projection + mass per shape_idx " *
                 "(worst-case wl_0=$(Printf.@sprintf("%.4f", wl_0_min_sweep)), " *
                 "|m_p|_max=$(Printf.@sprintf("%.4f", m_p_max_global)))")
        end

        for i_rv in 1:N_rv, i_bc in 1:N_bc, i_ab in 1:N_ab, i_bt in 1:N_bt
            r_v_base  = r_v_base_list[i_rv]
            bc_ratio  = bc_ratio_list[i_bc]
            ab_ratio  = ab_ratio_list[i_ab]
            gre_beta  = gre_beta_list[i_bt]

            shape_idx = (i_rv, i_bc, i_ab, i_bt)
            spheroid_mode = _is_spheroid(shape_kind, ab_ratio, gre_beta)

            # Write r_ve (geometric, same for all wl/m_p for this shape)
            sd["r_ve"][shape_idx...] = r_v_base

            # ── Per-shape mesh + projection + mass (reuse mode) ──────
            shape_mesh       = nothing
            shape_basis      = nothing
            shape_r_ve       = nothing
            shape_projection = nothing
            shape_mass       = nothing
            # Per-shape build / setup wall times — attributed in the
            # cost group to the first slot (i_pair=1, i_mp=1) that uses
            # the cached mesh/projection/mass.  Remain 0 for non-REUSE
            # mode (where per-slot build happens inside the inner loop).
            t_build_shape = 0.0
            t_setup_shape = 0.0
            if REUSE_PROJECTION_PER_SHAPE
                m_p_worst_xyz = fill(ComplexF64(m_p_max_global), 3)
                t_build_shape = @elapsed begin
                    shape_mesh, shape_basis, shape_r_ve = build_particle(
                        shape_kind, r_v_base, bc_ratio, ab_ratio, gre_beta,
                        wl_0_min_sweep, m_p_worst_xyz)
                end
                h_bar_shape = mean_edge_length(shape_mesh)
                pitch_shape = AIM_PITCH_RATIO * h_bar_shape
                _log("shape=$shape_idx  mesh: $(n_tets(shape_mesh)) tets, " *
                     "$(n_basis(shape_basis)) DOFs, " *
                     "r_ve=$(Printf.@sprintf("%.5f", shape_r_ve)), " *
                     "pitch=$(Printf.@sprintf("%.4f", pitch_shape)) " *
                     "($(Printf.@sprintf("%.1fs", t_build_shape)))")
                if SOLVER_METHOD !== :dense
                    t_setup_shape = @elapsed begin
                        grid = aim_grid(shape_basis.mesh;
                                         pitch = pitch_shape,
                                         padding = AIM_PADDING)
                        shape_projection = build_aim_projection(
                            shape_basis, grid; poly_order = 2, stencil = 3)
                        shape_mass = assemble_mass_matrix(shape_basis)
                    end
                    _log("  projection + mass cached " *
                         "($(Printf.@sprintf("%.1fs", t_setup_shape)))")
                end
                # r_ve from actual mesh volume (same across inner loop)
                sd["r_ve"][shape_idx...] = shape_r_ve
            end

            for i_pair in 1:N_pairs, i_mp in 1:N_m_p
                wl_0    = wl_m_m_pairs[i_pair, 1]
                m_m     = wl_m_m_pairs[i_pair, 2]
                m_p_xyz = ComplexF64.(m_p_xyz_list[i_mp, :])

                idx6 = (i_pair, i_mp, i_rv, i_bc, i_ab, i_bt)

                # ── Resume: skip if already computed ─────────────────────
                existing_mie = sd["S_fw_PCAS_mie"][idx6...]
                if imag(existing_mie) != 0.0
                    n_skip += 1
                    _log("Skip: idx=$idx6 (already computed) [$n_skip skipped]")
                    continue
                end

                println("─" ^ 64)
                mode_tag = spheroid_mode ? " [spheroid, L_solve=$N_beta]" : ""
                _log("idx=$idx6  wl_0=$(Printf.@sprintf("%.4f", wl_0)) μm  " *
                     "m_m=$(Printf.@sprintf("%.4f", m_m))  m_p=$m_p_xyz")
                _log("  r_v=$r_v_base  bc=$bc_ratio  ab=$ab_ratio  " *
                     "β=$gre_beta  N_ori=$num_orientations$mode_tag")

                # ── Per-slot cost accounting ─────────────────────────────
                RSSMonitor.reset!(mon)
                t_slot_start = time_ns()
                # Attribute shape-level build + setup to the FIRST slot
                # in this shape (REUSE mode); zero for the subsequent
                # slots that simply reuse the cached mesh / projection.
                t_build_slot = 0.0
                t_setup_slot = 0.0
                if REUSE_PROJECTION_PER_SHAPE && i_pair == 1 && i_mp == 1
                    t_build_slot = t_build_shape
                    t_setup_slot = t_setup_shape
                end

                # ── Build mesh + basis (unless reused from shape) ────────
                local mesh, basis, r_ve
                if REUSE_PROJECTION_PER_SHAPE
                    mesh, basis, r_ve = shape_mesh, shape_basis, shape_r_ve
                else
                    t_mesh = @elapsed begin
                        mesh, basis, r_ve = build_particle(
                            shape_kind, r_v_base, bc_ratio, ab_ratio, gre_beta,
                            wl_0, m_p_xyz)
                    end
                    t_build_slot = t_mesh     # per-slot mesh build
                    h_bar = mean_edge_length(mesh)
                    _log("  mesh: $(n_tets(mesh)) tets, $(n_basis(basis)) DOFs, " *
                         "r_ve=$(Printf.@sprintf("%.5f", r_ve)), " *
                         "h̄=$(Printf.@sprintf("%.4f", h_bar)), " *
                         "pitch=$(Printf.@sprintf("%.4f", AIM_PITCH_RATIO * h_bar)) " *
                         "($(Printf.@sprintf("%.1fs", t_mesh)))")
                    sd["r_ve"][shape_idx...] = r_ve
                end

                # ── Solve ────────────────────────────────────────────────
                # On failure, fill with NaN and continue to next condition
                # (matching block-DDA_Py behaviour). Only Ctrl+C stops the sweep.
                C_abs      = fill(NaN, num_orientations)
                C_ext      = fill(NaN, num_orientations)
                S_fw_theta = fill(NaN + NaN*im, num_orientations)
                S_fw_phi   = fill(NaN + NaN*im, num_orientations)
                S_bk       = fill(NaN + NaN*im, num_orientations)
                # Sentinel values used when the solver raises; overwritten
                # on success by the 6th return value of run_viem_*.
                solve_info = (iterations = 0, converged = false,
                              residual_norm = NaN)
                t0_solve = time_ns()
                try
                    if spheroid_mode
                        C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, solve_info =
                            run_viem_spheroid(basis, mesh, wl_0, m_m, m_p_xyz,
                                              N_alpha, N_beta, N_gamma;
                                              projection = shape_projection,
                                              mass = shape_mass)
                    else
                        C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, solve_info =
                            run_viem_general(basis, mesh, wl_0, m_m, m_p_xyz,
                                             euler_angles;
                                             projection = shape_projection,
                                             mass = shape_mass)
                    end
                catch e
                    if e isa InterruptException
                        _log("Interrupted — file closed cleanly.")
                        RSSMonitor.stop!(mon)
                        return
                    end
                    _log("  FAILED: $(sprint(showerror, e))")
                    _log("  filling with NaN and continuing")
                end
                t_solve = (time_ns() - t0_solve) / 1e9
                _log("  solve: $(Printf.@sprintf("%.1fs", t_solve))  " *
                     "iters=$(solve_info.iterations)  " *
                     "converged=$(solve_info.converged)")

                # ── Mie reference ────────────────────────────────────────
                C_abs_mie, C_ext_mie, S_fw_mean_mie, S_bk_mie =
                    compute_mie_reference(wl_0, m_m, r_v_base, m_p_xyz)

                # ── Write Euler angles ───────────────────────────────────
                euler_flat = Matrix{Float64}(undef, num_orientations, 3)
                for i in 1:num_orientations
                    euler_flat[i, 1] = euler_angles[i][1]
                    euler_flat[i, 2] = euler_angles[i][2]
                    euler_flat[i, 3] = euler_angles[i][3]
                end

                # ── Write results to HDF5 ────────────────────────────────
                sd["Euler_angles"][idx6..., :, :]    = euler_flat
                sd["C_abs"][idx6..., :]              = C_abs
                sd["C_ext"][idx6..., :]              = C_ext
                sd["S_fw_PCAS_theta"][idx6..., :]    = S_fw_theta
                sd["S_fw_PCAS_phi"][idx6..., :]      = S_fw_phi
                sd["S_bk_OCBS"][idx6..., :]          = S_bk
                sd["C_abs_mie"][idx6...]              = C_abs_mie
                sd["C_ext_mie"][idx6...]              = C_ext_mie
                sd["S_fw_PCAS_mie"][idx6...]          = S_fw_mean_mie
                sd["S_bk_OCBS_mie"][idx6...]          = S_bk_mie

                # ── Per-slot cost record (symmetric with DDA) ────────────
                # For the first slot in a REUSE shape, t_build_slot and
                # t_setup_slot were measured before t_slot_start (shared
                # shape-level phase), so we add them back here to keep
                # t_total_s = build + setup + solve + observables
                # semantically consistent with the DDA cost group.
                t_total_slot = (time_ns() - t_slot_start) / 1e9 +
                               t_build_slot + t_setup_slot
                peak_rss     = RSSMonitor.peak_bytes(mon)
                cost["t_build_s"][idx6...]        = t_build_slot
                cost["t_setup_s"][idx6...]        = t_setup_slot
                cost["t_solve_s"][idx6...]        = t_solve
                cost["t_total_s"][idx6...]        = t_total_slot
                cost["peak_rss_bytes"][idx6...]   = Int64(peak_rss)
                cost["n_tet"][idx6...]            = Int64(n_tets(mesh))
                cost["n_dof"][idx6...]            = Int64(n_basis(basis))
                cost["mean_edge_length"][idx6...] = mean_edge_length(mesh)
                cost["iters"][idx6...]            = Int64(solve_info.iterations)
                cost["converged"][idx6...]        = Int8(solve_info.converged ? 1 : 0)
                cost["solver_err"][idx6...]       = Float64(solve_info.residual_norm)

                # Per-iteration residual trace, NaN-padded to MAXITER_HISTORY.
                # Length mismatch (hist_n > MAXITER) means MAXITER was raised
                # above the schema's preallocated length — truncate to fit.
                hist = fill(NaN, MAXITER)
                eh = solve_info.residual_history
                hist_n = min(length(eh), MAXITER)
                hist[1:hist_n] .= eh[1:hist_n]
                cost["residual_history"][idx6..., :] = hist

                n_done += 1
                C_ext_mean = sum(filter(!isnan, C_ext)) /
                             max(1, count(!isnan, C_ext))
                _log("  C_ext(mean)=$(Printf.@sprintf("%.4e", C_ext_mean))  " *
                     "Mie C_ext=$(Printf.@sprintf("%.4e", C_ext_mie))  " *
                     "[$n_done done, $n_skip skipped / $n_total total]  " *
                     "t_total=$(Printf.@sprintf("%.1fs", t_total_slot))  " *
                     "peak_RSS=$(Printf.@sprintf("%.2f", peak_rss / 2^30)) GB")
                flush(stdout)
            end
        end

        RSSMonitor.stop!(mon)
        println("═" ^ 64)
        _log("Sweep complete: $n_done computed, $n_skip skipped, $n_total total")
    end
end

# Run
main()
