# Phase A memory & timing study on the Au doublet at R/k for k = 5, 6, 7.
#
# Purpose: measure the asymptotic memory reduction produced by the
# Phase A surface-moment AIM extension (commit 81c7961).  Phase-1a
# stored the boundary kernel K^B+K^C+K^D in a dense N_bnd × N upper-
# triangle matrix (`half_swg_extra`) that scaled as O(N^{5/3}) and
# was the dominant live-memory user (81–86 % tracked at R/5, R/6).
# Phase A absorbs those kernels into (a) the AIM far-field via a single
# new scalar projection Wsurf, and (b) the existing sparse
# `precorrection` for near pairs, so `half_swg_extra` becomes an empty
# sparse placeholder.
#
# Output: JSON record `phase_a_memory.json` + Markdown table printed
# to stdout for direct paste into README / theory_note.
#
# Usage:
#     julia --project=. benchmarks/cas_v2/doublet_mstm/phase_a_memory_study.jl
#
# Runtime (i7-1265U, 16 GB RAM, Julia 1.11):
#   R/5  ~90 s, R/6  ~4 min, R/7  ~20 min.
# Override which refinements to run via BM_LC_RATIOS env (comma-sep).

using LinearAlgebra: norm
using Printf
using JSON3
using SparseArrays: nnz, SparseMatrixCSC
using BlockVIEM

include(joinpath(@__DIR__, "config.jl"))

const DEFAULT_LC_RATIOS = (5, 6, 7)
const LC_RATIOS = let env = get(ENV, "BM_LC_RATIOS", "")
    isempty(env) ? DEFAULT_LC_RATIOS :
        Tuple(parse(Int, s) for s in split(strip(env), ','))
end

# Au at λ₀ = 638 nm (Johnson & Christy 1972) — high-contrast case that
# stresses the scalar divergence channel (K^B+K^C+K^D).
const MAT = (name = "Au", m_p = 0.17525 + 3.4830im)

# Measure RSS in bytes (Linux only).
function rss_bytes()
    try
        open("/proc/self/statm", "r") do io
            parts = split(readline(io))
            return parse(Int, parts[2]) * 4096
        end
    catch
        return -1
    end
end

function mean_edge_length(mesh)
    s = 0.0; c = 0
    for tet in mesh.tets, (a, b) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
        s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]])
        c += 1
    end
    return s / c
end

# Component-wise memory of the AIM operator.  Returns a dict with
# byte sizes for every large field, using Base.summarysize which
# walks pointer graphs and reports in-memory footprint (incl. sparse
# colptr/rowval/nzval arrays and FFT kernel).
function op_memory_breakdown(op::BlockVIEM.AIMOperator)
    proj = op.projection
    return (
        mass            = Base.summarysize(op.mass),
        Wx              = Base.summarysize(proj.Wx),
        Wy              = Base.summarysize(proj.Wy),
        Wz              = Base.summarysize(proj.Wz),
        Wdiv            = Base.summarysize(proj.Wdiv),
        Wsurf           = Base.summarysize(proj.Wsurf),
        G_hat           = Base.summarysize(op.G_hat),
        precorrection   = Base.summarysize(op.precorrection),
        half_swg_extra  = Base.summarysize(op.half_swg_extra),
        total           = Base.summarysize(op),
    )
end

# Human-readable MiB string.
mib(bytes::Integer) = @sprintf("%.1f", bytes / 1024 / 1024)

function measure_one(k::Integer)
    lc = R_MONOMER / k
    println("\n" * "="^72)
    @printf("[Phase A] Au doublet  lc = R/%d = %.5f μm\n", k, lc)
    println("="^72)

    centers = zeros(Float64, 3, 2)
    centers[3, 1] = -D_CENTRE / 2
    centers[3, 2] = +D_CENTRE / 2
    agg = SphereAggregate(centers, [R_MONOMER, R_MONOMER])

    mesh, mesh_path = mesh_sphere_aggregate(agg;
                         overlap_factor = 0.0, lc = lc, verbosity = 0)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N_dof = n_basis(basis)
    N_bnd = count(basis.is_boundary)
    hbar  = mean_edge_length(mesh)
    pitch = 0.5 * hbar
    @printf("  tets=%d  N_dof=%d  N_bnd=%d  h̄=%.5f  pitch=%.5f\n",
            length(mesh.tets), N_dof, N_bnd, hbar, pitch)

    k0 = 2π * M_MEDIUM / WL_0
    eps_p = ComplexF64(MAT.m_p)^2
    rss_before = rss_bytes()
    GC.gc()
    t_setup = @elapsed op = build_aim_operator(basis;
                            k0 = k0, eps_p = eps_p, eps_bg = 1.0,
                            pitch = pitch, padding = VIEM_PADDING)
    GC.gc()
    rss_after_setup = rss_bytes()

    mem = op_memory_breakdown(op)
    @printf("  setup: %.2f s\n", t_setup)
    @printf("  Wsurf nnz=%d  (Phase A surface-moment projection)\n",
            nnz(op.projection.Wsurf))
    @printf("  half_swg_extra nnz=%d  (Phase A: should be 0)\n",
            nnz(op.half_swg_extra))
    @printf("  component sizes [MiB]:\n")
    @printf("    mass            %s\n", mib(mem.mass))
    @printf("    Wx / Wy / Wz    %s / %s / %s\n",
            mib(mem.Wx), mib(mem.Wy), mib(mem.Wz))
    @printf("    Wdiv            %s\n", mib(mem.Wdiv))
    @printf("    Wsurf           %s  ← Phase A new\n", mib(mem.Wsurf))
    @printf("    G_hat (FFT)     %s\n", mib(mem.G_hat))
    @printf("    precorrection   %s\n", mib(mem.precorrection))
    @printf("    half_swg_extra  %s  (Phase-1a bottleneck, now 0)\n",
            mib(mem.half_swg_extra))
    @printf("    ─────────────────────────────\n")
    @printf("    total tracked   %s\n", mib(mem.total))

    # Solve 3 orientations via AIM block-BiCGSTAB.
    euler_list = [(0.0, β, 0.0) for β in BETAS]
    t_solve = @elapsed results = solve_cas_v2_orientations(
        basis, euler_list;
        wl_0 = WL_0, m_m = M_MEDIUM, m_p = ComplexF64(MAT.m_p),
        method = :aim_bicgstab, pitch = pitch, padding = VIEM_PADDING,
        tol = VIEM_TOL, maxiter = VIEM_MAXITER, verbose = false,
    )
    GC.gc()
    rss_after_solve = rss_bytes()
    @printf("  solve (3 orientations): %.2f s\n", t_solve)
    @printf("  RSS: before=%s MiB, after setup=%s MiB, after solve=%s MiB\n",
            mib(rss_before), mib(rss_after_setup), mib(rss_after_solve))

    # Record final observables for sanity (compared against R/5 baseline
    # in results_viem_refinement.json if present).
    fw_vals = [(β, res.S_fw_mean, res.S_fw_theta, res.S_fw_phi)
               for (β, res) in zip(BETAS, results)]
    for (β, S_mean, S_th, _S_ph) in fw_vals
        @printf("  β=%.4f  |S_fw_mean|=%.4e  Im S_fw_θ=%+.4e\n",
                β, abs(S_mean), imag(S_th))
    end

    rm(mesh_path; force = true)

    return Dict(
        "lc_ratio"      => k,
        "lc"            => lc,
        "n_tets"        => length(mesh.tets),
        "N_dof"         => N_dof,
        "N_bnd"         => N_bnd,
        "h_bar"         => hbar,
        "pitch"         => pitch,
        "t_setup_s"     => t_setup,
        "t_solve_s"     => t_solve,
        "memory_bytes"  => Dict(pairs(mem)),
        "nnz" => Dict(
            "mass"          => nnz(op.mass),
            "Wx"            => nnz(op.projection.Wx),
            "Wdiv"          => nnz(op.projection.Wdiv),
            "Wsurf"         => nnz(op.projection.Wsurf),
            "precorrection" => nnz(op.precorrection),
            "half_swg_extra"=> nnz(op.half_swg_extra),
        ),
        "rss_bytes"     => Dict(
            "before"       => rss_before,
            "after_setup"  => rss_after_setup,
            "after_solve"  => rss_after_solve,
        ),
        "observables"   => [Dict(
            "beta"         => β,
            "S_fw_mean_re" => real(S_mean),
            "S_fw_mean_im" => imag(S_mean),
            "S_fw_theta_re"=> real(S_th),
            "S_fw_theta_im"=> imag(S_th),
            "S_fw_phi_re"  => real(S_ph),
            "S_fw_phi_im"  => imag(S_ph),
        ) for (β, S_mean, S_th, S_ph) in fw_vals],
    )
end

function print_summary_table(records)
    println("\n" * "="^72)
    println("Phase A Au-doublet memory / timing summary")
    println("="^72)
    println("| lc/R | N_dof | N_bnd | Wsurf MiB | precorr MiB | half_swg MiB | total MiB | t_setup s | t_solve s |")
    println("|------|-------|-------|-----------|-------------|--------------|-----------|-----------|-----------|")
    for r in records
        mem = r["memory_bytes"]
        @printf("| 1/%-3d | %5d | %5d | %9s | %11s | %12s | %9s | %9.1f | %9.1f |\n",
                r["lc_ratio"], r["N_dof"], r["N_bnd"],
                mib(mem[:Wsurf]), mib(mem[:precorrection]),
                mib(mem[:half_swg_extra]), mib(mem[:total]),
                r["t_setup_s"], r["t_solve_s"])
    end
end

function main()
    println("block-VIEM.jl  Phase A memory study  Au doublet")
    println("  lc/R ratios: ", collect(LC_RATIOS))
    records = Dict{String,Any}[]
    for k in LC_RATIOS
        push!(records, measure_one(k))
    end
    print_summary_table(records)
    out_path = joinpath(@__DIR__, "phase_a_memory.json")
    open(out_path, "w") do io
        JSON3.pretty(io, Dict(
            "config" => Dict(
                "R_monomer" => R_MONOMER, "gap" => GAP,
                "wl_0" => WL_0, "m_medium" => M_MEDIUM,
                "betas" => collect(BETAS),
                "material" => Dict("name" => MAT.name,
                                   "m_p_re" => real(MAT.m_p),
                                   "m_p_im" => imag(MAT.m_p)),
                "lc_ratios" => collect(LC_RATIOS),
            ),
            "records" => records,
        ))
    end
    @printf("\nWrote %s\n", out_path)
end

main()
