# Mesh-refinement study for the 2-sphere doublet benchmark.
#
# Purpose: measure how the VIEM discretisation error on the CAS-v2
# forward observables changes when the mesh of the plasmonic-gold case
# is refined from lc = R/5 to lc = R/6.  With v0.8.2 the baseline
# lc = R/5 run has a 2.3 % error on the stiffest component
# (Im S_fw_θ at β = π/2).
#
# The MSTM side is held fixed at N_trunc = 15 (the converged reference
# from run_mstm.jl); only the VIEM mesh is refined.  Au is the only
# material swept: PS is already within 0.5 % at R/5.
#
# The four-level study (R/5 to R/8) of docs/benchmark_results.md §8 is
# run with phase_a_memory_study.jl and compute_refinement_errors.py.
# Its errors do not decrease monotonically below R/6, so the two-point
# exponent fitted from this script is not a convergence rate.
#
# Usage:
#     julia --project=. benchmarks/cas_v2/doublet_mstm/run_viem_refinement.jl
#
# Output:
#     results_viem_refinement.json  (per-lc, per-β complex amplitudes)

using LinearAlgebra: norm
using Printf
using JSON3
using BlockVIEM

include(joinpath(@__DIR__, "config.jl"))

# lc ratios tested: R/k with k ∈ LC_RATIOS.  Combined with the existing
# R/5 baseline (already in results_viem.json), this provides 2 points.
# Only R/6 is run here.  When this script was written (April 2026,
# before the Phase A surface-moment AIM) R/7 did not fit in 15 GB RAM.
# Finer levels now run with phase_a_memory_study.jl (R/8 peaks at
# about 5 GB).
const LC_RATIOS = (6,)

# Refinement is run on Au only: it has the largest baseline error and
# is the only case where the convergence rate is informative.
const REFINE_MATERIAL = (name = "Au", m_p = 0.17525 + 3.4830im)

function mean_edge_length(mesh)
    s = 0.0;  c = 0
    for tet in mesh.tets
        for (a, b) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
            s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]])
            c += 1
        end
    end
    return s / c
end

function solve_at_lc(lc::Float64, mat)
    println("\n" * "="^70)
    @printf("[VIEM] material=%s  m_p=%s  lc=%.5f (R/%.2f)\n",
            mat.name, mat.m_p, lc, R_MONOMER / lc)
    println("="^70)

    centers = zeros(Float64, 3, 2)
    centers[3, 1] = -D_CENTRE / 2
    centers[3, 2] = +D_CENTRE / 2
    agg = SphereAggregate(centers, [R_MONOMER, R_MONOMER])

    mesh, mesh_path = mesh_sphere_aggregate(agg;
                         overlap_factor = 0.0, lc = lc, verbosity = 0)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N_dof = n_basis(basis)
    hbar  = mean_edge_length(mesh)
    pitch = 0.5 * hbar
    @printf("  tets=%d  SWG-DOFs=%d  h̄=%.5f  AIM-pitch=%.5f\n",
            length(mesh.tets), N_dof, hbar, pitch)

    euler_list = [(0.0, β, 0.0) for β in BETAS]

    t_solve = @elapsed results, info = solve_cas_v2_orientations(
        basis, euler_list;
        wl_0 = WL_0, m_m = M_MEDIUM, m_p = ComplexF64(mat.m_p),
        method = VIEM_METHOD, precond = VIEM_PRECOND, pitch = pitch, padding = VIEM_PADDING,
        tol = VIEM_TOL, maxiter = VIEM_MAXITER, verbose = false,
        return_solve_info = true,
    )
    @printf("  solve: %.2f s  iters=%d  converged=%s  residual=%.3e\n",
            t_solve, info.iterations, info.converged, info.residual_norm)

    recs = Dict{String,Any}[]
    for (β, res) in zip(BETAS, results)
        push!(recs, Dict(
            "material"      => mat.name,
            "beta"          => β,
            "lc"            => lc,
            "lc_over_R"     => lc / R_MONOMER,
            "n_tets"        => length(mesh.tets),
            "N_dof"         => N_dof,
            "h_bar"         => hbar,
            "solve_time_s"  => t_solve,
            "solver_method"     => String(VIEM_METHOD),
            "solver_precond"    => String(VIEM_PRECOND),
            "solver_iterations" => info.iterations,
            "solver_converged"  => info.converged,
            "solver_residual"   => info.residual_norm,
            "S_fw_mean_re"  => real(res.S_fw_mean),
            "S_fw_mean_im"  => imag(res.S_fw_mean),
            "S_fw_theta_re" => real(res.S_fw_theta),
            "S_fw_theta_im" => imag(res.S_fw_theta),
            "S_fw_phi_re"   => real(res.S_fw_phi),
            "S_fw_phi_im"   => imag(res.S_fw_phi),
        ))
        @printf("  β=%.4f  |S_fw_mean|=%.4e  arg(S_fw_mean)=%+.3f rad\n",
                β, abs(res.S_fw_mean), angle(res.S_fw_mean))
    end
    rm(mesh_path; force = true)
    return recs
end

function main()
    println("block-VIEM.jl  CAS-v2 doublet mesh-refinement study")
    println("  material: $(REFINE_MATERIAL.name)")
    println("  lc / R ratios: $(["1/$k" for k in LC_RATIOS])")
    println("  WL_0=$(WL_0) μm, R=$(R_MONOMER) μm, gap=$(GAP) μm")

    all_records = Dict{String,Any}[]
    for k in LC_RATIOS
        lc = R_MONOMER / k
        recs = solve_at_lc(lc, REFINE_MATERIAL)
        append!(all_records, recs)
    end

    out_path = joinpath(@__DIR__, "results_viem_refinement.json")
    open(out_path, "w") do io
        JSON3.pretty(io, Dict(
            "config" => Dict(
                "R_monomer" => R_MONOMER,
                "gap"       => GAP,
                "d_centre"  => D_CENTRE,
                "wl_0"      => WL_0,
                "m_medium"  => M_MEDIUM,
                "betas"     => collect(BETAS),
                "lc_ratios" => collect(LC_RATIOS),
                "material"  => Dict(
                    "name" => REFINE_MATERIAL.name,
                    "m_p_re" => real(REFINE_MATERIAL.m_p),
                    "m_p_im" => imag(REFINE_MATERIAL.m_p),
                ),
            ),
            "results" => all_records,
        ))
    end
    println("\nWrote ", out_path)
end

main()
