# Mesh-refinement study for the 2-sphere doublet benchmark.
#
# Purpose: verify that the VIEM discretisation error on the CAS-v2
# forward observables converges at the theoretical linear-SWG rate
# `err ~ h^p` (with p ≈ 2 expected) for the plasmonic-gold case
# where the baseline lc = R/5 run leaves a 10.5 % error on the
# stiffest component (Im S_fw_θ at β = π/2).
#
# The MSTM side is held fixed at N_trunc = 15 (the converged reference
# from run_mstm.jl); only the VIEM mesh is refined.  Au is the only
# material swept — PS already converges to ~1.4 % at R/5 and is not
# the rate-limiting case.
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
# R/5 baseline (already in results_viem.json), this provides 2 points
# for a log-log convergence-rate slope.  Memory footprint of the AIM
# solver scales faster than N in practice (≈ O(N²) near-field storage
# for this dense-aggregate geometry), so on a 15 GB RAM workstation
# only R/6 is feasible as the finer point — R/7 (N≈30K DOFs) OOMs
# with current default block-BiCGSTAB buffer allocation.
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

    t_solve = @elapsed results = solve_cas_v2_orientations(
        basis, euler_list;
        wl_0 = WL_0, m_m = M_MEDIUM, m_p = ComplexF64(mat.m_p),
        method = :aim_bicgstab, pitch = pitch, padding = VIEM_PADDING,
        tol = VIEM_TOL, maxiter = VIEM_MAXITER, verbose = false,
    )
    @printf("  solve: %.2f s\n", t_solve)

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
