# block-VIEM.jl side of the 2-sphere doublet CAS-v2 benchmark.
#
# Usage:
#     julia --project=. benchmarks/cas_v2/doublet_mstm/run_viem.jl
#
# Outputs:
#     results_viem.json (keyed by "<material>/beta_<β>")
#
# Uses `SphereAggregate` + `mesh_sphere_aggregate` to build the doublet,
# then `solve_cas_v2_orientations(method=:aim_bicgstab)` with one Euler
# triple per BETA.

using LinearAlgebra: norm
using Printf
using JSON3
using BlockVIEM

include(joinpath(@__DIR__, "config.jl"))

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

function solve_one_material(mat)
    println("\n" * "="^70)
    println("[VIEM] material=$(mat.name)  m_p=$(mat.m_p)")
    println("="^70)

    # Build doublet aggregate: two spheres at (0,0,±d/2) with GAP>0
    centers = zeros(Float64, 3, 2)
    centers[3, 1] = -D_CENTRE / 2
    centers[3, 2] = +D_CENTRE / 2
    radii = [R_MONOMER, R_MONOMER]
    agg = SphereAggregate(centers, radii)

    lc = lc_for_material(ComplexF64(mat.m_p))
    @printf("  R=%.4f  d=%.4f  lc=%.5f (|m|=%.3f)\n",
            R_MONOMER, D_CENTRE, lc, abs(mat.m_p))

    mesh, mesh_path = mesh_sphere_aggregate(agg;
                         overlap_factor = 0.0,       # disjoint, no neck
                         lc = lc,
                         verbosity = 0)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N_dof = n_basis(basis)
    hbar = mean_edge_length(mesh)
    pitch = 0.5 * hbar
    @printf("  tets=%d  SWG-DOFs=%d  h̄=%.5f  AIM-pitch=%.5f\n",
            length(mesh.tets), N_dof, hbar, pitch)

    # Orientations: (α, β, γ) = (0, β, 0) — β is angle between incidence
    # and particle z-axis (= doublet axis).
    euler_list = [(0.0, β, 0.0) for β in BETAS]

    t_solve = @elapsed results = solve_cas_v2_orientations(
        basis, euler_list;
        wl_0   = WL_0,
        m_m    = M_MEDIUM,
        m_p    = ComplexF64(mat.m_p),
        method = :aim_bicgstab,
        pitch  = pitch,
        padding = VIEM_PADDING,
        tol     = VIEM_TOL,
        maxiter = VIEM_MAXITER,
        verbose = false,
    )
    @printf("  solve: %.2f s\n", t_solve)

    # Store records
    recs = Dict{String,Any}[]
    for (β, res) in zip(BETAS, results)
        push!(recs, Dict(
            "material"    => mat.name,
            "beta"        => β,
            "S_fw_mean_re"  => real(res.S_fw_mean),
            "S_fw_mean_im"  => imag(res.S_fw_mean),
            "S_fw_theta_re" => real(res.S_fw_theta),
            "S_fw_theta_im" => imag(res.S_fw_theta),
            "S_fw_phi_re"   => real(res.S_fw_phi),
            "S_fw_phi_im"   => imag(res.S_fw_phi),
            "S_bk_re"       => real(res.S_bk),
            "S_bk_im"       => imag(res.S_bk),
            "N_dof"         => N_dof,
            "n_tets"        => length(mesh.tets),
            "lc"            => lc,
        ))
        @printf("  β=%.4f  |S_fw_mean|=%.4e  arg(S_fw_mean)=%+.3f rad\n",
                β, abs(res.S_fw_mean), angle(res.S_fw_mean))
    end

    rm(mesh_path; force = true)
    return recs
end

function main()
    println("block-VIEM.jl  CAS-v2 doublet benchmark")
    println("  WL_0=$(WL_0) μm, M_MEDIUM=$(M_MEDIUM)")
    println("  monomer R=$(R_MONOMER) μm, gap=$(GAP) μm")

    all_records = Dict{String,Any}[]
    for mat in MATERIALS
        recs = solve_one_material(mat)
        append!(all_records, recs)
    end

    open(VIEM_JSON, "w") do io
        JSON3.pretty(io, Dict(
            "config" => Dict(
                "R_monomer" => R_MONOMER,
                "gap"       => GAP,
                "d_centre"  => D_CENTRE,
                "wl_0"      => WL_0,
                "m_medium"  => M_MEDIUM,
                "betas"     => collect(BETAS),
            ),
            "results" => all_records,
        ))
    end
    println("\nWrote ", VIEM_JSON)
end

main()
