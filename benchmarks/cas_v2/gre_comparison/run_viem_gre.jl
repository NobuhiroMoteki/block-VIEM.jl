# Cross-validate block-VIEM.jl against block-DDA_Py on GRE shapes (β > 0).
#
# Loads the coarse-grid deformation field s_coarse saved by run_dda_gre.py,
# reconstructs the identical GRE surface in VIEM, and compares S_fw_theta,
# S_fw_phi for each orientation.
#
# Usage:
#     julia --project=. benchmarks/cas_v2/gre_comparison/run_viem_gre.jl

using BlockVIEM
using BlockVIEM: Vec3
using StaticArrays
using LinearAlgebra: norm
using Printf
import JSON3

# ---------------------------------------------------------------------------
# Load DDA results
# ---------------------------------------------------------------------------
const DDA_FILE = joinpath(@__DIR__, "dda_gre_results.json")
isfile(DDA_FILE) || error("Run run_dda_gre.py first to generate $DDA_FILE")

dda = JSON3.read(read(DDA_FILE, String))
const WL_0   = Float64(dda.wl_0)
const M_M    = Float64(dda.m_m)
const M_P    = ComplexF64(dda.m_p_re + dda.m_p_im * im)
const M_P_MAX = abs(M_P)

println("=" ^ 70)
println("VIEM–DDA GRE cross-validation (β > 0)")
println("=" ^ 70)
@printf("  wl_0 = %.3f μm,  m_m = %.3f,  m_p = %s\n", WL_0, M_M, M_P)

# ---------------------------------------------------------------------------
# Result storage for summary table
# ---------------------------------------------------------------------------
struct CompareRow
    case_name::String
    bc::Float64
    ab::Float64
    beta::Float64
    ori_beta::Float64
    n_dda::Int
    n_viem::Int
    # S_fw_theta
    Sth_dda::ComplexF64
    Sth_viem::ComplexF64
    err_Sth::Float64
    # S_fw_phi
    Sph_dda::ComplexF64
    Sph_viem::ComplexF64
    err_Sph::Float64
end
rows = CompareRow[]

# ---------------------------------------------------------------------------
# Per-case solve
# ---------------------------------------------------------------------------
for case_json in dda.cases
    name     = String(case_json.name)
    r_v_base = Float64(case_json.r_v_base)
    bc_ratio = Float64(case_json.bc_ratio)
    ab_ratio = Float64(case_json.ab_ratio)
    beta     = Float64(case_json.beta)

    println("\n── Case $name: r_v=$r_v_base, bc=$bc_ratio, ab=$ab_ratio, β=$beta ──")

    # Load Python s_coarse (stored as list-of-lists, [N_theta][N_phi])
    θ_grid = Float64.(case_json.theta_grid)
    φ_grid = Float64.(case_json.phi_grid)
    Nθ = length(θ_grid)
    Nφ = length(φ_grid)
    s_coarse = Matrix{Float64}(undef, Nθ, Nφ)
    for i in 1:Nθ
        row = case_json.s_coarse[i]
        for j in 1:Nφ
            s_coarse[i, j] = Float64(row[j])
        end
    end

    # Build GRE mesh using the Python deformation field.
    # Use lc = 0.035 μm (≈ r_ve/6, comparable to DDA lattice spacing)
    # to keep the dense solve tractable (~2000 DOFs).
    p = GREParams(r_v_base, bc_ratio, ab_ratio, beta)
    lc = 0.035
    mesh, r_ve = gre_mesh_with_field(p, s_coarse, θ_grid, φ_grid; lc=lc)
    basis = build_swg_basis(mesh; include_boundary_faces=true)
    @printf("  mesh: %d tets, %d basis, r_ve = %.5f μm (DDA: %.5f)\n",
            n_tets(mesh), n_basis(basis), r_ve, Float64(case_json.r_ve_dda))

    # Build orientation list from DDA results
    euler_list = Tuple{Float64,Float64,Float64}[]
    for ori in case_json.orientations
        push!(euler_list, (Float64(ori.alpha), Float64(ori.beta_ori), Float64(ori.gamma)))
    end

    # Solve
    results = solve_cas_v2_orientations(basis, euler_list;
                wl_0=WL_0, m_m=M_M, m_p=M_P, method=:dense, verbose=false)

    # Compare
    for (k, ori_json) in enumerate(case_json.orientations)
        Sth_dda  = ComplexF64(ori_json.S_fw_theta_re + ori_json.S_fw_theta_im * im)
        Sph_dda  = ComplexF64(ori_json.S_fw_phi_re   + ori_json.S_fw_phi_im   * im)
        Sth_viem = results[k].S_fw_theta
        Sph_viem = results[k].S_fw_phi

        err_th = abs(Sth_viem - Sth_dda) / max(abs(Sth_dda), 1e-30)
        err_ph = abs(Sph_viem - Sph_dda) / max(abs(Sph_dda), 1e-30)

        β_ori = euler_list[k][2]
        @printf("  β_ori=%.4f  |ΔS_θ|/|S_θ| = %.2f%%  |ΔS_φ|/|S_φ| = %.2f%%\n",
                β_ori, 100*err_th, 100*err_ph)
        @printf("    DDA  S_θ = %+.5e %+.5ej   S_φ = %+.5e %+.5ej\n",
                real(Sth_dda), imag(Sth_dda), real(Sph_dda), imag(Sph_dda))
        @printf("    VIEM S_θ = %+.5e %+.5ej   S_φ = %+.5e %+.5ej\n",
                real(Sth_viem), imag(Sth_viem), real(Sph_viem), imag(Sph_viem))

        push!(rows, CompareRow(name, bc_ratio, ab_ratio, beta, β_ori,
                               Int(case_json.n_dipoles), n_basis(basis),
                               Sth_dda, Sth_viem, err_th,
                               Sph_dda, Sph_viem, err_ph))
    end
end

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
println("\n" * "=" ^ 70)
println("Summary (relative error in complex forward amplitudes)")
println("=" ^ 70)
@printf("%-5s %4s %4s %5s %6s %6s %6s  %8s %8s\n",
        "Case", "b/c", "a/b", "β", "β_ori", "N_DDA", "N_VIEM",
        "|ΔS_θ|%", "|ΔS_φ|%")
println("-" ^ 70)
for r in rows
    @printf("%-5s %4.1f %4.1f %5.2f %6.3f %6d %6d  %7.2f%%  %7.2f%%\n",
            r.case_name, r.bc, r.ab, r.beta, r.ori_beta,
            r.n_dda, r.n_viem, 100*r.err_Sth, 100*r.err_Sph)
end

# Write results JSON for documentation
output = joinpath(@__DIR__, "viem_gre_comparison.json")
open(output, "w") do io
    println(io, "[")
    for (i, r) in enumerate(rows)
        println(io, "  {")
        @printf(io, "    \"case\": \"%s\", \"bc\": %.1f, \"ab\": %.1f, \"beta\": %.2f,\n",
                r.case_name, r.bc, r.ab, r.beta)
        @printf(io, "    \"beta_ori\": %.4f, \"n_dda\": %d, \"n_viem\": %d,\n",
                r.ori_beta, r.n_dda, r.n_viem)
        @printf(io, "    \"err_Sth\": %.6f, \"err_Sph\": %.6f\n", r.err_Sth, r.err_Sph)
        print(io, "  }")
        i < length(rows) && print(io, ",")
        println(io)
    end
    println(io, "]")
end
println("\nWrote $output")
