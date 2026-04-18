# Convergence check of the MSTM VSWF truncation order for the
# doublet benchmark.  By default MSTMforCAS uses N=3 for x≈0.30; we
# override with a range of N values and report |S_fw_mean|, phase, and
# pairwise relative deviation to determine at what N the reference
# solution is converged to ≤ 1e-5.
#
# Usage:
#     julia --project=/home/moteki/Julia/MSTMforCAS.jl \
#           benchmarks/cas_v2/doublet_mstm/check_trunc_convergence.jl

using Printf
using LinearAlgebra
using MSTMforCAS

include(joinpath(@__DIR__, "config.jl"))

function doublet_centres(beta::Float64)
    ux, uy, uz = sin(beta), 0.0, cos(beta)
    p1 = (-D_CENTRE / 2) .* (ux, uy, uz)
    p2 = (+D_CENTRE / 2) .* (ux, uy, uz)
    positions = [p1[1] p2[1]; p1[2] p2[2]; p1[3] p2[3]]
    radii = [R_MONOMER, R_MONOMER]
    return positions, radii
end

function bh83_forward_to_cas_viem(S_fwd::NTuple{4,ComplexF64}, k::Real)
    S1, S2, S3, S4 = S_fwd
    mik = ComplexF64(0, -k)
    S_fw_theta = (S2 - im * S3) / mik
    S_fw_phi   = (S1 + im * S4) / mik
    S_fw_mean  = (S_fw_theta + S_fw_phi) / 2
    return S_fw_mean
end

function main()
    k_medium = 2π * M_MEDIUM / WL_0
    # Cap at N=6: N≥7 triggers downward-recurrence breakdown of the
    # Mie Riccati-Bessel precomputation at x≈0.3 in MSTMforCAS@v0.x
    # (see TMatrixSolver._precompute_T_values).  N=6 is the largest
    # value that returns physically meaningful results here.
    trunc_orders = (3, 4, 5, 6)

    println("MSTM VSWF truncation-order convergence (doublet benchmark)")
    println("  R=$R_MONOMER μm, gap=$GAP μm, λ₀=$WL_0 μm, x≈",
            round(k_medium * R_MONOMER; sigdigits=3))

    for mat in MATERIALS
        println("\n" * "="^74)
        println("  material=$(mat.name)   m_p=$(mat.m_p)   |m·x|≈",
                round(abs(mat.m_p) * k_medium * R_MONOMER; sigdigits=3))
        println("="^74)
        m_rel = ComplexF64(mat.m_p) / M_MEDIUM

        for β in BETAS
            positions, radii = doublet_centres(β)
            agg = AggregateGeometry(positions, radii, 2, "doublet_trunc_test")
            println(@sprintf("  β = %.4f rad", β))
            println("    N   |S_fw_mean|    arg(S_fw_mean)[rad]  rel.err vs N=max")

            # Baseline at largest N for reference
            ref_res, _ = compute_scattering(agg, m_rel, k_medium;
                              use_fft=false, tol=1e-10,
                              truncation_order = trunc_orders[end])
            S_ref = bh83_forward_to_cas_viem(ref_res.S_forward, k_medium)

            for N in trunc_orders
                r, _ = compute_scattering(agg, m_rel, k_medium;
                           use_fft=false, tol=1e-10,
                           truncation_order = N)
                S = bh83_forward_to_cas_viem(r.S_forward, k_medium)
                rel = abs(S - S_ref) / abs(S_ref)
                println(@sprintf("    %-4d %.6e   %+.4f          %.2e",
                                 N, abs(S), angle(S), rel))
            end
        end
    end
end

main()
