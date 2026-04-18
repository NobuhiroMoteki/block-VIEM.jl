# MSTMforCAS.jl side of the 2-sphere doublet CAS-v2 benchmark.
#
# Usage:
#     julia --project=/home/moteki/Julia/MSTMforCAS.jl \
#           benchmarks/cas_v2/doublet_mstm/run_mstm.jl
#
# For each material × orientation β:
#   - construct a 2-sphere aggregate with centres at
#       ±(d/2) · (sin β, 0, cos β)
#     so the doublet axis makes angle β with the lab-z incidence direction
#     (equivalent to block-VIEM's `cas_orientation(0, β, 0)` with the
#     particle fixed along lab-z).
#   - call `compute_scattering(agg, m_rel, k_medium)` which returns the
#     BH83 amplitude tuple `(S1, S2, S3, S4)` at forward (θ=0) and
#     backward (θ=π).
#   - convert to the CAS-v2 observable S_fw_mean using the block-DDA_Py
#     convention `S_fw_mean = (S_fw_theta + S_fw_phi)/2` for RCP incidence.
#
# The observable formula matches block-VIEM.jl's `compute_cas_observables`
# (postprocess.jl) up to possibly a scattering-plane sign convention
# on the BH83 off-diagonal elements (S3, S4).  See comparison.jl for the
# reconciliation.

using Printf
using LinearAlgebra
using MSTMforCAS

# Minimal JSON writer for Dict{String,Any} / Vector / primitives.
# Keeps run_mstm.jl free of JSON3 dependency.
_json_enc(x::Nothing) = "null"
_json_enc(x::Bool) = x ? "true" : "false"
_json_enc(x::Integer) = string(x)
_json_enc(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
_json_enc(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json_enc(v::AbstractVector) = "[" * join(_json_enc.(v), ", ") * "]"
function _json_enc(d::AbstractDict)
    parts = ["\"$k\": " * _json_enc(v) for (k, v) in d]
    return "{\n  " * join(parts, ",\n  ") * "\n}"
end
function dump_json(path::AbstractString, obj)
    open(path, "w") do io
        print(io, _json_enc(obj))
    end
end

include(joinpath(@__DIR__, "config.jl"))

"""
    doublet_centres(beta) -> (positions, radii)

Returns the two monomer centres rotated so the doublet axis makes angle
`beta` with the +z axis (rotation about y-axis).  Incidence in MSTM is
always along +z.
"""
function doublet_centres(beta::Float64)
    # Axis direction at rest: (0, 0, 1).  Rotate about y by +beta:
    #   Ry(β)·(0,0,1)^T = (sin β, 0, cos β)
    ux, uy, uz = sin(beta), 0.0, cos(beta)
    p1 = (-D_CENTRE / 2) .* (ux, uy, uz)
    p2 = (+D_CENTRE / 2) .* (ux, uy, uz)
    positions = [p1[1] p2[1]; p1[2] p2[2]; p1[3] p2[3]]   # (3, 2)
    radii = [R_MONOMER, R_MONOMER]
    return positions, radii
end

"""
    bh83_forward_to_cas_viem(S_fwd, k) -> (S_fw_theta, S_fw_phi, S_fw_mean)

Convert BH83 forward amplitudes `(S1, S2, S3, S4)` to the block-VIEM /
block-DDA_Py CAS-v2 convention for RCP incidence.

Two conversions are stacked:

1. BH83 → MI02 scattering-amplitude matrix (Mishchenko 2002):
       S11 = S2/(-ik),   S12 = S3/(ik),
       S21 = S4/(ik),    S22 = S1/(-ik).
   (The sign flip on the off-diagonals reflects the MI02 ê_φ choice
   of opposite handedness from BH83's ê_⊥.)

2. MI02 → CAS-v2 forward amplitudes (block-DDA_Py convention):
       S_fw_θ = S11 + i·S12 = (S2 − i·S3)/(-ik),
       S_fw_φ = S22 − i·S21 = (S1 + i·S4)/(-ik),
       S_fw_mean = (S_fw_θ + S_fw_φ)/2.

For aggregates with the x-z-plane reflection symmetry used in this
benchmark, S3 = S4 = 0 and the two off-diagonal terms drop out, but
the correct signs are retained for generality.
"""
function bh83_forward_to_cas_viem(S_fwd::NTuple{4,ComplexF64}, k::Real)
    S1, S2, S3, S4 = S_fwd
    mik = ComplexF64(0, -k)     # (-ik)
    S_fw_theta = (S2 - im * S3) / mik
    S_fw_phi   = (S1 + im * S4) / mik
    S_fw_mean  = (S_fw_theta + S_fw_phi) / 2
    return S_fw_theta, S_fw_phi, S_fw_mean
end

function solve_one_material(mat)
    println("\n" * "="^70)
    println("[MSTM] material=$(mat.name)  m_p=$(mat.m_p)")
    println("="^70)

    k_medium = 2π * M_MEDIUM / WL_0
    m_rel = ComplexF64(mat.m_p) / M_MEDIUM

    recs = Dict{String,Any}[]
    t_total = @elapsed for β in BETAS
        positions, radii = doublet_centres(β)
        agg = AggregateGeometry(positions, radii, 2, "doublet_benchmark")

        result, _ = compute_scattering(agg, m_rel, k_medium;
                         use_fft          = MSTM_USE_FFT,
                         tol              = MSTM_TOL,
                         truncation_order = MSTM_TRUNC_ORDER)
        S_fw_theta, S_fw_phi, S_fw_mean =
            bh83_forward_to_cas_viem(result.S_forward, k_medium)

        @printf("  β=%.4f  Qext=%.4f  |S_fw_mean|=%.4e  arg(S_fw_mean)=%+.3f rad  iters=%d\n",
                β, result.Q_ext, abs(S_fw_mean), angle(S_fw_mean), result.n_iterations)

        push!(recs, Dict(
            "material"      => mat.name,
            "beta"          => β,
            "S_fw_mean_re"  => real(S_fw_mean),
            "S_fw_mean_im"  => imag(S_fw_mean),
            "S_fw_theta_re" => real(S_fw_theta),
            "S_fw_theta_im" => imag(S_fw_theta),
            "S_fw_phi_re"   => real(S_fw_phi),
            "S_fw_phi_im"   => imag(S_fw_phi),
            "S1_re" => real(result.S_forward[1]),
            "S1_im" => imag(result.S_forward[1]),
            "S2_re" => real(result.S_forward[2]),
            "S2_im" => imag(result.S_forward[2]),
            "S3_re" => real(result.S_forward[3]),
            "S3_im" => imag(result.S_forward[3]),
            "S4_re" => real(result.S_forward[4]),
            "S4_im" => imag(result.S_forward[4]),
            "Q_ext" => result.Q_ext,
            "Q_sca" => result.Q_sca,
            "Q_abs" => result.Q_abs,
            "n_iterations"    => result.n_iterations,
            "truncation_order"=> result.truncation_order,
        ))
    end
    @printf("  total solve: %.2f s\n", t_total)
    return recs
end

function main()
    println("MSTMforCAS.jl  CAS-v2 doublet benchmark (exact reference)")
    println("  WL_0=$(WL_0) μm, M_MEDIUM=$(M_MEDIUM)")
    println("  monomer R=$(R_MONOMER) μm, gap=$(GAP) μm")

    all_records = Dict{String,Any}[]
    for mat in MATERIALS
        recs = solve_one_material(mat)
        append!(all_records, recs)
    end

    dump_json(MSTM_JSON, Dict(
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
    println("\nWrote ", MSTM_JSON)
end

main()
