# MSTM exact reference for paper-production doublet HDF5s.
#
# Reads geometry/material from a paper doublet HDF5 (created by
# viem_results/paper/doublet_*.jl) and runs MSTMforCAS.jl in its own
# Julia environment to compute the numerically exact CAS-v2 observables
# at every (a_eq, β) pair, where β = particle Euler-β =
# angle between incidence direction and the doublet axis.
#
# Output is written to a SEPARATE HDF5 (CLAUDE.md §6) so the VIEM
# production HDF5 stays clean.  Default output path:
#   viem_results/paper/mstm_<basename>.hdf5
# (e.g., doublet_n20.hdf5 → mstm_doublet_n20.hdf5)
#
# Usage:
#   julia --project=/home/moteki/Julia/MSTMforCAS.jl \
#         viem_results/paper/run_mstm_reference.jl \
#         viem_results/paper/doublet_n20.hdf5 [output_path]
#
# Conventions:
#   * Doublet axis at rest = +z; rotated by β about y so it makes angle β
#     with lab +z (matches viem_results/run_viem.jl spheroid mode at α=0,γ=0).
#   * truncation_order = 15  (CLAUDE.md / README.md:475-482; converged for
#     |m·x| ≈ 1 with the Miller ψ_n / j_n stability fixes in MSTMforCAS ≥ 0.4.3)
#   * BH83 → MI02 → CAS-v2 amplitude conversion as in
#     benchmarks/cas_v2/doublet_mstm/run_mstm.jl::bh83_forward_to_cas_viem.

using MSTMforCAS
using HDF5
using Printf
using Dates

# ──────────────────────────────────────────────────────────────────────
#  Settings (mirrors benchmarks/cas_v2/doublet_mstm/config.jl, but
#  parameters drive the sweep dimensions per the input HDF5)
# ──────────────────────────────────────────────────────────────────────
const MSTM_USE_FFT     = false
const MSTM_TOL         = 1e-8
const MSTM_TRUNC_ORDER = 15

# ──────────────────────────────────────────────────────────────────────
#  Geometry: doublet axis rotated by β about lab y-axis
# ──────────────────────────────────────────────────────────────────────
function doublet_centres(R::Float64, gap::Float64, β::Float64)
    d = 2 * R + gap                # centre-to-centre separation
    ux, uy, uz = sin(β), 0.0, cos(β)
    # positions: (3, 2) — column j is monomer j
    p1 = -(d / 2) .* (ux, uy, uz)
    p2 = +(d / 2) .* (ux, uy, uz)
    positions = [p1[1] p2[1]; p1[2] p2[2]; p1[3] p2[3]]
    radii = [R, R]
    return positions, radii
end

# ──────────────────────────────────────────────────────────────────────
#  BH83 forward/backward → CAS-v2 (block-DDA_Py / block-VIEM convention)
#  See benchmarks/cas_v2/doublet_mstm/run_mstm.jl for derivation.
# ──────────────────────────────────────────────────────────────────────
function bh83_to_cas_v2_forward(S_fwd::NTuple{4,ComplexF64}, k::Real)
    S1, S2, S3, S4 = S_fwd
    mik = ComplexF64(0, -k)        # (-ik)
    S_fw_theta = (S2 - im * S3) / mik
    S_fw_phi   = (S1 + im * S4) / mik
    S_fw_mean  = (S_fw_theta + S_fw_phi) / 2
    return S_fw_theta, S_fw_phi, S_fw_mean
end

function bh83_to_cas_v2_backward(S_bwd::NTuple{4,ComplexF64}, k::Real)
    # OCBS amplitude per docs/theory_note.tex Eq.(eq:S-bk-ocbs) and
    # bl_dda/scatterer.py:235 (block-DDA_Py reference implementation):
    #   S_bk_theta = S_M11(π) + i·S_M12(π)
    #   S_bk_phi   = S_M22(π) - i·S_M21(π)
    #   S_bk       = (-S_bk_theta + S_bk_phi) / √2
    #              = (-S_M11 + S_M22 - i·S_M12 - i·S_M21)(π) / √2
    # where S_M is Mishchenko's amplitude matrix (length units), obtained
    # from BH83's S₁..S₄ by S_M11 = S₂/(−ik), S_M22 = S₁/(−ik),
    # S_M12 = S₃/(ik), S_M21 = S₄/(ik).
    S1, S2, S3, S4 = S_bwd
    mik = ComplexF64(0, -k)
    ik  = ComplexF64(0,  k)
    S11 = S2 / mik
    S22 = S1 / mik
    S12 = S3 / ik
    S21 = S4 / ik
    return (-S11 + S22 - im*S12 - im*S21) / sqrt(2)
end

# ──────────────────────────────────────────────────────────────────────
#  Euler-β grid (matches viem_results/run_viem.jl / postprocess.jl)
#  cos β equally spaced in (-1, 1), N_β equal-area divisions.
# ──────────────────────────────────────────────────────────────────────
function euler_beta_grid(N_beta::Int)
    cos_beta = range(1 - 1/N_beta, -1 + 1/N_beta, length=N_beta)
    return acos.(cos_beta)
end

# ──────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────
_log(msg) = println("[$(Dates.format(now(), "HH:MM:SS"))] $msg")

# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
function main()
    length(ARGS) >= 1 || error("Usage: julia --project=/path/to/MSTMforCAS.jl " *
        "viem_results/paper/run_mstm_reference.jl <viem_doublet.hdf5> [out.hdf5]")
    in_path = ARGS[1]
    isfile(in_path) || error("Input HDF5 not found: $in_path")

    out_path = length(ARGS) >= 2 ? ARGS[2] :
        joinpath(dirname(in_path), "mstm_" * basename(in_path))

    _log("MSTM reference for $in_path → $out_path")

    # ── Read sweep config from VIEM doublet HDF5 ──────────────────────
    a_eq_list = Float64[]
    m_p_xyz   = ComplexF64[]
    wl_0      = 0.0
    m_m       = 0.0
    N_beta    = 0
    shape_kind_in = ""
    h5open(in_path, "r") do f
        t = f["target"]
        shape_kind_in = haskey(attributes(t), "shape_kind") ?
            String(read_attribute(t, "shape_kind")) : "gre"
        shape_kind_in == "doublet" ||
            error("Input HDF5 is shape_kind=\"$shape_kind_in\", not " *
                  "\"doublet\".  Refusing to compute MSTM reference.")

        wl_pairs = read(t["wl_m_m_pairs"])
        wl_0 = wl_pairs[1, 1]
        m_m  = wl_pairs[1, 2]

        m_p_xyz = ComplexF64.(read(t["m_p_xyz_list"])[1, :])
        a_eq_list = Float64.(read(t["r_v_base_list"]))
        N_beta = Int(read_attribute(t, "N_beta_ori"))
    end

    # MSTM treats the medium as homogeneous, so the relative refractive
    # index is m_p / m_m.  For an isotropic doublet we take m_p_xyz[1]
    # (paper sweeps are isotropic — anisotropy is not meaningful for
    # spherical monomers).
    m_p   = m_p_xyz[1]
    m_rel = ComplexF64(m_p) / ComplexF64(m_m)
    k_med = 2π * m_m / wl_0

    β_list = euler_beta_grid(N_beta)
    R_list = a_eq_list ./ 2.0^(1/3)
    g_list = 0.1 .* R_list

    _log("config: wl_0=$wl_0 μm  m_m=$m_m  m_p=$m_p  m_rel=$m_rel")
    _log("a_eq sweep: $a_eq_list  →  R = $(round.(R_list, digits=5))  " *
         "gap = $(round.(g_list, digits=5))")
    _log("Euler β grid (N_β=$N_beta): $(round.(β_list, digits=4))")
    _log("MSTM: tol=$MSTM_TOL  truncation_order=$MSTM_TRUNC_ORDER  " *
         "use_fft=$MSTM_USE_FFT")

    n_rv = length(a_eq_list)
    n_β  = length(β_list)

    Q_ext = fill(NaN, n_rv, n_β); Q_sca = similar(Q_ext); Q_abs = similar(Q_ext)
    C_ext = fill(NaN, n_rv, n_β); C_sca = similar(C_ext); C_abs = similar(C_ext)
    S_fw_theta = fill(NaN+NaN*im, n_rv, n_β)
    S_fw_phi   = similar(S_fw_theta)
    S_fw_mean  = similar(S_fw_theta)
    S_bk       = similar(S_fw_theta)
    n_iters    = zeros(Int, n_rv, n_β)
    converged  = zeros(Int, n_rv, n_β)

    t_total = @elapsed for i_rv in 1:n_rv
        R   = R_list[i_rv]
        gap = g_list[i_rv]
        geom_area = π * a_eq_list[i_rv]^2
        for (i_β, β) in enumerate(β_list)
            positions, radii = doublet_centres(R, gap, β)
            agg = AggregateGeometry(positions, radii, 2,
                                    "paper_doublet_iv$(i_rv)_iβ$(i_β)")
            t_one = @elapsed begin
                result, _ = compute_scattering(agg, m_rel, k_med;
                                use_fft          = MSTM_USE_FFT,
                                tol              = MSTM_TOL,
                                truncation_order = MSTM_TRUNC_ORDER)
            end

            S_fwθ, S_fwφ, S_fwm = bh83_to_cas_v2_forward(result.S_forward, k_med)
            S_bkv               = bh83_to_cas_v2_backward(result.S_backward, k_med)

            Q_ext[i_rv,i_β] = result.Q_ext
            Q_sca[i_rv,i_β] = result.Q_sca
            Q_abs[i_rv,i_β] = result.Q_abs
            C_ext[i_rv,i_β] = result.Q_ext * geom_area
            C_sca[i_rv,i_β] = result.Q_sca * geom_area
            C_abs[i_rv,i_β] = result.Q_abs * geom_area
            S_fw_theta[i_rv,i_β] = S_fwθ
            S_fw_phi[i_rv,i_β]   = S_fwφ
            S_fw_mean[i_rv,i_β]  = S_fwm
            S_bk[i_rv,i_β]       = S_bkv
            n_iters[i_rv,i_β]    = result.n_iterations
            converged[i_rv,i_β]  = 1   # MSTMforCAS throws on non-convergence

            @printf("  i_rv=%d β=%.4f  Q_ext=%.4e |S_fw_mean|=%.4e |S_bk|=%.4e iters=%d (%.2fs)\n",
                    i_rv, β, result.Q_ext, abs(S_fwm), abs(S_bkv),
                    result.n_iterations, t_one)
            flush(stdout)
        end
    end
    _log(@sprintf("MSTM total wall time: %.1fs", t_total))

    # ── Write output HDF5 ────────────────────────────────────────────
    h5open(out_path, "w") do f
        g = create_group(f, "target")
        attrs(g)["scattering_code"] = "MSTMforCAS.jl (multi-sphere T-matrix, exact reference)"
        attrs(g)["source_viem_h5"]  = abspath(in_path)
        attrs(g)["shape_kind"]      = "doublet"
        attrs(g)["wl_0_um"]         = wl_0
        attrs(g)["m_m"]             = m_m
        attrs(g)["truncation_order"]= MSTM_TRUNC_ORDER
        attrs(g)["solver_tol"]      = MSTM_TOL
        attrs(g)["use_fft"]         = MSTM_USE_FFT
        attrs(g)["units"]           = "C:[um^2], S:[um], beta:[rad], a_eq:[um]"
        attrs(g)["S_definition"]    = ("S(0)_theta = (S2-iS3)/(-ik), " *
                                       "S(0)_phi   = (S1+iS4)/(-ik), " *
                                       "S_fw_mean  = (S(0)_theta + S(0)_phi)/2, " *
                                       "S_bk = (S11+S22+i*S12-i*S21)(180°)/sqrt(2)")
        attrs(g)["geometry"]        = "doublet, monomer R = a_eq / 2^(1/3), " *
                                      "gap (surface-to-surface) = 0.1 R, " *
                                      "axis at rest = +z, β = angle between " *
                                      "incidence (+z lab) and doublet axis"

        write_dataset(g, "m_p",          ComplexF64.(m_p_xyz))
        write_dataset(g, "a_eq_um",      Float64.(a_eq_list))
        write_dataset(g, "R_monomer_um", Float64.(R_list))
        write_dataset(g, "gap_um",       Float64.(g_list))
        write_dataset(g, "beta_rad",     Float64.(β_list))

        obs = create_group(g, "observables")
        write_dataset(obs, "Q_ext",      Q_ext)
        write_dataset(obs, "Q_sca",      Q_sca)
        write_dataset(obs, "Q_abs",      Q_abs)
        write_dataset(obs, "C_ext",      C_ext)
        write_dataset(obs, "C_sca",      C_sca)
        write_dataset(obs, "C_abs",      C_abs)
        write_dataset(obs, "S_fw_theta", S_fw_theta)
        write_dataset(obs, "S_fw_phi",   S_fw_phi)
        write_dataset(obs, "S_fw_mean",  S_fw_mean)
        write_dataset(obs, "S_bk",       S_bk)

        diag = create_group(g, "diagnostics")
        write_dataset(diag, "n_iterations", n_iters)
        write_dataset(diag, "converged",    converged)
    end

    _log("Wrote $out_path")
    _log("Indexed by [i_rv, i_β]; cross-reference VIEM via " *
         "viem_results/paper/$(basename(in_path)) using r_v_base_list and the " *
         "spheroid-mode β grid (acos((1-2i_β-1)/N_β-style); see code).")
end

main()
