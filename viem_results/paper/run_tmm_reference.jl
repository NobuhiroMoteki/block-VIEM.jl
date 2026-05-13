# T-matrix (EBCM) exact reference for paper-production oblate HDF5s.
#
# Reads geometry/material from a paper oblate HDF5 (created by
# block-DDA_Py / block-VIEM.jl side) and runs TransitionMatrices.jl in its
# own Julia environment to compute numerically exact CAS-v2 observables at
# every (a_eq, β) pair, where β = particle Euler-β = angle between
# incidence direction (lab +z) and the oblate symmetry axis (body +z).
#
# Output is written to a SEPARATE HDF5 (consistent with `run_mstm_reference.jl`):
#   viem_results/paper/tmm_oblate_<material>.hdf5
#
# Usage:
#   julia --project=viem_results/paper/tmm_env \
#         viem_results/paper/run_tmm_reference.jl \
#         viem_results/paper/oblate_n15.hdf5 [out.hdf5]
#
# Conventions:
#   * Oblate symmetry axis at rest = +z; rotated by β about y so it makes
#     angle β with lab +z (matches block-VIEM.jl spheroid mode at α=γ=0).
#   * Semi-axes a:b:c = 3:3:1 (b/c = 3 from `oblate_*.py`).
#     Volume preservation: V = (4π/3)·a²·c = (4π/3)·a_eq³.
#     With a = b and c = a/3 ⇒ a = a_eq · 3^(1/3), c = a_eq · 3^(-2/3).
#   * Amplitude convention: TransitionMatrices.amplitude_matrix returns
#     Mishchenko (2002) F-matrix scaled by 1/k₁ (length units), basis
#     (E_θ, E_φ) tied to lab incidence direction. Empirically verified:
#     F_TM[i,j] = S_BH[i,j] / (-i·k), where S_BH is BH83's [S2 S3; S4 S1].
#   * CAS-v2 forward / backward formulas (matching `run_mstm_reference.jl`):
#       S_fw_theta = (S2 - i·S3) / (-i·k)
#                  = F_TM_fw[1,1] - i·F_TM_fw[1,2]
#       S_fw_phi   = (S1 + i·S4) / (-i·k)
#                  = F_TM_fw[2,2] + i·F_TM_fw[2,1]
#       S_bk_theta = S_M11(π) + i·S_M12(π) = F_bk[1,1] + i·F_bk[1,2]
#       S_bk_phi   = S_M22(π) - i·S_M21(π) = F_bk[2,2] - i·F_bk[2,1]
#       S_bk       = (-S_bk_theta + S_bk_phi) / √2
#                  = (-F_bk[1,1] + F_bk[2,2] - i·F_bk[1,2] - i·F_bk[2,1]) / √2
#     (per docs/theory_note.tex Eq. eq:S-bk-ocbs and bl_dda/scatterer.py)
#   * Per-orientation cross sections:
#       C_ext = (4π/k) · Im(S_fw_mean)                 (optical theorem)
#       C_sca = (1/2) ∮ (|F[1,1]|²+|F[1,2]|²+|F[2,1]|²+|F[2,2]|²) dΩ
#       C_abs = C_ext - C_sca

using TransitionMatrices
using TransitionMatrices: Spheroid, transition_matrix, amplitude_matrix
using HDF5
using Rotations: RotZYZ
using FastGaussQuadrature: gausslegendre
using StaticArrays
using Printf
using Dates
using LinearAlgebra: norm

# ──────────────────────────────────────────────────────────────────────
#  Settings
# ──────────────────────────────────────────────────────────────────────
const TM_THRESHOLD = 1e-5      # convergence threshold for n_max / Ng auto-tuning
const TM_NDGS      = 4
# Spherical quadrature grid for C_sca integral (sufficient for x ≲ 5)
const N_THETA_QUAD = 60
const N_PHI_QUAD   = 60

# ──────────────────────────────────────────────────────────────────────
#  Euler-β grid (matches viem_results/run_viem.jl spheroid mode)
# ──────────────────────────────────────────────────────────────────────
function euler_beta_grid(N_beta::Int)
    cos_beta = range(1 - 1/N_beta, -1 + 1/N_beta, length=N_beta)
    return acos.(cos_beta)
end

# ──────────────────────────────────────────────────────────────────────
#  Forward / backward amplitude → CAS-v2 (uses TM amplitude_matrix
#  output directly; algebraically equivalent to bh83_to_cas_v2_*).
# ──────────────────────────────────────────────────────────────────────
function tm_to_cas_v2_forward(F_fw::AbstractMatrix{<:Complex})
    S_fw_theta = F_fw[1, 1] - im * F_fw[1, 2]
    S_fw_phi   = F_fw[2, 2] + im * F_fw[2, 1]
    S_fw_mean  = (S_fw_theta + S_fw_phi) / 2
    return S_fw_theta, S_fw_phi, S_fw_mean
end

function tm_to_cas_v2_backward(F_bk::AbstractMatrix{<:Complex})
    # Theory-note Eq.(eq:S-bk-ocbs):
    #   S_bk_theta = S_M11(π) + i·S_M12(π) = F_bk[1,1] + i·F_bk[1,2]
    #   S_bk_phi   = S_M22(π) - i·S_M21(π) = F_bk[2,2] - i·F_bk[2,1]
    #   S_bk       = (-S_bk_theta + S_bk_phi) / √2
    return (-F_bk[1, 1] + F_bk[2, 2] - im * F_bk[1, 2] - im * F_bk[2, 1]) / sqrt(2)
end

# ──────────────────────────────────────────────────────────────────────
#  Per-orientation C_sca via spherical Gauss–Legendre quadrature.
#  Integrand: (1/2)·Σ_{ij}|F_TM[i,j]|² (unpolarized scattering matrix
#  trace, summed over 4 elements then halved for unpolarized incidence).
# ──────────────────────────────────────────────────────────────────────
function csca_quadrature(𝐓, λ_med::Real, rot::RotZYZ;
                          n_theta::Int=N_THETA_QUAD, n_phi::Int=N_PHI_QUAD)
    x_q, w_q = gausslegendre(n_theta)
    ϑ_q = acos.(x_q)
    dφ = 2π / n_phi
    φ_q = collect(range(0, 2π - dφ, length=n_phi))
    Csca = 0.0
    for (i, ϑ) in enumerate(ϑ_q)
        for φ in φ_q
            F = amplitude_matrix(𝐓, 0.0, 0.0, ϑ, φ; λ=λ_med, rot=rot)
            fsq = abs2(F[1,1]) + abs2(F[1,2]) + abs2(F[2,1]) + abs2(F[2,2])
            Csca += (fsq / 2) * w_q[i] * dφ
        end
    end
    return Csca
end

# ──────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────
_log(msg) = (println("[$(Dates.format(now(), "HH:MM:SS"))] $msg"); flush(stdout))

# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
function main()
    length(ARGS) >= 1 || error("Usage: julia --project=tmm_env " *
        "run_tmm_reference.jl <viem_oblate.hdf5> [out.hdf5]")
    in_path = ARGS[1]
    isfile(in_path) || error("Input HDF5 not found: $in_path")
    out_path = length(ARGS) >= 2 ? ARGS[2] :
        joinpath(dirname(in_path), "tmm_" * basename(in_path))

    _log("TMM reference for $in_path → $out_path")

    a_eq_list   = Float64[]
    m_p_xyz     = ComplexF64[]
    wl_0        = 0.0
    m_m         = 0.0
    N_beta      = 0
    bc_ratio    = 0.0
    ab_ratio    = 0.0
    gre_beta = 0.0
    h5open(in_path, "r") do f
        t = f["target"]
        # block-VIEM.jl encodes oblate as shape_kind="gre" + bc_ratio + ab_ratio + gre_beta=0
        # rather than a dedicated "oblate" tag. We accept either label and rely on
        # bc_ratio > 1 + gre_beta ≈ 0 to confirm an oblate spheroid.
        shape_kind_in = haskey(attributes(t), "shape_kind") ?
            String(read_attribute(t, "shape_kind")) : "?"
        spheroid_mode = haskey(attributes(t), "spheroid_mode") ?
            Int(read_attribute(t, "spheroid_mode")) : 0
        spheroid_mode == 1 ||
            error("Input HDF5 spheroid_mode=$spheroid_mode (expected 1). " *
                  "shape_kind=\"$shape_kind_in\" — refusing to compute T-matrix reference.")

        wl_pairs = read(t["wl_m_m_pairs"])
        wl_0 = wl_pairs[1, 1]
        m_m  = wl_pairs[1, 2]

        m_p_xyz   = ComplexF64.(read(t["m_p_xyz_list"])[1, :])
        a_eq_list = Float64.(read(t["r_v_base_list"]))
        N_beta    = Int(read_attribute(t, "N_beta_ori"))

        # block-VIEM.jl writes geometry as 1-element datasets, not attributes.
        bc_ratio  = Float64(read(t["bc_ratio_list"])[1])
        ab_ratio  = Float64(read(t["ab_ratio_list"])[1])
        gre_beta  = Float64(read(t["gre_beta_list"])[1])
    end

    abs(gre_beta) < 1e-12 ||
        error("gre_beta = $gre_beta ≠ 0; this script handles smooth oblate " *
              "spheroids only (no GRE deformation).")

    # Oblate (a=b, c < a) with bc_ratio = b/c, ab_ratio = a/b = 1.
    # Volume-equivalent: a²·c = a_eq³.
    abs(ab_ratio - 1.0) < 1e-10 ||
        error("ab_ratio = $ab_ratio ≠ 1; this script only handles a=b spheroids.")
    bc_ratio > 1.0 ||
        error("bc_ratio = $bc_ratio ≤ 1; expected oblate (b/c > 1).")

    # a = a_eq · bc_ratio^(1/3),  c = a / bc_ratio = a_eq · bc_ratio^(-2/3)
    a_um = a_eq_list .* bc_ratio^(1/3)
    c_um = a_eq_list .* bc_ratio^(-2/3)

    m_p   = m_p_xyz[1]
    m_rel = ComplexF64(m_p) / ComplexF64(m_m)
    λ_med = wl_0 / m_m
    k_med = 2π * m_m / wl_0     # = 2π / λ_med

    β_list = euler_beta_grid(N_beta)

    _log("config: wl_0=$wl_0 μm  m_m=$m_m  m_p=$m_p  m_rel=$m_rel")
    _log("a_eq sweep: $a_eq_list  →  (a, c) μm = $(round.(a_um, digits=5)), $(round.(c_um, digits=5))")
    _log("oblate aspect c/a = $(round(1/bc_ratio, digits=4))   (paper geometry: 1/3)")
    _log("Euler β grid (N_β=$N_beta): $(round.(β_list, digits=4))")
    _log("TMM: convergence threshold=$TM_THRESHOLD  ndgs=$TM_NDGS")
    _log("C_sca quadrature: $(N_THETA_QUAD)×$(N_PHI_QUAD)")

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
        a, c = a_um[i_rv], c_um[i_rv]
        s = Spheroid{Float64, ComplexF64}(a, c, m_rel)
        geom_area = π * a_eq_list[i_rv]^2

        local 𝐓
        t_T = @elapsed begin
            try
                𝐓 = transition_matrix(s, λ_med;
                                      threshold=TM_THRESHOLD, ndgs=TM_NDGS)
            catch e
                _log("  i_rv=$i_rv (a=$(round(a,digits=5)), c=$(round(c,digits=5))) " *
                     "transition_matrix FAILED: $e")
                continue
            end
        end
        _log("  i_rv=$i_rv  T-matrix: a=$(round(a,digits=5)) c=$(round(c,digits=5))  " *
             "($(round(t_T,digits=2))s)")

        for (i_β, β) in enumerate(β_list)
            rot = RotZYZ(0.0, β, 0.0)

            t_one = @elapsed begin
                F_fw = amplitude_matrix(𝐓, 0.0, 0.0, 0.0, 0.0; λ=λ_med, rot=rot)
                F_bk = amplitude_matrix(𝐓, 0.0, 0.0, π,   0.0; λ=λ_med, rot=rot)

                S_fwθ, S_fwφ, S_fwm = tm_to_cas_v2_forward(F_fw)
                S_bkv               = tm_to_cas_v2_backward(F_bk)

                Cext = (4π / k_med) * imag(S_fwm)
                Csca = csca_quadrature(𝐓, λ_med, rot)
                Cabs = Cext - Csca
            end

            Q_ext[i_rv, i_β] = Cext / geom_area
            Q_sca[i_rv, i_β] = Csca / geom_area
            Q_abs[i_rv, i_β] = Cabs / geom_area
            C_ext[i_rv, i_β] = Cext
            C_sca[i_rv, i_β] = Csca
            C_abs[i_rv, i_β] = Cabs
            S_fw_theta[i_rv, i_β] = S_fwθ
            S_fw_phi[i_rv,   i_β] = S_fwφ
            S_fw_mean[i_rv,  i_β] = S_fwm
            S_bk[i_rv,       i_β] = S_bkv
            n_iters[i_rv,    i_β] = 1                  # EBCM is direct, no iterations
            converged[i_rv,  i_β] = (isfinite(Cext) && isfinite(Csca) &&
                                     Cabs > -1e-3 * abs(Cext)) ? 1 : 0

            @printf("    β=%.4f  Cext=%.4e Cabs=%.4e Csca=%.4e |S_fw_mean|=%.4e |S_bk|=%.4e (%.2fs)\n",
                    β, Cext, Cabs, Csca, abs(S_fwm), abs(S_bkv), t_one)
            flush(stdout)
        end
    end
    _log(@sprintf("TMM total wall time: %.1fs", t_total))

    # ── Write output HDF5 (MSTM-compatible schema) ────────────────────
    h5open(out_path, "w") do f
        g = create_group(f, "target")
        attrs(g)["scattering_code"] = "TransitionMatrices.jl (EBCM, exact reference)"
        attrs(g)["source_viem_h5"]  = abspath(in_path)
        attrs(g)["shape_kind"]      = "oblate"
        attrs(g)["wl_0_um"]         = wl_0
        attrs(g)["m_m"]             = m_m
        attrs(g)["truncation_order"]= 0      # auto-converged; not user-fixed
        attrs(g)["solver_tol"]      = TM_THRESHOLD
        attrs(g)["use_fft"]         = UInt8(0)
        attrs(g)["units"]           = "C:[um^2], S:[um], beta:[rad], a_eq:[um]"
        attrs(g)["S_definition"]    = ("S(0)_theta = (S2-iS3)/(-ik), " *
                                       "S(0)_phi   = (S1+iS4)/(-ik), " *
                                       "S_fw_mean  = (S(0)_theta + S(0)_phi)/2, " *
                                       "S_bk = (S11+S22+i*S12-i*S21)(180°)/sqrt(2)")
        attrs(g)["geometry"]        = ("oblate spheroid, semi-axes a:b:c = $(bc_ratio):$(bc_ratio):1 " *
                                       "(a=b>c), symmetry axis = +z, " *
                                       "β = angle between incidence (+z lab) and c-axis")

        write_dataset(g, "m_p",          ComplexF64.(m_p_xyz))
        write_dataset(g, "a_eq_um",      Float64.(a_eq_list))
        write_dataset(g, "a_um",         Float64.(a_um))
        write_dataset(g, "c_um",         Float64.(c_um))
        write_dataset(g, "aspect_ratio", Float64(1/bc_ratio))    # c/a; oblate < 1
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
end

main()
