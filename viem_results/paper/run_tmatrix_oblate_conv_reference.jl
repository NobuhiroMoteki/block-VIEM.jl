# T-matrix exact reference for the FIG-4 ℓ_c convergence sweep on oblate.
#
# Existing tmm_oblate_<mat>.hdf5 (produced by run_tmatrix_oblate_reference.jl)
# carries the production β grid (cos β ∈ {±0.8, ±0.4, 0}, β=0 is excluded).
# The convergence-sweep observables in convergence_oblate_<mat>.hdf5 are at
# β=0 single orientation, so the production TMM grid cannot be used as the
# fig-4 reference.  This script fills exactly that gap: TMM at (a_eq=0.1 μm,
# β=0) per material, written to tmm_oblate_conv_<mat>.hdf5 with a minimal
# (1,1) (a_eq, β) grid layout that matches the existing schema.
#
# Output schema mirrors run_tmatrix_oblate_reference.jl so the same
# block-DDA_Py side loader (_plot_io.load_tmm) can read it; a_eq_um and
# beta_rad are simply length-1 arrays.
#
# Usage:
#   julia --project=/home/moteki/Julia/TransitionMatrices.jl \
#         viem_results/paper/run_tmatrix_oblate_conv_reference.jl <material>
#   <material> ∈ {n15, n20, Au}
# Without arguments, runs all three materials in sequence.

using Pkg
Pkg.activate("/home/moteki/Julia/TransitionMatrices.jl")

using TransitionMatrices
using HDF5
using Printf
using Dates
using Rotations: RotZYZ

# Convergence-sweep slot
const A_EQ_CONV = 0.10    # μm  (matches viem_results/paper/run_lc_convergence.jl)
const BETA_CONV = 0.0     # rad (single-orientation ZYZ identity)
const WL_0      = 0.638   # μm
const M_M       = 1.0
const BC_RATIO  = 3.0     # oblate 3:3:1

# TMM solver settings — same as production reference.
const TMM_NMAX = 30
const TMM_NG   = 200

# Material → m_p (matches CLAUDE.md §2 and viem_results/paper/_common.jl)
const M_P_MAP = Dict(
    "n15" => ComplexF64(1.5,    0.01),
    "n20" => ComplexF64(2.0,    0.0),
    "Au"  => ComplexF64(0.17525, 3.4830),
)

function oblate_semi_axes(a_eq::Float64; bc_ratio::Float64 = BC_RATIO)
    a_pol = a_eq / bc_ratio ^ (2.0 / 3.0)
    a_eq_axis = bc_ratio * a_pol
    return a_eq_axis, a_pol
end

function cas_v2_observables(S_fwd::AbstractMatrix, S_bwd::AbstractMatrix)
    # PCAS forward channels and OCBS backward (docs/theory_note.tex Eq. eq:S-bk):
    #   S_bk = (−S_bk_θ + S_bk_φ) / √2,
    #   S_bk_θ = S11(π) + i S12(π);  S_bk_φ = S22(π) − i S21(π).
    # Backward basis-convention bridge: TransitionMatrices.jl returns the
    # scattering matrix in the standard spherical basis at (θ_s, φ_s). At
    # the backward direction (π−β, π) this basis is anti-parallel to
    # theory_note's convention (theta_sca_bk = −theta_inc, phi_sca_bk =
    # +phi_inc) used by VIEM postprocess.jl and block-DDA_Py: BOTH
    # scattered basis vectors flip sign, so every S element acquires a
    # (−1) factor.  Apply that conversion before the OCBS formula.
    S11_fw, S12_fw = S_fwd[1,1], S_fwd[1,2]
    S21_fw, S22_fw = S_fwd[2,1], S_fwd[2,2]
    S_fw_theta = S11_fw + 1im * S12_fw
    S_fw_phi   = S22_fw - 1im * S21_fw

    S11_bk = -S_bwd[1,1]; S12_bk = -S_bwd[1,2]
    S21_bk = -S_bwd[2,1]; S22_bk = -S_bwd[2,2]
    S_bk_theta = S11_bk + 1im * S12_bk
    S_bk_phi   = S22_bk - 1im * S21_bk
    S_bk       = (-S_bk_theta + S_bk_phi) / sqrt(2)

    return S_fw_theta, S_fw_phi, S_bk
end

_log(msg) = (println("[$(Dates.format(now(), "HH:MM:SS"))] $msg"); flush(stdout))

function run_one(material::String, out_dir::String)
    haskey(M_P_MAP, material) ||
        error("unknown material '$material'; expected one of $(collect(keys(M_P_MAP)))")

    m_p = M_P_MAP[material]
    m_rel = m_p / ComplexF64(M_M)
    λ_med = WL_0 / M_M
    k_med = 2π / λ_med

    a_eq = A_EQ_CONV
    β    = BETA_CONV
    a_eq_axis, a_pol = oblate_semi_axes(a_eq)

    _log("[$material] a_eq=$a_eq μm  m_p=$m_p  m_rel=$m_rel")
    _log("[$material] a=$a_eq_axis  c=$a_pol  (a/c=$(a_eq_axis/a_pol))")

    s = Spheroid{Float64, ComplexF64}(a_eq_axis, a_pol, m_rel)
    t_T = @elapsed 𝐓 = transition_matrix(s, λ_med, TMM_NMAX, TMM_NG)
    _log(@sprintf("[%s] T-matrix nₘₐₓ=%d  Ng=%d  built in %.2fs",
                   material, TMM_NMAX, TMM_NG, t_T))

    # Forward = incidence direction in particle frame; for β=0 this is
    # the symmetry axis +z, ϑ=0 (degenerate spherical-cap → use ϑ=β=0,
    # φ=0 by convention; amplitude_matrix handles the limit).
    𝐒_fwd = amplitude_matrix(𝐓, β, 0.0,        β,        0.0; λ = λ_med)
    𝐒_bwd = amplitude_matrix(𝐓, β, 0.0, π - β, π;          λ = λ_med)

    S_fwθ, S_fwφ, S_bkv = cas_v2_observables(𝐒_fwd, 𝐒_bwd)

    S_theta_theta = 𝐒_fwd[1,1]
    S_phi_phi     = 𝐒_fwd[2,2]
    C_ext_v = (2π / k_med) * imag(S_theta_theta + S_phi_phi)
    geom_area = π * a_eq^2
    Q_ext_v   = C_ext_v / geom_area

    @printf("[%s] Q_ext=%.6f  C_ext=%.4e  |S_fw_θ|=%.4e  |S_fw_φ|=%.4e  |S_bk|=%.4e\n",
            material, Q_ext_v, C_ext_v, abs(S_fwθ), abs(S_fwφ), abs(S_bkv))

    # Layout: (n_rv=1, n_β=1) — matches run_tmatrix_oblate_reference.jl
    Q_ext   = reshape([Q_ext_v], (1, 1))
    C_ext   = reshape([C_ext_v], (1, 1))
    S_fw_th = reshape([S_fwθ],   (1, 1))
    S_fw_ph = reshape([S_fwφ],   (1, 1))
    S_fw_mn = reshape([(S_fwθ + S_fwφ)/2], (1, 1))
    S_bk_a  = reshape([S_bkv],   (1, 1))
    converged = ones(Int, 1, 1)

    out_path = joinpath(out_dir, "tmm_oblate_conv_$(material).hdf5")
    h5open(out_path, "w") do f
        g = create_group(f, "target")
        attrs(g)["scattering_code"] = "TransitionMatrices.jl (EBCM, exact axisymmetric reference)"
        attrs(g)["purpose"]         = "fig-4 ℓ_c convergence reference at β=0, a_eq=0.1 μm"
        attrs(g)["shape_kind"]      = "oblate"
        attrs(g)["wl_0_um"]         = WL_0
        attrs(g)["m_m"]             = M_M
        attrs(g)["bc_ratio"]        = BC_RATIO
        attrs(g)["ab_ratio"]        = 1.0
        attrs(g)["nₘₐₓ"]            = TMM_NMAX
        attrs(g)["Ng"]              = TMM_NG
        attrs(g)["units"]           = "C:[um^2], S:[um], beta:[rad], a_eq:[um]"

        write_dataset(g, "m_p",      ComplexF64.([m_p, m_p, m_p]))
        write_dataset(g, "a_eq_um",  Float64.([a_eq]))
        write_dataset(g, "beta_rad", Float64.([β]))

        obs = create_group(g, "observables")
        write_dataset(obs, "Q_ext",      Q_ext)
        write_dataset(obs, "C_ext",      C_ext)
        write_dataset(obs, "S_fw_theta", S_fw_th)
        write_dataset(obs, "S_fw_phi",   S_fw_ph)
        write_dataset(obs, "S_fw_mean",  S_fw_mn)
        write_dataset(obs, "S_bk",       S_bk_a)

        diag = create_group(g, "diagnostics")
        write_dataset(diag, "converged", converged)
    end
    _log("[$material] wrote $out_path")
    return out_path
end

function main()
    out_dir = "/home/moteki/Julia/block-VIEM.jl/viem_results/paper"
    mats = isempty(ARGS) ? ["n15", "n20", "Au"] : ARGS
    for m in mats
        run_one(m, out_dir)
    end
end

main()
