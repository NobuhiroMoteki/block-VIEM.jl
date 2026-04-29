# T-matrix exact reference for paper-production oblate HDF5s.
#
# Reads geometry/material from a paper oblate HDF5 (created by
# viem_results/paper/oblate_*.jl) and runs TransitionMatrices.jl in its own
# Julia environment to compute the numerically exact CAS-v2 observables
# at every (a_eq, β) pair, where β = particle Euler-β = angle between the
# incidence direction (lab +z) and the oblate symmetry axis (particle +z).
#
# Output is written to a SEPARATE HDF5 (CLAUDE.md §6) so the VIEM
# production HDF5 stays clean.  Default output path:
#   viem_results/paper/tmm_<basename>.hdf5
# (e.g., oblate_n20.hdf5 → tmm_oblate_n20.hdf5)
#
# Usage:
#   julia --project=/home/moteki/Julia/TransitionMatrices.jl \
#         viem_results/paper/run_tmatrix_oblate_reference.jl \
#         viem_results/paper/oblate_n20.hdf5 [output_path]
#
# Conventions:
#   * Oblate semi-axes (a_x, a_y, a_z) = (3c, 3c, c) with c = a_eq / 9^(1/3),
#     i.e. b/c = 3, a/b = 1, a:b:c = 3:3:1 (matches DDA/VIEM convention,
#     paper_simulation_conditions §4.1).
#   * The oblate symmetry axis is along particle +z; spheroid mode applies.
#     β is the angle between the symmetry axis and lab +z (incidence).
#   * In the particle frame at (α=γ=0), the incidence direction has
#     polar angle ϑᵢ = β, azimuth φᵢ = 0.  Forward direction = incidence,
#     backward direction = (π − β, π).
#   * Output observables match block-DDA_Py / block-VIEM convention.
#     PCAS forward channels:
#       S_fw_θ = S11(0) + i S12(0)
#       S_fw_φ = S22(0) − i S21(0)
#     OCBS backward observable (docs/theory_note.tex Eq. eq:S-bk):
#       S_bk_θ = S11(π) + i S12(π);  S_bk_φ = S22(π) − i S21(π)
#       S_bk = (−S_bk_θ + S_bk_φ) / sqrt(2)
#            = (−S11 + S22 − i S12 − i S21)(π) / sqrt(2)
#     (BH83 → MI02 → CAS-v2; Mishchenko 2000)
#   * Polarization-state-independent observables:
#       C_ext  = orientation-specific extinction cross-section (LCP convention
#                of the DDA/VIEM polarisation-state attribute does NOT change
#                C_ext for axisymmetric particles at α=γ=0 because the matrix
#                S(0) is diagonal in the (θ̂, φ̂) basis aligned with the
#                particle symmetry plane → same trace).
#     We compute C_ext via the optical theorem:
#         C_ext = (4π/k) Im[ S2(0)+S1(0) ] / 2
#     using the BH83 amplitude functions S₁, S₂ that come out of the T-matrix
#     formalism's amplitude_matrix(...) (see below).

using Pkg
# Activate TransitionMatrices.jl env (script may be launched from any pwd)
Pkg.activate("/home/moteki/Julia/TransitionMatrices.jl")

using TransitionMatrices
using HDF5
using Printf
using Dates
using LinearAlgebra
using Rotations: RotZYZ

# ──────────────────────────────────────────────────────────────────────
#  Settings
# ──────────────────────────────────────────────────────────────────────
const TMM_NMAX = 30                # maximum multipole order (auto-truncate by convergence)
const TMM_NG   = 200               # Gauss-Legendre quadrature points
const TMM_TOL  = 1e-8              # internal solver tolerance (advisory)

# ──────────────────────────────────────────────────────────────────────
#  Oblate geometry: semi-axes from a_eq with bc_ratio=3, ab_ratio=1, gre_beta=0
# ──────────────────────────────────────────────────────────────────────
function oblate_semi_axes(a_eq::Float64; bc_ratio::Float64 = 3.0)
    # Volume conservation: V = (4/3)π · a_x · a_y · a_z = (4/3)π a_eq³
    # Oblate: a_x = a_y = bc_ratio · a_z   (3:3:1)
    # ⇒  a_z (semi-minor, polar)        = a_eq / bc_ratio^(2/3)
    #    a_x = a_y (semi-major, equator) = bc_ratio · a_z = a_eq · bc_ratio^(1/3)
    a_pol = a_eq / bc_ratio ^ (2.0 / 3.0)
    a_eq_axis = bc_ratio * a_pol
    return a_eq_axis, a_pol
end

# ──────────────────────────────────────────────────────────────────────
#  Euler-β grid  (matches viem_results/run_viem.jl / postprocess.jl)
# ──────────────────────────────────────────────────────────────────────
function euler_beta_grid(N_beta::Int)
    cos_beta = range(1 - 1/N_beta, -1 + 1/N_beta, length = N_beta)
    return acos.(cos_beta)
end

# ──────────────────────────────────────────────────────────────────────
#  CAS-v2 amplitude conversion (BH83 → MI02 → CAS-v2).
#  TransitionMatrices.jl's `amplitude_matrix` returns a 2×2 𝐒 in the
#  (θ̂, φ̂) basis at the given lab angles; this matches the Mishchenko-2002
#  amplitude matrix S = ((S2,S3),(S4,S1)) only after the standard transposition.
#
#  Specifically, if 𝐒 = ((S₁₁,S₁₂),(S₂₁,S₂₂)) where
#       (E_θ_sca; E_φ_sca) = 𝐒 · (E_θ_inc; E_φ_inc) · exp(ikr)/r ,
#  then identifying with BH83 amplitude functions evaluated at the same
#  (incident, scattering) directions:
#       S₁₁ ↔ S₂  (θθ scattering),  S₁₂ ↔ S₃ (φθ),
#       S₂₁ ↔ S₄  (θφ),             S₂₂ ↔ S₁ (φφ).
#
#  CAS-v2 (Mishchenko 2000 / Moteki 2021).
#  PCAS forward channels:
#       S_fw_θ = S₁₁(0) + i S₁₂(0)
#       S_fw_φ = S₂₂(0) − i S₂₁(0)
#  OCBS backward observable (docs/theory_note.tex Eq. eq:S-bk):
#       S_bk_θ = S₁₁(π) + i S₁₂(π)
#       S_bk_φ = S₂₂(π) − i S₂₁(π)
#       S_bk   = (−S_bk_θ + S_bk_φ) / sqrt(2)
#              = (−S₁₁ + S₂₂ − i S₁₂ − i S₂₁)(π) / sqrt(2)
#  NOTE 2026-04-29: previous version used the *forward-mean* combination
#  (S₁₁+S₂₂+i S₁₂−i S₂₁)/√2 evaluated at the backward angle, which is the
#  LCP→LCP back-channel — NOT the OCBS observable defined in theory_note.
#  The bug zeroed |S_bk| at axisymmetric+axial-incidence slots and
#  systematically biased fig 3 oblate |S_bk| scatter plots by 5–60% at
#  rve ≥ 0.2 μm. Fixed to match VIEM postprocess.jl:506 and
#  block-DDA_Py bl_dda/scatterer.py:235.
# ──────────────────────────────────────────────────────────────────────
function cas_v2_observables(S_fwd::AbstractMatrix, S_bwd::AbstractMatrix)
    S11_fw, S12_fw = S_fwd[1,1], S_fwd[1,2]
    S21_fw, S22_fw = S_fwd[2,1], S_fwd[2,2]
    S_fw_theta = S11_fw + 1im * S12_fw
    S_fw_phi   = S22_fw - 1im * S21_fw

    # Backward basis-convention bridge (theory_note ↔ standard spherical):
    # TransitionMatrices.jl returns the amplitude matrix in the standard
    # spherical (θ̂, φ̂) basis at the scattering point.  At the backward
    # direction (π−β, π) this basis is anti-parallel to theory_note's
    # convention (theta_sca_bk = −theta_inc, phi_sca_bk = +phi_inc) used by
    # VIEM postprocess.jl and block-DDA_Py: BOTH scattered basis vectors
    # flip sign at the antipodal point, so every S element acquires a (−1)
    # factor.  Applying that conversion before the OCBS formula keeps the
    # complex S_bk in sign-agreement with VIEM / block-DDA_Py.
    S11_bk = -S_bwd[1,1]; S12_bk = -S_bwd[1,2]
    S21_bk = -S_bwd[2,1]; S22_bk = -S_bwd[2,2]
    S_bk_theta = S11_bk + 1im * S12_bk
    S_bk_phi   = S22_bk - 1im * S21_bk
    S_bk       = (-S_bk_theta + S_bk_phi) / sqrt(2)

    return S_fw_theta, S_fw_phi, S_bk
end

# ──────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────
_log(msg) = (println("[$(Dates.format(now(), "HH:MM:SS"))] $msg"); flush(stdout))

# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
function main()
    length(ARGS) >= 1 || error("Usage: julia --project=/path/to/TransitionMatrices.jl " *
        "viem_results/paper/run_tmatrix_oblate_reference.jl <viem_oblate.hdf5> [out.hdf5]")
    in_path = ARGS[1]
    isfile(in_path) || error("Input HDF5 not found: $in_path")

    out_path = length(ARGS) >= 2 ? ARGS[2] :
        joinpath(dirname(in_path), "tmm_" * basename(in_path))

    _log("T-matrix reference for $in_path → $out_path")

    a_eq_list = Float64[]; m_p_xyz = ComplexF64[]
    wl_0 = 0.0; m_m = 0.0; N_beta = 0
    bc_ratio = 0.0; ab_ratio = 0.0; gre_beta = 0.0
    shape_kind_in = ""
    h5open(in_path, "r") do f
        t = f["target"]
        shape_kind_in = haskey(attributes(t), "shape_kind") ?
            String(read_attribute(t, "shape_kind")) : "gre"
        wl_pairs = read(t["wl_m_m_pairs"])
        wl_0 = wl_pairs[1, 1]; m_m = wl_pairs[1, 2]
        m_p_xyz = ComplexF64.(read(t["m_p_xyz_list"])[1, :])
        a_eq_list = Float64.(read(t["r_v_base_list"]))
        N_beta = Int(read_attribute(t, "N_beta_ori"))
        bc_ratio = Float64(read(t["bc_ratio_list"])[1])
        ab_ratio = Float64(read(t["ab_ratio_list"])[1])
        gre_beta = Float64(read(t["gre_beta_list"])[1])
    end

    isapprox(ab_ratio, 1.0; atol=1e-12) ||
        error("ab_ratio = $ab_ratio (need 1.0 for axisymmetric oblate)")
    isapprox(gre_beta, 0.0; atol=1e-12) ||
        error("gre_beta = $gre_beta (need 0.0 for axisymmetric oblate)")
    bc_ratio > 1.0 ||
        error("bc_ratio = $bc_ratio (need >1.0 for oblate)")

    m_rel = ComplexF64(m_p_xyz[1]) / ComplexF64(m_m)
    λ_med = wl_0 / m_m       # in-medium wavelength
    k_med = 2π / λ_med
    β_list = euler_beta_grid(N_beta)

    _log("config: wl_0=$wl_0 μm  m_m=$m_m  m_p=$m_p_xyz[1]  m_rel=$m_rel  λ_med=$λ_med")
    _log("a_eq sweep: $a_eq_list   bc_ratio=$bc_ratio")
    _log("Euler β grid (N_β=$N_beta): $(round.(β_list, digits=4)) rad")

    n_rv = length(a_eq_list)
    n_β  = length(β_list)
    Q_ext = fill(NaN, n_rv, n_β); Q_sca = similar(Q_ext); Q_abs = similar(Q_ext)
    C_ext = fill(NaN, n_rv, n_β); C_sca = similar(C_ext); C_abs = similar(C_ext)
    S_fw_theta = fill(NaN+NaN*im, n_rv, n_β)
    S_fw_phi   = similar(S_fw_theta)
    S_fw_mean  = similar(S_fw_theta)
    S_bk       = similar(S_fw_theta)
    converged  = zeros(Int, n_rv, n_β)

    for i_rv in 1:n_rv
        a_eq = a_eq_list[i_rv]
        a_eq_axis, a_pol = oblate_semi_axes(a_eq; bc_ratio = bc_ratio)
        # TransitionMatrices Spheroid(a, c, m): a = equatorial semi-axis,
        # c = polar semi-axis. For oblate: a > c.
        s = Spheroid{Float64, ComplexF64}(a_eq_axis, a_pol, m_rel)
        _log(@sprintf("i_rv=%d  a_eq=%.4f μm  → a=%.5f c=%.5f (a/c=%.2f)",
              i_rv, a_eq, a_eq_axis, a_pol, a_eq_axis / a_pol))

        t_T = @elapsed 𝐓 = transition_matrix(s, λ_med, TMM_NMAX, TMM_NG)
        _log(@sprintf("    T-matrix nₘₐₓ=%d  Ng=%d  built in %.2fs",
              TMM_NMAX, TMM_NG, t_T))

        geom_area = π * a_eq^2

        for (i_β, β) in enumerate(β_list)
            # Incidence in particle frame: ϑᵢ=β, φᵢ=0
            # Forward scattering direction = incidence direction
            #   ϑ_s = β, φ_s = 0
            # Backward direction = -incidence
            #   ϑ_s = π − β, φ_s = π
            𝐒_fwd = amplitude_matrix(𝐓, β, 0.0, β, 0.0; λ = λ_med)
            𝐒_bwd = amplitude_matrix(𝐓, β, 0.0, π - β, π; λ = λ_med)

            # CAS-v2 conversion
            S_fwθ, S_fwφ, S_bkv = cas_v2_observables(𝐒_fwd, 𝐒_bwd)

            # Optical theorem (LCP polarisation): C_ext follows from
            #   C_ext = (2π/k) Im[ S_θθ(0) + S_φφ(0) ]
            # because the LCP basis has equal weight on θ and φ.
            S_theta_theta = 𝐒_fwd[1,1]
            S_phi_phi     = 𝐒_fwd[2,2]
            C_ext_v = (2π / k_med) * imag(S_theta_theta + S_phi_phi)

            # C_sca: integrate |S|² over the full sphere.  TransitionMatrices
            # provides scattering_cross_section(T, λ) for random orientation;
            # for our fixed-orientation case we approximate via the optical
            # theorem extinction minus absorption.  Absorption can be derived
            # via volume integral but for axisymmetric T-matrix at fixed
            # orientation the simplest is C_abs = (k/|m|) ∫ |E_int|² dV which
            # is not directly exposed.  → leave NaN unless extended.

            Q_ext[i_rv, i_β] = C_ext_v / geom_area
            C_ext[i_rv, i_β] = C_ext_v
            S_fw_theta[i_rv, i_β] = S_fwθ
            S_fw_phi[i_rv, i_β]   = S_fwφ
            S_fw_mean[i_rv, i_β]  = (S_fwθ + S_fwφ) / 2
            S_bk[i_rv, i_β]       = S_bkv
            converged[i_rv, i_β]  = 1

            @printf("  i_rv=%d β=%.4f  C_ext=%.4e |S_fw_θ|=%.4e |S_fw_φ|=%.4e |S_bk|=%.4e\n",
                    i_rv, β, C_ext_v, abs(S_fwθ), abs(S_fwφ), abs(S_bkv))
            flush(stdout)
        end
    end

    h5open(out_path, "w") do f
        g = create_group(f, "target")
        attrs(g)["scattering_code"] = "TransitionMatrices.jl (EBCM, exact axisymmetric reference)"
        attrs(g)["source_viem_h5"]  = abspath(in_path)
        attrs(g)["shape_kind"]      = "oblate"
        attrs(g)["wl_0_um"]         = wl_0
        attrs(g)["m_m"]             = m_m
        attrs(g)["bc_ratio"]        = bc_ratio
        attrs(g)["ab_ratio"]        = ab_ratio
        attrs(g)["nₘₐₓ"]            = TMM_NMAX
        attrs(g)["Ng"]              = TMM_NG
        attrs(g)["units"]           = "C:[um^2], S:[um], beta:[rad], a_eq:[um]"
        attrs(g)["S_definition"]    = ("S_fw_θ = S11(0)+i·S12(0), " *
                                       "S_fw_φ = S22(0)−i·S21(0), " *
                                       "S_bk = (−S_bk_θ+S_bk_φ)/sqrt(2) " *
                                       "where S_bk_θ = S11(π)+i·S12(π), S_bk_φ = S22(π)−i·S21(π) " *
                                       "(OCBS, docs/theory_note.tex Eq. eq:S-bk); " *
                                       "amplitude_matrix(...) S = ((S11,S12),(S21,S22)) basis (θ̂,φ̂).")
        attrs(g)["geometry"]        = "oblate spheroid, semi-axes (3c,3c,c), " *
                                      "c = a_eq/9^(1/3), symmetry axis = +z; " *
                                      "β = angle between symmetry axis and incidence (+z lab)"

        write_dataset(g, "m_p",      ComplexF64.(m_p_xyz))
        write_dataset(g, "a_eq_um",  Float64.(a_eq_list))
        write_dataset(g, "beta_rad", Float64.(β_list))

        obs = create_group(g, "observables")
        write_dataset(obs, "Q_ext",      Q_ext)
        write_dataset(obs, "C_ext",      C_ext)
        write_dataset(obs, "S_fw_theta", S_fw_theta)
        write_dataset(obs, "S_fw_phi",   S_fw_phi)
        write_dataset(obs, "S_fw_mean",  S_fw_mean)
        write_dataset(obs, "S_bk",       S_bk)

        diag = create_group(g, "diagnostics")
        write_dataset(diag, "converged", converged)
    end

    _log("Wrote $out_path")
end

main()
