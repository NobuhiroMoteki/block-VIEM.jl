# Mesh refinement study: convergence rate of S(0), S(180), and cross sections.
#
# Computes VIEM scattering for a sphere at multiple mesh resolutions and
# compares against Mie theory to extract convergence rates.
#
# Usage: julia --project -t auto test/convergence_S0.jl

using LinearAlgebra: norm, dot
using StaticArrays
using Printf
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

include(joinpath(@__DIR__, "mie_reference.jl"))

function generate_sphere_mesh(radius::Float64, lc::Float64)
    path = joinpath(tempdir(), "sphere_conv_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere_conv")
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

include(joinpath(@__DIR__, "mie_internal_field.jl"))  # for ricatti_jn etc.

"""
    mie_scattering_coefficients_outgoing(; wl_0, m_m, r_p, m_p)

Compute Mie a_n, b_n using the OUTGOING wave ξ^out = ψ - iχ = x h^(1).

IMPORTANT: mie_reference.jl uses ξ = ψ + iχ = x h^(2) (incoming wave),
which gives correct cross sections but WRONG complex scattering amplitudes.
For S(0), S(180) we need the outgoing-wave coefficients.
"""
function mie_scattering_coefficients_outgoing(; wl_0::Real, m_m::Real, r_p::Real, m_p::Number)
    k_bg = 2π * m_m / wl_0
    x = k_bg * r_p
    m_r = ComplexF64(m_p) / m_m
    y = m_r * x

    nstop = floor(Int, abs(x) + 4 * abs(x)^(1 / 3) + 2)
    nstop = max(nstop, 3)

    # Logarithmic derivative D_n(mx) via downward recurrence
    nmx = floor(Int, max(nstop, abs(y)) + 15)
    DD = zeros(ComplexF64, nmx + 1)
    for n in nmx:-1:1
        DD[n] = n / y - 1 / (DD[n + 1] + n / y)
    end

    a = zeros(ComplexF64, nstop + 1)
    b = zeros(ComplexF64, nstop + 1)

    for n in 1:nstop
        psi_x = ricatti_jn(n, x)
        psi_x_d = ricatti_jn_deriv(n, x)

        # χ_n = -x y_n(x) (BH definition)
        chi_x = -Float64(x) * Float64(bessely(n + 0.5, Float64(x))) * sqrt(π / (2x))
        chi_x_d = begin
            chi_prev = -Float64(x) * Float64(bessely(n - 1 + 0.5, Float64(x))) * sqrt(π / (2x))
            chi_prev - n / x * chi_x
        end

        # ξ^out = ψ - iχ = x h^(1)  (outgoing, W = +i)
        xi_x = ComplexF64(psi_x) - im * chi_x
        xi_x_d = ComplexF64(psi_x_d) - im * chi_x_d

        # Use BH Eq 4.88 full form with ψ, ψ', ξ, ξ' directly
        psi_y = ricatti_jn(n, y)
        psi_y_d = ricatti_jn_deriv(n, y)

        # a_n = (m_r ψ_n(y) ψ_n'(x) - ψ_n'(y) ψ_n(x)) / (m_r ψ_n(y) ξ_n'(x) - ψ_n'(y) ξ_n(x))
        num_a2 = m_r * psi_y * psi_x_d - psi_y_d * ComplexF64(psi_x)
        den_a2 = m_r * psi_y * xi_x_d - psi_y_d * xi_x
        a[n + 1] = num_a2 / den_a2

        # b_n = (ψ_n(y) ψ_n'(x) - m_r ψ_n'(y) ψ_n(x)) / (ψ_n(y) ξ_n'(x) - m_r ψ_n'(y) ξ_n(x))
        num_b = psi_y * psi_x_d - m_r * psi_y_d * ComplexF64(psi_x)
        den_b = psi_y * xi_x_d - m_r * psi_y_d * xi_x
        b[n + 1] = num_b / den_b
    end

    return (; a, b, nstop, x, m_r)
end

"""
    mie_forward_amplitude(; wl_0, m_m, r_p, m_p) -> ComplexF64

BH S(0) = (1/2) Σ (2n+1)(a_n + b_n) using outgoing-wave a_n, b_n.
"""
function mie_forward_amplitude(; wl_0, m_m, r_p, m_p)
    sc = mie_scattering_coefficients_outgoing(; wl_0=wl_0, m_m=m_m, r_p=r_p, m_p=m_p)
    S0 = zero(ComplexF64)
    for n in 1:sc.nstop
        S0 += (2n + 1) * (sc.a[n + 1] + sc.b[n + 1])
    end
    return S0 / 2
end

"""
    mie_backward_amplitude(; wl_0, m_m, r_p, m_p) -> ComplexF64

BH S₂(180) = (1/2) Σ (2n+1)(-1)^n (a_n - b_n) using outgoing-wave a_n, b_n.
"""
function mie_backward_amplitude(; wl_0, m_m, r_p, m_p)
    sc = mie_scattering_coefficients_outgoing(; wl_0=wl_0, m_m=m_m, r_p=r_p, m_p=m_p)
    S180 = zero(ComplexF64)
    for n in 1:sc.nstop
        S180 += (2n + 1) * (-1)^n * (sc.a[n + 1] - sc.b[n + 1])
    end
    return S180 / 2
end

# ============================================================================
# Relationship between VIEM F and BH S:
#
# VIEM defines: E^sca ~ [exp(-jk₀r)/(4πr)] F(r̂)
# BH defines:   E^sca ~ [exp(+ikr)/(-ikr)] S(θ) (for each polarization)
#
# Convention conversion (exp(+jωt) → exp(-iωt)):
#   F_VIEM is in engineering convention, S_BH is in physics convention.
#   The cross section relation gives:
#     C_ext = -Im(S_fw_s) / k₀    (our convention, from compute_scattering)
#     C_ext = (4π/k²) Re(S_BH(0)) (BH convention)
#
# The exact relationship is found by matching the cross section formulas.
# From VIEM: C_ext = -Im(ê·F)/(k₀|E₀|²)
# From BH:   C_ext = (4π/k₀²) Re(S_BH(0))
#
# So: -Im(S_fw_s)/k₀ = (4π/k₀²) Re(S_BH(0))
#     Im(S_fw_s) = -(4π/k₀) Re(S_BH(0))
#
# And matching Re parts via the full amplitude correspondence:
#     S_fw_s = -(4π/k₀) conj(S_BH(0))  ... to be determined empirically.
#
# Rather than derive the full mapping, we compare VIEM cross sections and
# raw amplitudes independently.
# ============================================================================

function run_convergence(; m_p, label, lc_values)
    radius = 1.0
    wl_0 = 10.0
    m_m = 1.0
    eps_bg = m_m^2
    eps_p = ComplexF64(m_p)^2
    k0 = 2π * m_m / wl_0

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

    println("=" ^ 78)
    println("CONVERGENCE STUDY: $label (m = $m_p)")
    println("=" ^ 78)

    # Storage for convergence data
    results = []

    for lc in lc_values
        path = generate_sphere_mesh(radius, lc)
        mesh = read_msh(path)
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        V_mesh = sum(mesh.tet_volumes)
        r_ve = (3V_mesh / (4π))^(1 / 3)

        # Effective h: average edge length ~ lc
        h_eff = lc

        # Mie reference at volume-equivalent radius
        mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        S0_mie = mie_forward_amplitude(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        S180_mie = mie_backward_amplitude(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)

        # Solve VIEM
        dr = BlockVIEM.duffy_reference_rule(7)
        Z = assemble_impedance_matrix(basis; k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                      duffy_rule=dr, symmetrize=true)
        k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
        b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg)
        D = Z \ b

        # Scattering (farfield method)
        scat = compute_scattering(basis, D;
                                  k_hat=k_hat, E0=E0, k0=k0,
                                  eps_p=eps_p, eps_bg=eps_bg,
                                  csca_method=:farfield, n_theta=20)

        # Also get optical theorem result for comparison
        scat_ot = compute_scattering(basis, D;
                                     k_hat=k_hat, E0=E0, k0=k0,
                                     eps_p=eps_p, eps_bg=eps_bg,
                                     csca_method=:optical_theorem)

        push!(results, (; lc, h_eff, N, r_ve, mie, S0_mie, S180_mie,
                          scat, scat_ot))

        @printf("  lc=%.3f  N=%5d  r_ve=%.4f\n", lc, N, r_ve)
    end

    # --- Convention mapping: Mie S_BH(0) → VIEM S_fw_s ---
    # E^sca_VIEM = [exp(-jkr)/(4πr)] F,  S_fw_s = ê_s · F
    # E^sca_BH   = [exp(+ikr)/(-ikr)] S_BH
    # E_VIEM = conj(E_BH) gives:  S_fw_s = -(4πi/k₀) conj(S_BH(0))
    #   Re(S_fw_s) = -(4π/k₀) Im(S_BH(0))
    #   Im(S_fw_s) = -(4π/k₀) Re(S_BH(0))

    for r in results
        S_BH = r.S0_mie
        # Convention mapping (derived from E_eng = conj(E_BH)):
        #   Forward:  S_fw_s = -(4πi/k₀) conj(S_BH(0))
        #   Backward: S_bak  = +(4πi/k₀) conj(S₂(π))
        # The sign difference comes from ê_θ(π) = -ê_θ(0) at the south pole.
        r_data = (; r...,
            S_fw_s_mie = -(4π * im / k0) * conj(S_BH),
            S_bak_mie = +(4π * im / k0) * conj(r.S180_mie))
        results[findfirst(x -> x === r, results)] = r_data
    end

    # --- Print convergence table ---
    println("\n  Cross sections:")
    println("  " * "-" ^ 74)
    @printf("  %5s %5s | %10s %10s %10s | %10s %10s\n",
            "lc", "N", "C_abs err%", "C_sca_ff%", "C_ext_ff%", "C_sca_ot%", "C_ext_ot%")
    println("  " * "-" ^ 74)
    for r in results
        ea = 100 * abs(r.scat.C_abs - r.mie.C_abs) / abs(r.mie.C_abs)
        es_ff = 100 * abs(r.scat.C_sca - r.mie.C_sca) / abs(r.mie.C_sca)
        ee_ff = 100 * abs(r.scat.C_ext - r.mie.C_ext) / abs(r.mie.C_ext)
        es_ot = 100 * abs(r.scat_ot.C_sca - r.mie.C_sca) / abs(r.mie.C_sca)
        ee_ot = 100 * abs(r.scat_ot.C_ext - r.mie.C_ext) / abs(r.mie.C_ext)
        @printf("  %5.3f %5d | %9.2f%% %9.2f%% %9.2f%% | %9.2f%% %9.2f%%\n",
                r.lc, r.N, ea, es_ff, ee_ff, es_ot, ee_ot)
    end

    # --- Forward amplitude absolute error ---
    println("\n  Forward amplitude S_fw_s (absolute error |ΔS|):")
    println("  " * "-" ^ 78)
    @printf("  %5s %5s | %10s %10s %10s | %10s %10s\n",
            "lc", "N", "|ΔRe(Ss)|", "|ΔIm(Ss)|", "|ΔSs|", "|Ss_mie|", "|ΔSs|/|Ss|")
    println("  " * "-" ^ 78)
    for r in results
        Ss_v = r.scat_ot.S_fw_s
        Ss_m = r.S_fw_s_mie
        dRe = abs(real(Ss_v) - real(Ss_m))
        dIm = abs(imag(Ss_v) - imag(Ss_m))
        dS = abs(Ss_v - Ss_m)
        Sm = abs(Ss_m)
        @printf("  %5.3f %5d | %10.4e %10.4e %10.4e | %10.4e %9.2f%%\n",
                r.lc, r.N, dRe, dIm, dS, Sm, 100 * dS / Sm)
    end

    # --- Backward amplitude ---
    println("\n  Backward amplitude S_bak (absolute error |ΔS|):")
    println("  " * "-" ^ 60)
    @printf("  %5s %5s | %10s %10s %10s\n",
            "lc", "N", "|ΔS_bak|", "|S_bak_mie|", "|ΔS|/|S|")
    println("  " * "-" ^ 60)
    for r in results
        Sb_v = r.scat_ot.S_bak
        Sb_m = r.S_bak_mie
        dS = abs(Sb_v - Sb_m)
        Sm = abs(Sb_m)
        @printf("  %5.3f %5d | %10.4e %10.4e %9.2f%%\n",
                r.lc, r.N, dS, Sm, 100 * dS / max(Sm, 1e-30))
    end

    # --- Convergence rate estimation ---
    if length(results) >= 2
        println("\n  Convergence rates (log-log slope of absolute error):")
        println("  " * "-" ^ 60)

        h_vals = [r.h_eff for r in results]
        abs_errs = [abs(r.scat.C_abs - r.mie.C_abs) for r in results]
        sca_ff_errs = [abs(r.scat.C_sca - r.mie.C_sca) for r in results]
        ext_ff_errs = [abs(r.scat.C_ext - r.mie.C_ext) for r in results]
        Ss_abs_errs = [abs(r.scat_ot.S_fw_s - r.S_fw_s_mie) for r in results]
        Re_Ss_errs = [abs(real(r.scat_ot.S_fw_s) - real(r.S_fw_s_mie)) for r in results]
        Im_Ss_errs = [abs(imag(r.scat_ot.S_fw_s) - imag(r.S_fw_s_mie)) for r in results]
        Sb_errs = [abs(r.scat_ot.S_bak - r.S_bak_mie) for r in results]

        for (name, errs) in [("C_abs", abs_errs),
                             ("C_sca (farfield)", sca_ff_errs),
                             ("C_ext (farfield)", ext_ff_errs),
                             ("|ΔS_fw_s|", Ss_abs_errs),
                             ("|ΔRe(S_fw_s)|", Re_Ss_errs),
                             ("|ΔIm(S_fw_s)|", Im_Ss_errs),
                             ("|ΔS_bak|", Sb_errs)]
            valid = all(e -> e > 0, errs)
            if valid && length(errs) >= 2
                rates = Float64[]
                for i in 2:length(errs)
                    r = log(errs[i] / errs[i-1]) / log(h_vals[i] / h_vals[i-1])
                    push!(rates, r)
                end
                @printf("    %-25s  rates: %s  avg: %.2f\n",
                        name,
                        join([@sprintf("%.2f", r) for r in rates], ", "),
                        sum(rates) / length(rates))
            else
                @printf("    %-25s  (cannot compute rate)\n", name)
            end
        end
    end

    println()
    return results
end

# ============================================================================
# Run for two contrast levels
# ============================================================================

# Mesh sizes: coarse → fine
lc_values = [0.7, 0.5, 0.35, 0.25]

println("\n" * "=" ^ 78)
println("   CONVERGENCE STUDY: S(0), S(180), cross sections")
println("=" ^ 78)

results_low = run_convergence(m_p=1.5 + 0.01im, label="Low contrast",
                              lc_values=lc_values)

results_fe = run_convergence(m_p=2.5 + 0.5im, label="Iron oxide",
                             lc_values=lc_values)

println("=" ^ 78)
println("   CONVERGENCE STUDY COMPLETE")
println("=" ^ 78)
