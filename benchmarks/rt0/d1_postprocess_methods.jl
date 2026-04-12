# Phase D.1: compare convergence of CAS-v2 observables computed via
# different post-processing methods, on the same dense VIEM solution.
#
# Methods for cross sections:
#   M-A:  C_abs = D†MD-based (current default)
#         C_sca = far-field integral
#         C_ext = C_abs + C_sca
#   M-B:  C_ext = optical theorem -Im(<E0*, F_fw>) / (k0 |E0|²)
#         C_abs = D†MD-based (same as M-A)
#         C_sca = C_ext - C_abs
#   M-C:  C_ext = optical theorem (same as M-B)
#         C_sca = far-field integral (same as M-A)
#         C_abs = C_ext - C_sca   (no D†MD ever used!)
#
# Methods M-A and M-B share C_abs but differ in C_ext, C_sca.
# Method M-C avoids D†MD entirely — if it converges faster, the
# postprocessing is the bottleneck (not the discretization).
#
# Also reports complex S_fw_s, S_bak amplitudes against Mie reference.

using LinearAlgebra: norm, dot
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))
include(joinpath(TEST_DIR, "mie_internal_field.jl"))

using SpecialFunctions: bessely

function mie_scattering_coefficients_outgoing(; wl_0::Real, m_m::Real, r_p::Real, m_p::Number)
    k_bg = 2π * m_m / wl_0
    x = k_bg * r_p
    m_r = ComplexF64(m_p) / m_m
    y = m_r * x
    nstop = floor(Int, abs(x) + 4 * abs(x)^(1 / 3) + 2)
    nstop = max(nstop, 3)
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
        chi_x = -Float64(x) * Float64(bessely(n + 0.5, Float64(x))) * sqrt(π / (2x))
        chi_prev = -Float64(x) * Float64(bessely(n - 1 + 0.5, Float64(x))) * sqrt(π / (2x))
        chi_x_d = chi_prev - n / x * chi_x
        xi_x = ComplexF64(psi_x) - im * chi_x
        xi_x_d = ComplexF64(psi_x_d) - im * chi_x_d
        psi_y = ricatti_jn(n, y)
        psi_y_d = ricatti_jn_deriv(n, y)
        a[n + 1] = (m_r * psi_y * psi_x_d - psi_y_d * ComplexF64(psi_x)) /
                   (m_r * psi_y * xi_x_d - psi_y_d * xi_x)
        b[n + 1] = (psi_y * psi_x_d - m_r * psi_y_d * ComplexF64(psi_x)) /
                   (psi_y * xi_x_d - m_r * psi_y_d * xi_x)
    end
    return (; a, b, nstop, x, m_r)
end

function mie_S0(; wl_0, m_m, r_p, m_p)
    sc = mie_scattering_coefficients_outgoing(; wl_0, m_m, r_p, m_p)
    S0 = zero(ComplexF64)
    for n in 1:sc.nstop
        S0 += (2n + 1) * (sc.a[n + 1] + sc.b[n + 1])
    end
    return S0 / 2
end

function mie_S180(; wl_0, m_m, r_p, m_p)
    sc = mie_scattering_coefficients_outgoing(; wl_0, m_m, r_p, m_p)
    S180 = zero(ComplexF64)
    for n in 1:sc.nstop
        S180 += (2n + 1) * (-1)^n * (sc.a[n + 1] - sc.b[n + 1])
    end
    return S180 / 2
end

function sphere_mesh_d1(lc)
    path = joinpath(tempdir(), "sph_d1_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("d1_$(lc)")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path)
    gmsh.finalize()
    return path
end

function mean_h(mesh)
    s = 0.0; c = 0
    for tet in mesh.tets, (a,b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
    end
    return s/c
end

function run_d1(; m_p, wl_0, lc_values, label::String)
    m_m = 1.0
    eps_bg = m_m^2
    eps_p = ComplexF64(m_p)^2
    k0 = 2π * m_m / wl_0
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
    e_s = Vec3(1, 0, 0)

    flush(stdout)
    println("=" ^ 100); flush(stdout)
    @printf("  D.1: %s   m=%s   wl=%.3f   k0=%.4f\n", label, string(m_p), wl_0, k0)
    println("=" ^ 100); flush(stdout)

    results = NamedTuple[]
    for lc in lc_values
        mesh = read_msh(sphere_mesh_d1(lc))
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        V_mesh = total_volume(mesh)
        r_ve = (3V_mesh / (4π))^(1/3)
        h_bar = mean_h(mesh)

        # Mie reference
        mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        S0_BH = mie_S0(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        S180_BH = mie_S180(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        # BH → engineering convention
        S_fw_s_mie = -(4π * im / k0) * conj(S0_BH)
        S_bak_mie = +(4π * im / k0) * conj(S180_BH)

        # Solve
        t_solve = @elapsed res = solve_direct(basis; k0=k0, eps_p=eps_p,
                                                eps_bg=eps_bg, k_hat=k_hat, E0=E0)
        D = res.D_coeffs

        # Method A: C_sca far-field, C_abs DhMD, C_ext = sum
        scA = compute_scattering(basis, D; k_hat=k_hat, E0=E0, k0=k0,
                                  eps_p=eps_p, eps_bg=eps_bg,
                                  csca_method=:farfield, n_theta=20)
        # Method B: same as A's C_abs and S_fw, but C_ext = optical theorem
        scB = compute_scattering(basis, D; k_hat=k_hat, E0=E0, k0=k0,
                                  eps_p=eps_p, eps_bg=eps_bg,
                                  csca_method=:optical_theorem, n_theta=20)
        # Method C: C_ext from B, C_sca from A, C_abs = B.C_ext - A.C_sca
        C_ext_C = scB.C_ext
        C_sca_C = scA.C_sca
        C_abs_C = C_ext_C - C_sca_C

        push!(results, (; lc, N, h=h_bar, r_ve, t_solve,
                          mie_C_abs=mie.C_abs, mie_C_sca=mie.C_sca, mie_C_ext=mie.C_ext,
                          mie_S_fw_s=S_fw_s_mie, mie_S_bak=S_bak_mie,
                          # Method A
                          A_C_abs=scA.C_abs, A_C_sca=scA.C_sca, A_C_ext=scA.C_ext,
                          # Method B
                          B_C_abs=scB.C_abs, B_C_sca=scB.C_sca, B_C_ext=scB.C_ext,
                          # Method C
                          C_C_abs=C_abs_C, C_C_sca=C_sca_C, C_C_ext=C_ext_C,
                          # S amplitudes (same in A and B)
                          S_fw_s=scA.S_fw_s, S_bak=scA.S_bak))
    end

    # Print errors per method
    function err_table(label::String, vals_field::Symbol, mie_field::Symbol)
        println("\n  --- $label ---"); flush(stdout)
        @printf("  %6s %6s | %12s %8s | %12s %8s | %12s %8s\n",
                "lc", "N", "M-A", "err%", "M-B", "err%", "M-C", "err%")
        println("  " * "-"^88); flush(stdout)
        for r in results
            mv = getfield(r, mie_field)
            vA = getfield(r, Symbol("A_" * string(vals_field)))
            vB = getfield(r, Symbol("B_" * string(vals_field)))
            vC = getfield(r, Symbol("C_" * string(vals_field)))
            eA = 100*abs(vA - mv)/abs(mv)
            eB = 100*abs(vB - mv)/abs(mv)
            eC = 100*abs(vC - mv)/abs(mv)
            @printf("  %6.3f %6d | %12.4e %6.2f%% | %12.4e %6.2f%% | %12.4e %6.2f%%\n",
                    r.lc, r.N, vA, eA, vB, eB, vC, eC)
            flush(stdout)
        end
    end

    err_table("C_abs", :C_abs, :mie_C_abs)
    err_table("C_sca", :C_sca, :mie_C_sca)
    err_table("C_ext", :C_ext, :mie_C_ext)

    # Convergence rates per method
    println("\n  --- Convergence rates (avg log-log slope) ---"); flush(stdout)
    @printf("  %12s | %8s %8s %8s\n", "observable", "M-A", "M-B", "M-C")
    println("  " * "-"^46); flush(stdout)

    function avg_rate(method::String, field::Symbol, mie_field::Symbol)
        hs = [r.h for r in results]
        errs = [abs(getfield(r, Symbol(method * "_" * string(field))) - getfield(r, mie_field)) /
                abs(getfield(r, mie_field)) for r in results]
        rates = Float64[]
        for i in 2:length(errs)
            (errs[i-1] > 0 && errs[i] > 0) || continue
            push!(rates, log(errs[i] / errs[i-1]) / log(hs[i] / hs[i-1]))
        end
        return isempty(rates) ? 0.0 : sum(rates) / length(rates)
    end

    for (name, field, mie_field) in [
            ("C_abs", :C_abs, :mie_C_abs),
            ("C_sca", :C_sca, :mie_C_sca),
            ("C_ext", :C_ext, :mie_C_ext),
        ]
        rA = avg_rate("A", field, mie_field)
        rB = avg_rate("B", field, mie_field)
        rC = avg_rate("C", field, mie_field)
        @printf("  %12s | %8.2f %8.2f %8.2f\n", name, rA, rB, rC)
        flush(stdout)
    end

    # S_fw_s and S_bak amplitudes
    println("\n  --- Forward scattering amplitude S_fw_s ---"); flush(stdout)
    @printf("  %6s %6s | %14s %14s | %14s %14s | %10s\n",
            "lc", "N", "Re(VIEM)", "Re(Mie)", "Im(VIEM)", "Im(Mie)", "|Δ|/|S|")
    println("  " * "-"^96); flush(stdout)
    for r in results
        Sv = r.S_fw_s; Sm = r.mie_S_fw_s
        rel = abs(Sv - Sm) / abs(Sm)
        @printf("  %6.3f %6d | %+14.4e %+14.4e | %+14.4e %+14.4e | %9.2f%%\n",
                r.lc, r.N, real(Sv), real(Sm), imag(Sv), imag(Sm), 100*rel)
        flush(stdout)
    end

    println("\n  --- Backward scattering amplitude S_bak ---"); flush(stdout)
    @printf("  %6s %6s | %14s %14s | %14s %14s | %10s\n",
            "lc", "N", "Re(VIEM)", "Re(Mie)", "Im(VIEM)", "Im(Mie)", "|Δ|/|S|")
    println("  " * "-"^96); flush(stdout)
    for r in results
        Sv = r.S_bak; Sm = r.mie_S_bak
        rel = abs(Sv - Sm) / abs(Sm)
        @printf("  %6.3f %6d | %+14.4e %+14.4e | %+14.4e %+14.4e | %9.2f%%\n",
                r.lc, r.N, real(Sv), real(Sm), imag(Sv), imag(Sm), 100*rel)
        flush(stdout)
    end

    println(); flush(stdout)
    return results
end

# Run for two contrast levels using dense solver only
results_low = run_d1(
    label = "low contrast m=1.5+0.01i",
    m_p = 1.5 + 0.01im,
    wl_0 = 10.0,
    lc_values = [0.7, 0.5, 0.35, 0.25],
)

results_hi = run_d1(
    label = "high contrast eps=10+1i",
    m_p = sqrt(10.0 + 1.0im),
    wl_0 = 2π / 0.316,
    lc_values = [0.7, 0.5, 0.35, 0.25],
)
