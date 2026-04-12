# Phase B.1: RT0 mesh-refinement convergence study for CAS-v2 observables.
#
# For each lc, solve the Mie sphere problem with RT0 + AIM (or dense for
# small meshes) and compare all CAS-v2 observables against analytical Mie
# theory. Extracts the convergence rate and extrapolates the mesh size
# needed to reach 2% error.
#
# CAS-v2 observables measured:
#   - S_fw_s, S_fw_p (complex forward scattering amplitudes)
#   - S_bak          (complex backward amplitude)
#   - C_ext, C_abs, C_sca
#
# Reference values come from Mie theory at the volume-equivalent radius
# of each mesh (so geometric discretization of the sphere surface is
# separated from the VIEM discretization error).

using LinearAlgebra: norm, dot
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))
include(joinpath(TEST_DIR, "mie_internal_field.jl"))

# ============================================================================
# Mie outgoing-wave scattering coefficients (copied from test/convergence_S0.jl
# to avoid triggering its end-of-file side-effect)
# ============================================================================
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
        chi_x_d = begin
            chi_prev = -Float64(x) * Float64(bessely(n - 1 + 0.5, Float64(x))) * sqrt(π / (2x))
            chi_prev - n / x * chi_x
        end
        xi_x = ComplexF64(psi_x) - im * chi_x
        xi_x_d = ComplexF64(psi_x_d) - im * chi_x_d
        psi_y = ricatti_jn(n, y)
        psi_y_d = ricatti_jn_deriv(n, y)
        num_a2 = m_r * psi_y * psi_x_d - psi_y_d * ComplexF64(psi_x)
        den_a2 = m_r * psi_y * xi_x_d - psi_y_d * xi_x
        a[n + 1] = num_a2 / den_a2
        num_b = psi_y * psi_x_d - m_r * psi_y_d * ComplexF64(psi_x)
        den_b = psi_y * xi_x_d - m_r * psi_y_d * xi_x
        b[n + 1] = num_b / den_b
    end

    return (; a, b, nstop, x, m_r)
end

function mie_forward_amplitude(; wl_0, m_m, r_p, m_p)
    sc = mie_scattering_coefficients_outgoing(; wl_0=wl_0, m_m=m_m, r_p=r_p, m_p=m_p)
    S0 = zero(ComplexF64)
    for n in 1:sc.nstop
        S0 += (2n + 1) * (sc.a[n + 1] + sc.b[n + 1])
    end
    return S0 / 2
end

function mie_backward_amplitude(; wl_0, m_m, r_p, m_p)
    sc = mie_scattering_coefficients_outgoing(; wl_0=wl_0, m_m=m_m, r_p=r_p, m_p=m_p)
    S180 = zero(ComplexF64)
    for n in 1:sc.nstop
        S180 += (2n + 1) * (-1)^n * (sc.a[n + 1] - sc.b[n + 1])
    end
    return S180 / 2
end

function sphere_mesh_b1(lc)
    path = joinpath(tempdir(), "sph_b1_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("b1_$(lc)")
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
    s/c
end

"""
Run one convergence scenario. `solver` is either :dense or :iterative.
Returns a vector of results (one per lc).
"""
function run_scenario(; label::String, m_p::Number, wl_0::Real, lc_values::Vector{Float64},
                      solver_cutoff_N::Int = 1500)
    m_m = 1.0
    eps_bg = m_m^2
    eps_p = ComplexF64(m_p)^2
    k0 = 2π * m_m / wl_0
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

    println("=" ^ 88); flush(stdout)
    @printf("  B.1 scenario: %s   (m=%s, wl=%.3f, k0=%.4f, ka_ve≈%.3f)\n",
            label, string(m_p), wl_0, k0, k0 * 0.97); flush(stdout)
    println("=" ^ 88); flush(stdout)

    @printf("%7s %6s %7s %7s | %10s %10s %7s | %10s %10s %7s | %10s %10s %7s\n",
            "lc", "N", "ka_ve", "t(s)",
            "C_abs_V", "C_abs_M", "err%",
            "C_sca_V", "C_sca_M", "err%",
            "C_ext_V", "C_ext_M", "err%")
    println("-"^128); flush(stdout)

    results = NamedTuple[]
    for lc in lc_values
        path = sphere_mesh_b1(lc)
        mesh = read_msh(path)
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        V_mesh = total_volume(mesh)
        r_ve = (3V_mesh / (4π))^(1/3)
        h_bar = mean_h(mesh)
        ka_ve = k0 * r_ve

        mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        S0_mie = mie_forward_amplitude(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        S180_mie = mie_backward_amplitude(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
        # BH → engineering convention (from convergence_S0.jl)
        S_fw_s_mie = -(4π * im / k0) * conj(S0_mie)
        S_bak_mie = +(4π * im / k0) * conj(S180_mie)

        solver_used = :dense
        t_solve = 0.0
        D_coeffs = nothing
        if N <= solver_cutoff_N
            t_solve = @elapsed res = solve_direct(basis; k0=k0, eps_p=eps_p,
                                                    eps_bg=eps_bg, k_hat=k_hat, E0=E0)
            D_coeffs = res.D_coeffs
        else
            solver_used = :iterative
            pitch = 0.5 * h_bar
            t_solve = @elapsed res = solve_iterative(basis; k0=k0, eps_p=eps_p,
                                                       eps_bg=eps_bg, k_hat=k_hat, E0=E0,
                                                       pitch=pitch, padding=4,
                                                       tol=1e-7, maxiter=400)
            D_coeffs = res.D_coeffs
        end

        scat = compute_scattering(basis, D_coeffs;
                                   k_hat=k_hat, E0=E0, k0=k0,
                                   eps_p=eps_p, eps_bg=eps_bg,
                                   csca_method=:farfield, n_theta=20)

        ea = 100 * abs(scat.C_abs - mie.C_abs) / abs(mie.C_abs)
        es = 100 * abs(scat.C_sca - mie.C_sca) / abs(mie.C_sca)
        ex = 100 * abs(scat.C_ext - mie.C_ext) / abs(mie.C_ext)

        @printf("%7.3f %6d %7.3f %7.1f | %10.3e %10.3e %6.2f%% | %10.3e %10.3e %6.2f%% | %10.3e %10.3e %6.2f%%\n",
                lc, N, ka_ve, t_solve,
                scat.C_abs, mie.C_abs, ea,
                scat.C_sca, mie.C_sca, es,
                scat.C_ext, mie.C_ext, ex)
        flush(stdout)

        push!(results, (; lc=lc, N=N, h=h_bar, r_ve=r_ve, solver=solver_used,
                          t_solve=t_solve,
                          C_abs=scat.C_abs, C_sca=scat.C_sca, C_ext=scat.C_ext,
                          S_fw_s=scat.S_fw_s, S_bak=scat.S_bak,
                          mie_C_abs=mie.C_abs, mie_C_sca=mie.C_sca, mie_C_ext=mie.C_ext,
                          mie_S_fw_s=S_fw_s_mie, mie_S_bak=S_bak_mie))
    end

    # Convergence rates (log-log slope of error vs h)
    println(); flush(stdout)
    println("  Convergence rates (slope of log(err) vs log(h)):"); flush(stdout)
    hs = [r.h for r in results]
    for (name, v_errs) in [
            ("C_abs", [abs(r.C_abs - r.mie_C_abs)/abs(r.mie_C_abs) for r in results]),
            ("C_sca", [abs(r.C_sca - r.mie_C_sca)/abs(r.mie_C_sca) for r in results]),
            ("C_ext", [abs(r.C_ext - r.mie_C_ext)/abs(r.mie_C_ext) for r in results]),
        ]
        rates = Float64[]
        for i in 2:length(v_errs)
            if v_errs[i-1] > 0 && v_errs[i] > 0
                r = log(v_errs[i] / v_errs[i-1]) / log(hs[i] / hs[i-1])
                push!(rates, r)
            end
        end
        rates_str = join([@sprintf("%5.2f", r) for r in rates], ", ")
        mean_r = isempty(rates) ? 0.0 : sum(rates)/length(rates)
        @printf("    %-8s rates: [%s]  avg=%.2f\n", name, rates_str, mean_r)
        flush(stdout)
    end

    # Extrapolation: at what h does the error drop to 2%?
    println(); flush(stdout)
    println("  Extrapolation (from finest two points, assuming O(h^p)):"); flush(stdout)
    if length(results) >= 2
        h1, h2 = results[end-1].h, results[end].h
        for (name, v_errs) in [
                ("C_abs", [abs(r.C_abs - r.mie_C_abs)/abs(r.mie_C_abs) for r in results]),
                ("C_sca", [abs(r.C_sca - r.mie_C_sca)/abs(r.mie_C_sca) for r in results]),
                ("C_ext", [abs(r.C_ext - r.mie_C_ext)/abs(r.mie_C_ext) for r in results]),
            ]
            e1, e2 = v_errs[end-1], v_errs[end]
            if e1 > 0 && e2 > 0
                p = log(e2/e1) / log(h2/h1)
                if abs(p) > 1e-3
                    h_target = h2 * (0.02 / e2)^(1/p)
                    @printf("    %-8s  p=%.2f  current err=%.1f%%  target h≈%.3f (%.1f× finer)\n",
                            name, p, 100*e2, h_target, h2/h_target)
                else
                    @printf("    %-8s  p≈0 (non-converging)  current err=%.1f%%\n",
                            name, 100*e2)
                end
                flush(stdout)
            end
        end
    end
    println(); flush(stdout)
    return results
end

# ============================================================================
# Scenarios
# ============================================================================
flush(stdout)
println("\n\n"); flush(stdout)

# Scenario 1: low contrast (dielectric), ka ≈ 0.628
results_low = run_scenario(
    label = "low contrast m=1.5+0.01i, wl=10",
    m_p = 1.5 + 0.01im,
    wl_0 = 10.0,
    lc_values = [0.7, 0.5, 0.35, 0.25, 0.18, 0.13],
    solver_cutoff_N = 1500,
)

# Scenario 2: high contrast like buff-em test
results_hi = run_scenario(
    label = "buff-em like eps=10+1i, ka=0.316",
    m_p = sqrt(10.0 + 1.0im),
    wl_0 = 2π / 0.316,
    lc_values = [0.7, 0.5, 0.35, 0.25, 0.18],
    solver_cutoff_N = 1500,
)
