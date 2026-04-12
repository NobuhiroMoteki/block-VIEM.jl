# Phase D.2: test whether increasing the far-field quadrature order
# improves the accuracy of the forward scattering amplitude (in
# particular its imaginary part, which is tiny vs the real part).
#
# Hypothesis: Im(F_x) = O(h^α) errors from the default 5-point rule
# contaminate a value that's only ~5% of Re(F_x), making it appear
# noisy or even wrong-sign.
#
# Fix: use higher-order rule (degree 5+ via TET_QUAD_64PT) for the
# far-field integration. This is O(N × rule.n) so cheap even for
# fine meshes.

using LinearAlgebra: norm, dot
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, TET_QUAD_5PT, TET_QUAD_64PT, TET_QUAD_125PT,
                 far_field_amplitude, tet_collapsed_rule
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))
include(joinpath(TEST_DIR, "mie_internal_field.jl"))
using SpecialFunctions: bessely

# Re-declare Mie outgoing helpers (same as D.1)
function mie_so(; wl_0, m_m, r_p, m_p)
    k_bg = 2π * m_m / wl_0
    x = k_bg * r_p
    m_r = ComplexF64(m_p) / m_m
    y = m_r * x
    nstop = max(floor(Int, abs(x) + 4*abs(x)^(1/3) + 2), 3)
    nmx = floor(Int, max(nstop, abs(y)) + 15)
    DD = zeros(ComplexF64, nmx + 1)
    for n in nmx:-1:1
        DD[n] = n/y - 1 / (DD[n+1] + n/y)
    end
    S0 = zero(ComplexF64); S180 = zero(ComplexF64)
    for n in 1:nstop
        psi_x = ricatti_jn(n, x); psi_x_d = ricatti_jn_deriv(n, x)
        chi_x = -Float64(x) * Float64(bessely(n + 0.5, Float64(x))) * sqrt(π/(2x))
        chi_prev = -Float64(x) * Float64(bessely(n - 1 + 0.5, Float64(x))) * sqrt(π/(2x))
        chi_x_d = chi_prev - n/x * chi_x
        xi_x = ComplexF64(psi_x) - im*chi_x
        xi_x_d = ComplexF64(psi_x_d) - im*chi_x_d
        psi_y = ricatti_jn(n, y); psi_y_d = ricatti_jn_deriv(n, y)
        an = (m_r*psi_y*psi_x_d - psi_y_d*ComplexF64(psi_x)) /
             (m_r*psi_y*xi_x_d - psi_y_d*xi_x)
        bn = (psi_y*psi_x_d - m_r*psi_y_d*ComplexF64(psi_x)) /
             (psi_y*xi_x_d - m_r*psi_y_d*xi_x)
        S0 += (2n+1)*(an+bn)
        S180 += (2n+1)*(-1)^n*(an-bn)
    end
    return S0/2, S180/2
end

function sphere_mesh_d2(lc)
    path = joinpath(tempdir(), "sph_d2_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("d2_$(lc)")
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

function run_d2(; m_p, wl_0, lc_values, label::String)
    m_m = 1.0
    eps_bg = 1.0
    eps_p = ComplexF64(m_p)^2
    k0 = 2π * m_m / wl_0
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

    flush(stdout)
    println("=" ^ 100); flush(stdout)
    @printf("  D.2: %s   k0=%.4f\n", label, k0)
    println("=" ^ 100); flush(stdout)

    # Rules to test
    rules = [
        ("5pt  (deg 3)", TET_QUAD_5PT),
        ("64pt (deg 5)", TET_QUAD_64PT),
        ("125pt (deg 7)", TET_QUAD_125PT),
    ]

    for lc in lc_values
        mesh = read_msh(sphere_mesh_d2(lc))
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        V_mesh = total_volume(mesh)
        r_ve = (3V_mesh/(4π))^(1/3)
        h_bar = mean_h(mesh)

        # Mie reference at r_ve
        S0_BH, S180_BH = mie_so(; wl_0, m_m, r_p=r_ve, m_p)
        S_fw_mie = -(4π * im / k0) * conj(S0_BH)
        S_bak_mie = +(4π * im / k0) * conj(S180_BH)

        # Solve (dense)
        res = solve_direct(basis; k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                            k_hat=k_hat, E0=E0)
        D = res.D_coeffs

        println("\n  lc=$lc  N=$N  h̄=$(round(h_bar,digits=4))"); flush(stdout)
        @printf("  %-14s | %+14s %+14s | %+14s %+14s | %10s\n",
                "rule", "Re(F_fw_x)", "Im(F_fw_x)", "Re(F_bak_x)", "Im(F_bak_x)", "|ΔF_fw|%")
        println("  " * "-"^94); flush(stdout)

        # Ground-truth row first
        @printf("  %-14s | %+14.6e %+14.6e | %+14.6e %+14.6e | %9s\n",
                "MIE exact",
                real(S_fw_mie), imag(S_fw_mie),
                real(S_bak_mie), imag(S_bak_mie), "—")

        for (name, rule) in rules
            F_fw = far_field_amplitude(basis, D; k_hat_sca=k_hat, k0=k0,
                                        eps_p=eps_p, eps_bg=eps_bg, rule=rule)
            F_bak = far_field_amplitude(basis, D; k_hat_sca=-k_hat, k0=k0,
                                         eps_p=eps_p, eps_bg=eps_bg, rule=rule)
            S_fw_v = F_fw[1]   # s-pol = x component
            S_bak_v = F_bak[1]
            relF = 100 * abs(S_fw_v - S_fw_mie) / abs(S_fw_mie)
            @printf("  %-14s | %+14.6e %+14.6e | %+14.6e %+14.6e | %9.2f%%\n",
                    name, real(S_fw_v), imag(S_fw_v),
                    real(S_bak_v), imag(S_bak_v), relF)
            flush(stdout)
        end
    end
    println(); flush(stdout)
end

run_d2(
    label = "low contrast m=1.5+0.01i, wl=10",
    m_p = 1.5 + 0.01im,
    wl_0 = 10.0,
    lc_values = [0.5, 0.35, 0.25],
)

run_d2(
    label = "high contrast eps=10+1i, ka=0.316",
    m_p = sqrt(10.0 + 1.0im),
    wl_0 = 2π / 0.316,
    lc_values = [0.5, 0.35, 0.25],
)
