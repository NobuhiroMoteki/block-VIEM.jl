# V5: optical-theorem consistency test for Im(Z) / radiation reaction.
#
# Investigation #1 from memory/v2_v3_dda_comparison.md.
#
# For a LOSSLESS particle (Im(eps_p) = 0), the continuum optical theorem
# forces C_ext = C_sca exactly. Any gap between
#
#   C_ext  := -Im(E0* . F_fw) / (k0 |E0|^2)           (optical theorem)
#   C_sca  := (1 / |E0|^2) (1/k0^2) int |F|^2 dOmega  (far-field power)
#
# in the discrete solution is a direct measurement of energy-non-conservation
# coming from the discretization of Im(Z) (the radiation-reaction piece of
# the integral operator). In DDA, CR2009 bakes in the analytic radiation
# reaction `(1 - ika) exp(ika) - 1`; in VIEM this must be produced by the
# Duffy-quadrature of Im(helmholtz_green) on the self-pair.
#
# What we print per (lc):
#   C_ext_OT   : optical theorem value
#   C_sca_FF   : far-field power integral
#   C_abs_vol  : volumetric Joule loss (exactly 0 for lossless)
#   gap        : (C_ext_OT - C_sca_FF) / C_sca_FF   -- should be 0 in continuum
#   mie_err_OT : (C_ext_OT - C_ext_Mie) / C_ext_Mie
#   mie_err_FF : (C_sca_FF - C_sca_Mie) / C_sca_Mie
#
# Interpretation:
#   - If |gap| >> mie_err_FF:   Im(Z) discretization is the dominant error
#     (radiation-reaction leak). This is the outcome predicted by the bug
#     hypothesis -- points firmly at the self-term Duffy quadrature.
#   - If |gap| ~ mie_err_FF:    both observables have the same discretization
#     error; the issue is mass-term / f-dot-f' dominated, not Im(Z).
#   - If |gap| << mie_err_FF:   Im(Z) is accurate; the error lives in the
#     real part of the operator (static Coulomb / nullspace handling).

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v5_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v5_$(lc)")
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
    for tet in mesh.tets, (a, b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
    end
    s / c
end

# Pure lossless dielectric, same real(m) as the v3/v4 low-contrast study
const m_p    = 1.5 + 0.0im
const wl_0   = 10.0
const m_m    = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0     = 2π * m_m / wl_0
const k_hat  = Vec3(0, 0, 1)
const E0     = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

println("=" ^ 96); flush(stdout)
println("  V5: optical theorem consistency test (LOSSLESS m=1.5+0i, wl=10, ka_ve~0.628)")
println("  Gap between C_ext(optical theorem) and C_sca(far field) measures Im(Z) discretization error.")
println("=" ^ 96); flush(stdout)

@printf("%6s %6s %8s %7s | %10s %10s %8s | %10s %10s | %10s %10s\n",
        "lc", "N_swg", "h_bar", "t(s)",
        "C_ext_OT", "C_sca_FF", "gap%",
        "C_ext_Mie", "err_OT%", "C_sca_Mie", "err_FF%")
println("-"^132); flush(stdout)

for lc in (0.35, 0.25, 0.18, 0.15)
    path = sphere_mesh(lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    V_mesh = total_volume(mesh)
    r_ve = (3V_mesh / (4π))^(1/3)
    h_bar = mean_h(mesh)

    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)

    # Solve: use dense for small, AIM for larger.
    if N <= 1500
        t = @elapsed res = solve_direct(basis;
                                         k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                         k_hat=k_hat, E0=E0)
    else
        pitch = 0.5 * h_bar
        t = @elapsed res = solve_iterative(basis;
                                            k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                            k_hat=k_hat, E0=E0,
                                            pitch=pitch, padding=4,
                                            tol=1e-7, maxiter=600)
    end

    # Far-field method: computes C_sca from far-field, then C_ext = C_abs + C_sca
    # Because C_abs_volumetric is 0 (lossless), this gives C_ext_FF = C_sca_FF.
    scat_ff = compute_scattering(basis, res.D_coeffs;
                                  k_hat=k_hat, E0=E0, k0=k0,
                                  eps_p=eps_p, eps_bg=eps_bg,
                                  csca_method=:farfield, n_theta=30)
    # Optical-theorem method: computes C_ext from Im(E0* . F_fw), then C_sca = C_ext - C_abs.
    scat_ot = compute_scattering(basis, res.D_coeffs;
                                  k_hat=k_hat, E0=E0, k0=k0,
                                  eps_p=eps_p, eps_bg=eps_bg,
                                  csca_method=:optical_theorem, n_theta=30)

    C_sca_ff = scat_ff.C_sca
    C_ext_ot = scat_ot.C_ext
    gap = (C_ext_ot - C_sca_ff) / C_sca_ff

    err_ot = (C_ext_ot - mie.C_ext) / mie.C_ext
    err_ff = (C_sca_ff - mie.C_sca) / mie.C_sca

    @printf("%6.3f %6d %8.3f %7.1f | %10.3e %10.3e %+7.2f%% | %10.3e %+7.2f%% | %10.3e %+7.2f%%\n",
            lc, N, h_bar, t,
            C_ext_ot, C_sca_ff, 100 * gap,
            mie.C_ext, 100 * err_ot,
            mie.C_sca, 100 * err_ff)
    flush(stdout)
end

println()
println("  Interpretation guide:")
println("    |gap| >> |err_FF|  -> Im(Z) radiation reaction leak (investigation #1 confirmed)")
println("    |gap| ~= |err_FF| -> real(Z) dominant, Im(Z) fine (move to investigation #2 or #3)")
flush(stdout)
