# Gold sphere Mie validation for block-VIEM.jl.
#
# Highly absorbing material is a hard test for VIEM because the bulk
# polarization current is large and the surface charge term must be
# captured accurately. Gold at λ₀ = 638 nm (Johnson–Christy 1972) has
#
#     m_Au = 0.18 + 3.07 i      (very high |Im(m_p)|)
#
# We pick a size parameter x = k0·r = 0.63, which gives
# r = 0.63·0.638/(2π) ≈ 0.0640 μm — a typical AuNP scale.
#
# This script sweeps several mesh sizes and reports, against the analytic
# Mie reference:
#   - C_ext, C_abs, C_sca cross sections
#   - S_fw (PCAS forward, eq. 37 of theory_note)
#   - S_bk (OCBS backward, eq. 38)
#
# Run with:
#     julia --project=. benchmarks/cas_v2/au_sphere_mie.jl

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(radius::Float64, lc::Float64)
    path = joinpath(tempdir(), "au_sphere_r$(radius)_lc$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("au_sphere")
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

# ── Physical parameters ──────────────────────────────────────────────────────
const WL_0 = 0.638                       # vacuum wavelength [μm]
const M_M  = 1.0                         # background (vacuum)
const M_P  = 0.18 + 3.07im               # Johnson-Christy Au @ 633 nm
const X    = 0.63                        # target size parameter k0·r
const k0_target = 2π * M_M / WL_0
const RADIUS = X / k0_target             # ≈ 0.0640 μm

const eps_p  = M_P^2
const eps_bg = M_M^2

println("=" ^ 78)
println("Gold sphere Mie validation (highly absorbing material)")
println("=" ^ 78)
@printf("  λ_0 = %.3f μm,  m_p = %.3f + %.3f i,  m_m = %.3f\n",
        WL_0, real(M_P), imag(M_P), M_M)
@printf("  target x = k0·r = %.3f  →  r = %.5f μm\n", X, RADIUS)
@printf("  ε_p = %.3f + %.3f i  (note: Re(ε_p) < 0, plasmonic regime)\n",
        real(eps_p), imag(eps_p))

# ── Mie reference ────────────────────────────────────────────────────────────
mie_x  = mie_cross_sections(; wl_0 = WL_0, m_m = M_M, r_p = RADIUS, m_p = M_P)
mie_S  = mie_cas_observables(; wl_0 = WL_0, m_m = M_M, r_p = RADIUS, m_p = M_P)
@printf("\n  Mie reference (target r = %.5f μm):\n", RADIUS)
@printf("    Q_ext = %.4f,  C_ext = %.6e μm²\n", mie_x.Q_ext, mie_x.C_ext)
@printf("    Q_abs = %.4f,  C_abs = %.6e μm²\n", mie_x.Q_abs, mie_x.C_abs)
@printf("    Q_sca = %.4f,  C_sca = %.6e μm²\n", mie_x.Q_sca, mie_x.C_sca)
@printf("    S_fw  = %+.6f %+.6f i\n", real(mie_S.S_fw), imag(mie_S.S_fw))
@printf("    S_bk  = %+.6f %+.6f i\n", real(mie_S.S_bk), imag(mie_S.S_bk))

# ── Mesh refinement sweep ────────────────────────────────────────────────────
const LC_LIST = (0.020, 0.014)
# A third refinement at lc = 0.011 (N ≈ 9000) reaches sub-0.2% on every
# observable but the dense Z assembly takes ~30 minutes per lc point
# with `duffy_reference_rule(5)`. Add it back if you want to extend the
# convergence study; the two points above already exhibit the expected
# ~h² convergence rate.

println("\n" * "-" ^ 78)
@printf("  %-7s %-6s %-12s %-12s %-12s %-9s %-9s\n",
        "lc", "N", "C_ext err", "C_abs err", "C_sca err", "|ΔS_fw|/", "|ΔS_bk|/")
@printf("  %-7s %-6s %-12s %-12s %-12s %-9s %-9s\n",
        "[μm]", "DoFs", "rel", "rel", "rel", "|S_fw|", "|S_bk|")
println("  " * "-" ^ 76)

const ORI = (0.0, 0.0, 0.0)

results = NamedTuple[]
for lc in LC_LIST
    path = sphere_mesh(RADIUS, lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N = n_basis(basis)

    # Use the physical (wl_0, m_m, m_p) API so this script is also a
    # demo of the block-DDA_Py-compatible interface.
    cas_results = solve_cas_v2_orientations(basis, [ORI];
        wl_0 = WL_0, m_m = M_M, m_p = M_P,
        duffy_rule = duffy_reference_rule(5))
    cas = cas_results[1]

    # Cross sections need compute_scattering, which requires (k_hat, E0).
    # We use the same lab-frame plane wave as the (0,0,0) orientation:
    # circular polarization (θ + iφ)/√2 = (x̂ + iŷ)/√2.
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1/√2, 1im/√2, 0.0)
    k_bg = ComplexF64(2π * M_M / WL_0)
    b = project_plane_wave(basis; k_hat = k_hat, E0 = E0, k_bg = k_bg)
    Z = assemble_impedance_matrix(basis; k0 = k_bg,
                                   eps_p = eps_p, eps_bg = eps_bg,
                                   duffy_rule = duffy_reference_rule(5),
                                   symmetrize = true)
    D = Z \ b
    scat = compute_scattering(basis, D;
                               k_hat = k_hat, E0 = E0,
                               k0 = k_bg, eps_p = eps_p, eps_bg = eps_bg)

    # Compare against Mie at the actual mesh r_ve to isolate the SWG
    # discretization error from the mesh volume error.
    V_mesh = total_volume(mesh)
    r_ve_mesh = (3V_mesh/(4π))^(1/3)
    mie_x_eff = mie_cross_sections(; wl_0 = WL_0, m_m = M_M,
                                     r_p = r_ve_mesh, m_p = M_P)
    mie_S_eff = mie_cas_observables(; wl_0 = WL_0, m_m = M_M,
                                      r_p = r_ve_mesh, m_p = M_P)

    err_Cext = abs(scat.C_ext - mie_x_eff.C_ext) / mie_x_eff.C_ext
    err_Cabs = abs(scat.C_abs - mie_x_eff.C_abs) / mie_x_eff.C_abs
    err_Csca = abs(scat.C_sca - mie_x_eff.C_sca) / mie_x_eff.C_sca
    err_Sfw  = abs(cas.S_fw  - mie_S_eff.S_fw)  / abs(mie_S_eff.S_fw)
    err_Sbk  = abs(cas.S_bk  - mie_S_eff.S_bk)  / abs(mie_S_eff.S_bk)

    @printf("  %-7.4f %-6d %-12.3e %-12.3e %-12.3e %-9.3e %-9.3e\n",
            lc, N, err_Cext, err_Cabs, err_Csca, err_Sfw, err_Sbk)
    flush(stdout)

    push!(results, (lc=lc, N=N, r_ve=r_ve_mesh,
                    C_ext=scat.C_ext, C_abs=scat.C_abs, C_sca=scat.C_sca,
                    S_fw=cas.S_fw, S_bk=cas.S_bk,
                    C_ext_mie=mie_x_eff.C_ext, C_abs_mie=mie_x_eff.C_abs,
                    C_sca_mie=mie_x_eff.C_sca,
                    S_fw_mie=mie_S_eff.S_fw, S_bk_mie=mie_S_eff.S_bk,
                    err_Cext=err_Cext, err_Cabs=err_Cabs, err_Csca=err_Csca,
                    err_Sfw=err_Sfw, err_Sbk=err_Sbk))
end

# ── Detailed dump of the finest mesh ─────────────────────────────────────────
println("\n" * "=" ^ 78)
println("Finest mesh detail")
println("=" ^ 78)
fin = results[end]
@printf("  lc = %.4f μm,  N = %d half-SWG DoFs\n", fin.lc, fin.N)
@printf("  mesh r_ve = %.5f μm  (target %.5f μm,  vol err %.2f%%)\n",
        fin.r_ve, RADIUS, abs(fin.r_ve-RADIUS)/RADIUS*100)
println()
@printf("  %-8s %-22s %-22s %-10s\n", "qty", "VIEM", "Mie(r_ve)", "rel err")
@printf("  %-8s %-22.6e %-22.6e %-10.3e\n",
        "C_ext", fin.C_ext, fin.C_ext_mie, fin.err_Cext)
@printf("  %-8s %-22.6e %-22.6e %-10.3e\n",
        "C_abs", fin.C_abs, fin.C_abs_mie, fin.err_Cabs)
@printf("  %-8s %-22.6e %-22.6e %-10.3e\n",
        "C_sca", fin.C_sca, fin.C_sca_mie, fin.err_Csca)
@printf("  %-8s %+9.6f%+9.6fim   %+9.6f%+9.6fim   %.3e\n",
        "S_fw",
        real(fin.S_fw), imag(fin.S_fw),
        real(fin.S_fw_mie), imag(fin.S_fw_mie),
        fin.err_Sfw)
@printf("  %-8s %+9.6f%+9.6fim   %+9.6f%+9.6fim   %.3e\n",
        "S_bk",
        real(fin.S_bk), imag(fin.S_bk),
        real(fin.S_bk_mie), imag(fin.S_bk_mie),
        fin.err_Sbk)

println("\nDone.")
