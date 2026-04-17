# Gold sphere Mie validation for block-VIEM.jl.
#
# Highly absorbing material is a hard test for VIEM because the bulk
# polarization current is large and the surface charge term must be
# captured accurately. Gold at λ₀ = 638 nm (Johnson & Christy 1972,
# via refractiveindex.info) has
#
#     m_Au = 0.17525 + 3.4830 i      (very high |Im(m_p)|)
#
# We pick a size parameter x = k0·r = 0.63, which gives
# r = 0.63·0.638/(2π) ≈ 0.0640 μm — a typical AuNP scale.
#
# This script sweeps several mesh sizes and runs BOTH dense (LU) and
# AIM-BiCGSTAB solvers at each mesh, reporting accuracy and timing
# against the analytic Mie reference:
#   - C_ext, C_abs, C_sca cross sections
#   - S_fw (PCAS forward)
#   - S_bk (OCBS backward)
#
# Run with:
#     julia --project=. -t auto benchmarks/cas_v2/au_sphere_mie.jl

using LinearAlgebra: norm, lu
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
const M_P  = 0.17525 + 3.4830im          # Johnson & Christy 1972, Au @ 638 nm
const X    = 0.63                        # target size parameter k0·r
const k0_target = 2π * M_M / WL_0
const RADIUS = X / k0_target             # ≈ 0.0640 μm

const eps_p  = M_P^2
const eps_bg = M_M^2
const k_bg   = ComplexF64(2π * M_M / WL_0)

println("=" ^ 90)
println("Gold sphere Mie validation (highly absorbing material)")
println("=" ^ 90)
@printf("  λ_0 = %.3f μm,  m_p = %.5f + %.4f i,  m_m = %.3f\n",
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

# ── Helper: compute errors vs Mie at the actual mesh r_ve ────────────────────
function compute_errors(basis, D_coeffs, cas_result, mesh)
    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1/√2, 1im/√2, 0.0)

    scat = compute_scattering(basis, D_coeffs;
                               k_hat=k_hat, E0=E0,
                               k0=k_bg, eps_p=eps_p, eps_bg=eps_bg,
                               csca_method=:farfield)

    V_mesh = total_volume(mesh)
    r_ve = (3V_mesh/(4π))^(1/3)
    mie_x_eff = mie_cross_sections(; wl_0=WL_0, m_m=M_M, r_p=r_ve, m_p=M_P)
    mie_S_eff = mie_cas_observables(; wl_0=WL_0, m_m=M_M, r_p=r_ve, m_p=M_P)

    return (
        r_ve     = r_ve,
        err_Cext = abs(scat.C_ext - mie_x_eff.C_ext) / mie_x_eff.C_ext,
        err_Cabs = abs(scat.C_abs - mie_x_eff.C_abs) / mie_x_eff.C_abs,
        err_Csca = abs(scat.C_sca - mie_x_eff.C_sca) / mie_x_eff.C_sca,
        err_Sfw  = abs(cas_result.S_fw - mie_S_eff.S_fw) / abs(mie_S_eff.S_fw),
        err_Sbk  = abs(cas_result.S_bk - mie_S_eff.S_bk) / abs(mie_S_eff.S_bk),
        C_ext = scat.C_ext, C_abs = scat.C_abs, C_sca = scat.C_sca,
        S_fw = cas_result.S_fw, S_bk = cas_result.S_bk,
        C_ext_mie = mie_x_eff.C_ext, C_abs_mie = mie_x_eff.C_abs,
        C_sca_mie = mie_x_eff.C_sca,
        S_fw_mie = mie_S_eff.S_fw, S_bk_mie = mie_S_eff.S_bk,
    )
end

# ── Mesh refinement sweep ────────────────────────────────────────────────────
const LC_LIST = (0.020, 0.014)
const ORI = (0.0, 0.0, 0.0)
const DUFFY = duffy_reference_rule(5)

println("\n" * "-" ^ 90)
@printf("  %-7s %-5s %-14s %-10s %-10s %-10s %-9s %-9s %-8s\n",
        "lc", "N", "method", "C_ext err", "C_abs err", "C_sca err",
        "S_fw err", "S_bk err", "t_total")
println("  " * "-" ^ 88)

for lc in LC_LIST
    path = sphere_mesh(RADIUS, lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces=true)
    N = n_basis(basis)

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1/√2, 1im/√2, 0.0)

    # ── Dense (LU) ───────────────────────────────────────────────────────
    GC.gc()
    t_dense = @elapsed begin
        cas_dense, D_dense = solve_cas_v2_orientations(basis, [ORI];
            wl_0=WL_0, m_m=M_M, m_p=M_P,
            duffy_rule=DUFFY, method=:dense, return_D=true)
    end
    e_d = compute_errors(basis, @view(D_dense[:, 1]), cas_dense[1], mesh)
    @printf("  %-7.4f %-5d %-14s %-10.3e %-10.3e %-10.3e %-9.3e %-9.3e %6.1fs\n",
            lc, N, "dense (LU)", e_d.err_Cext, e_d.err_Cabs, e_d.err_Csca,
            e_d.err_Sfw, e_d.err_Sbk, t_dense)
    flush(stdout)

    # ── AIM + BiCGSTAB ───────────────────────────────────────────────────
    GC.gc()
    t_aim = @elapsed begin
        cas_aim, D_aim = solve_cas_v2_orientations(basis, [ORI];
            wl_0=WL_0, m_m=M_M, m_p=M_P,
            duffy_rule=DUFFY, method=:aim_bicgstab,
            padding=4, tol=1e-8, maxiter=400, return_D=true)
    end
    e_a = compute_errors(basis, @view(D_aim[:, 1]), cas_aim[1], mesh)
    @printf("  %-7.4f %-5d %-14s %-10.3e %-10.3e %-10.3e %-9.3e %-9.3e %6.1fs\n",
            lc, N, "AIM-BiCGSTAB", e_a.err_Cext, e_a.err_Cabs, e_a.err_Csca,
            e_a.err_Sfw, e_a.err_Sbk, t_aim)
    flush(stdout)

    println("  " * "-" ^ 88)
end

# ── Summary ──────────────────────────────────────────────────────────────────
println("\nDone.")
