# V4: VIEM-SWG at lc=0.15 vs DDA at d=0.14 (1.3% C_abs error).
#
# The DDA reference comes from benchmarks/rt0/v3_dda_results.json (low contrast,
# m=1.5+0.01i, wl=10, ka_ve≈0.628). The question is whether VIEM-SWG closes the
# 4-5x per-DOF gap when pushed to comparable resolution.
#
# Outcomes (from memory/v2_v3_dda_comparison.md):
#   1. VIEM-SWG ≈ 1.3% -> no bug, SWG just has poor DOF efficiency
#   2. VIEM-SWG ≈ 5-10% -> ~4x gap is real, points to a discretization issue
#   3. VIEM-SWG > 10%   -> strong evidence of a bug; investigate self-term

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v4_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v4_$(lc)")
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

# DDA reference (from v3_dda_results.json, low contrast row d=0.15)
const DDA_D014_C_ABS_ERR_PCT = 1.245
const DDA_D014_N_DOF = 1496 * 3   # vector DOFs
const DDA_D014_LATTICE = 0.1409

# Physical setup: low contrast, ka_ve ~ 0.628
const m_p   = 1.5 + 0.01im
const wl_0  = 10.0
const m_m   = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0    = 2π * m_m / wl_0
const k_hat = Vec3(0, 0, 1)
const E0    = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

println("=" ^ 88); flush(stdout)
println("  V4: VIEM-SWG vs DDA at d=0.14 (1.3% C_abs error)")
println("  Low contrast: m=1.5+0.01i, wl=10, ka_ve~0.628")
println("=" ^ 88); flush(stdout)

@printf("%6s %7s %7s %8s %8s | %10s %10s %7s | %10s %10s %7s | %10s %10s %7s\n",
        "lc", "N_swg", "h_bar", "ka_ve", "t(s)",
        "C_abs_V", "C_abs_M", "err%",
        "C_sca_V", "C_sca_M", "err%",
        "C_ext_V", "C_ext_M", "err%")
println("-"^132); flush(stdout)

# Walk in: lc=0.18 first (sanity, should slot between b1's 0.18 and 0.13),
# then lc=0.15 (the head-to-head with DDA d=0.14).
results = NamedTuple[]
for lc in (0.18, 0.15)
    path = sphere_mesh(lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    V_mesh = total_volume(mesh)
    r_ve = (3V_mesh / (4π))^(1/3)
    h_bar = mean_h(mesh)
    ka_ve = k0 * r_ve

    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)

    pitch = 0.5 * h_bar
    t = @elapsed res = solve_iterative(basis;
                                        k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                        k_hat=k_hat, E0=E0,
                                        pitch=pitch, padding=4,
                                        tol=1e-7, maxiter=600)
    scat = compute_scattering(basis, res.D_coeffs;
                               k_hat=k_hat, E0=E0, k0=k0,
                               eps_p=eps_p, eps_bg=eps_bg,
                               csca_method=:farfield, n_theta=20)

    ea = 100 * abs(scat.C_abs - mie.C_abs) / abs(mie.C_abs)
    es = 100 * abs(scat.C_sca - mie.C_sca) / abs(mie.C_sca)
    ex = 100 * abs(scat.C_ext - mie.C_ext) / abs(mie.C_ext)

    @printf("%6.3f %7d %7.3f %8.4f %8.1f | %10.3e %10.3e %6.2f%% | %10.3e %10.3e %6.2f%% | %10.3e %10.3e %6.2f%%\n",
            lc, N, h_bar, ka_ve, t,
            scat.C_abs, mie.C_abs, ea,
            scat.C_sca, mie.C_sca, es,
            scat.C_ext, mie.C_ext, ex)
    flush(stdout)
    push!(results, (; lc, N, h_bar, ka_ve, t, ea, es, ex))
end

println(); flush(stdout)
println("  --- DDA reference (v3_dda_results.json, d=0.14, N_dipole=1496) ---")
@printf("    DDA C_abs err = %.2f%%   (vector DOFs = %d, lattice = %.3f)\n",
        DDA_D014_C_ABS_ERR_PCT, DDA_D014_N_DOF, DDA_D014_LATTICE)

if !isempty(results)
    r = results[end]
    gap = r.ea / DDA_D014_C_ABS_ERR_PCT
    println()
    @printf("  VIEM-SWG lc=%.2f : N_swg=%d, C_abs err=%.2f%%   gap to DDA = %.1fx\n",
            r.lc, r.N, r.ea, gap)
    println()
    if r.ea < 2.0
        println("  Verdict: outcome 1 (no bug; SWG closes the gap with refinement).")
    elseif r.ea < 10.0
        println("  Verdict: outcome 2 (4x DDA-vs-VIEM gap is real -- discretization issue).")
    else
        println("  Verdict: outcome 3 (>10%, strong bug evidence; inspect self-term).")
    end
end
flush(stdout)
