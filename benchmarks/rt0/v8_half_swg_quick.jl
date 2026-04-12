# V8 quick: half-SWG (K^A+K^B+K^C+K^D) sanity check at lc=0.5 only.
#
# Compare dense-solve cross sections with and without boundary-face DOFs
# against Mie, same mesh, same solver, same post-processing. Purpose:
# verify that the Stage 2 implementation moves the error in the CORRECT
# direction before investing in larger meshes.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v8_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v8_$(lc)")
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

function run_one(basis, k0, eps_p, eps_bg, k_hat, E0, wl_0, m_m, m_p, r_ve)
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
    b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg)
    t_asm = @elapsed Z = assemble_impedance_matrix(basis;
                                                     k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                                     symmetrize=true)
    t_sol = @elapsed D = Z \ b
    scat = compute_scattering(basis, D;
                                k_hat=k_hat, E0=E0, k0=k0,
                                eps_p=eps_p, eps_bg=eps_bg,
                                csca_method=:farfield, n_theta=20)
    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
    return (; t_asm, t_sol, scat, mie, N=length(b))
end

const m_p    = 1.5 + 0.01im
const wl_0   = 10.0
const m_m    = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0     = 2π * m_m / wl_0
const k_hat  = Vec3(0, 0, 1)
const E0     = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

lc = 0.5
path = sphere_mesh(lc)
mesh = read_msh(path)
V_mesh = total_volume(mesh)
r_ve = (3V_mesh / (4π))^(1/3)

println("=" ^ 96); flush(stdout)
println("  V8 quick: half-SWG sanity (lc=$(lc), m=$(m_p), ka_ve≈$(round(k0*r_ve, digits=3)))")
println("=" ^ 96); flush(stdout)

basis_int = build_swg_basis(mesh; include_boundary_faces = false)
basis_full = build_swg_basis(mesh; include_boundary_faces = true)
println("  Basis sizes: interior-only = $(n_basis(basis_int)),",
        "  half-SWG = $(n_basis(basis_full)) (+$(n_basis(basis_full) - n_basis(basis_int)))")
flush(stdout)

println()
println("  Running interior-only (K^A only)...")
flush(stdout)
r_int = run_one(basis_int, k0, eps_p, eps_bg, k_hat, E0, wl_0, m_m, m_p, r_ve)

println("  Running half-SWG (K^A+B+C+D)...")
flush(stdout)
r_full = run_one(basis_full, k0, eps_p, eps_bg, k_hat, E0, wl_0, m_m, m_p, r_ve)

println()
@printf("%18s | %8s %8s %9s | %10s %+7s %% | %10s %+7s %% | %10s %+7s %%\n",
        "basis", "N", "t_asm", "t_sol",
        "C_abs", "err", "C_sca", "err", "C_ext", "err")
println("-" ^ 110)

for (label, r) in (("interior-only", r_int), ("half-SWG", r_full))
    ea = 100 * (r.scat.C_abs - r.mie.C_abs) / r.mie.C_abs
    es = 100 * (r.scat.C_sca - r.mie.C_sca) / r.mie.C_sca
    ex = 100 * (r.scat.C_ext - r.mie.C_ext) / r.mie.C_ext
    @printf("%18s | %8d %8.2f %9.2f | %10.4e %+7.2f %% | %10.4e %+7.2f %% | %10.4e %+7.2f %%\n",
            label, r.N, r.t_asm, r.t_sol,
            r.scat.C_abs, ea, r.scat.C_sca, es, r.scat.C_ext, ex)
end
println()

@printf("  Mie ref at r_ve=%.4f: C_abs=%.4e  C_sca=%.4e  C_ext=%.4e\n",
        r_ve, r_int.mie.C_abs, r_int.mie.C_sca, r_int.mie.C_ext)
flush(stdout)
