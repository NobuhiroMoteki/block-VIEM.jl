# V9: half-SWG via the AIM iterative solver at lc = 0.35, 0.25, 0.18.
#
# Verifies that the Stage-2.7 AIM extension reproduces the dense half-SWG
# errors found in v8b at moderate mesh sizes, and enables solving larger
# meshes than dense LU allows.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v9_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v9_$(lc)")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path); gmsh.finalize()
    return path
end

function mean_h(mesh)
    s = 0.0; c = 0
    for tet in mesh.tets, (a, b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
    end
    s / c
end

const m_p    = 1.5 + 0.01im
const wl_0   = 10.0
const m_m    = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0     = 2π * m_m / wl_0
const k_hat  = Vec3(0, 0, 1)
const E0     = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

println("=" ^ 116); flush(stdout)
println("  V9: half-SWG via AIM iterative solver (m=1.5+0.01i, wl=10, ka_ve~0.63)")
println("=" ^ 116); flush(stdout)

@printf("%6s %8s %6s %7s %8s %6s | %10s %+7s %% | %10s %+7s %% | %10s %+7s %%\n",
        "lc", "N", "Nbnd", "h_bar", "t_build", "iter",
        "C_abs", "err", "C_sca", "err", "C_ext", "err")
println("-"^118); flush(stdout)

for lc in (0.5, 0.35, 0.25, 0.18)
    path = sphere_mesh(lc)
    mesh = read_msh(path)
    V_mesh = total_volume(mesh)
    r_ve = (3V_mesh / (4π))^(1/3)
    h_bar = mean_h(mesh)
    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)

    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N = n_basis(basis)
    N_bnd = sum(basis.is_boundary)

    pitch = 0.5 * h_bar
    t_build = @elapsed res = solve_iterative(basis;
                                               k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                               k_hat=k_hat, E0=E0,
                                               pitch=pitch, padding=4,
                                               tol=1e-7, maxiter=600)
    scat = compute_scattering(basis, res.D_coeffs;
                                k_hat=k_hat, E0=E0, k0=k0,
                                eps_p=eps_p, eps_bg=eps_bg,
                                csca_method=:farfield, n_theta=40)

    ea = 100 * (scat.C_abs - mie.C_abs) / mie.C_abs
    es = 100 * (scat.C_sca - mie.C_sca) / mie.C_sca
    ex = 100 * (scat.C_ext - mie.C_ext) / mie.C_ext

    @printf("%6.3f %8d %6d %7.3f %8.2f %6d | %10.4e %+7.3f   | %10.4e %+7.3f   | %10.4e %+7.3f\n",
            lc, N, N_bnd, h_bar, t_build, res.iterations,
            scat.C_abs, ea, scat.C_sca, es, scat.C_ext, ex)
    flush(stdout)
end
