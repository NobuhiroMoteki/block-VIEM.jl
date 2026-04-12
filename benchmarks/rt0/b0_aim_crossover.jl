# Phase B.0: find the N at which AIM beats dense LU for RT0/SWG.
#
# Times direct vs iterative solves at several mesh resolutions so the
# Phase B.1 convergence study can pick the right path for each lc.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_b0_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("b0_$(lc)")
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

const M_P   = 1.5 + 0.01im
const EPS_P = ComplexF64(M_P)^2
const EPS_BG = 1.0
const K0    = 2π / 10.0
const K_HAT = Vec3(0, 0, 1)
const E0    = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

flush(stdout)
println("=" ^ 88); flush(stdout)
println("  Phase B.0: AIM vs dense LU crossover (RT0, ε_p = $EPS_P, k0 = $(round(K0, digits=4)))"); flush(stdout)
println("=" ^ 88); flush(stdout)
@printf("%6s %6s %6s | %10s %10s | %10s %6s %10s | %10s %8s\n",
        "lc", "Ntet", "N", "t_dense(s)", "C_abs_d", "t_iter(s)", "niter", "C_abs_i",
        "|ΔD|/|D|", "speedup")
println("-"^100); flush(stdout)

for lc in [0.7, 0.5, 0.35, 0.25, 0.18]
    path = sphere_mesh(lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    h_bar = mean_h(mesh)

    t_dense = @elapsed res_d = solve_direct(basis; k0=K0, eps_p=EPS_P, eps_bg=EPS_BG,
                                              k_hat=K_HAT, E0=E0)
    scat_d = compute_scattering(basis, res_d.D_coeffs; k_hat=K_HAT, E0=E0,
                                 k0=K0, eps_p=EPS_P, eps_bg=EPS_BG,
                                 csca_method=:farfield, n_theta=15)

    pitch = 0.5 * h_bar
    t_iter = @elapsed res_i = solve_iterative(basis; k0=K0, eps_p=EPS_P, eps_bg=EPS_BG,
                                                k_hat=K_HAT, E0=E0,
                                                pitch=pitch, padding=4,
                                                tol=1e-6, maxiter=200)
    scat_i = compute_scattering(basis, res_i.D_coeffs; k_hat=K_HAT, E0=E0,
                                 k0=K0, eps_p=EPS_P, eps_bg=EPS_BG,
                                 csca_method=:farfield, n_theta=15)
    rel_D = norm(res_i.D_coeffs - res_d.D_coeffs) / norm(res_d.D_coeffs)

    @printf("%6.3f %6d %6d | %10.2f %10.4e | %10.2f %6d %10.4e | %10.2e %8.2f\n",
            lc, length(mesh.tets), N,
            t_dense, scat_d.C_abs,
            t_iter, res_i.iterations, scat_i.C_abs,
            rel_D, t_dense / t_iter)
    flush(stdout)
end
