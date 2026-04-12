# Phase A.5: AIM + BiCGSTAB integration test.
#
# Solves the full VIEM system iteratively using the AIM operator as the
# MVP, via solve_iterative in src/solver.jl. Compares against the dense
# direct solver (solve_direct) on the same mesh and material.
#
# Acceptance:
#   - BiCGSTAB converges to relative residual < 1e-6 in < 100 iterations
#   - |D_aim - D_direct| / |D_direct| < 1e-3 (consistency)
#   - C_abs, C_sca from the iterative solve match the direct solve to <1%

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const MESH_PATH = joinpath(REPO, "benchmarks", "external", "buff-em",
                            "examples", "MieScattering", "Sphere_677.vmsh")

function mean_edge_length(mesh)
    total = 0.0
    count = 0
    edges = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
    for tet in mesh.tets
        for (a, b) in edges
            total += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]])
            count += 1
        end
    end
    return total / count
end

println("=" ^ 78); flush(stdout)
println("  Phase A.5: AIM + BiCGSTAB integration test"); flush(stdout)
println("=" ^ 78); flush(stdout)

mesh = read_msh(MESH_PATH)
basis = build_swg_basis(mesh)
N = n_basis(basis)
h_mean = mean_edge_length(mesh)
println("  mesh  = Sphere_677.vmsh, N_swg = $N, h̄ = $(round(h_mean, digits=4))")

const K0     = 0.316
const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0

k_hat = Vec3(0, 0, 1)
E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

println("  k0 = $K0, eps_p = $EPS_P")
println()

# Direct solve (reference)
println("[1/2] Direct (dense LU) solve ...")
t_direct = @elapsed res_direct = solve_direct(basis; k0 = K0, eps_p = EPS_P,
                                                eps_bg = EPS_BG,
                                                k_hat = k_hat, E0 = E0)
@printf("  wall time: %.2f s\n", t_direct)
@printf("  residual:  %.3e\n", res_direct.residual_norm)
println()

# Iterative solve
const PITCH = 0.5 * h_mean
const PADDING = 4
println("[2/2] Iterative (AIM + BiCGSTAB) solve ...")
@printf("  pitch = %.4f (= 0.5 × h̄), padding = %d\n", PITCH, PADDING)
t_iter = @elapsed res_iter = solve_iterative(basis; k0 = K0, eps_p = EPS_P,
                                               eps_bg = EPS_BG,
                                               k_hat = k_hat, E0 = E0,
                                               pitch = PITCH, padding = PADDING,
                                               tol = 1e-6, maxiter = 200)
@printf("  wall time:   %.2f s\n", t_iter)
@printf("  iterations:  %d\n", res_iter.iterations)
@printf("  residual:    %.3e\n", res_iter.residual_norm)
@printf("  converged:   %s\n", res_iter.converged)
println()

# Consistency of solution vectors
rel_D = norm(res_iter.D_coeffs - res_direct.D_coeffs) / norm(res_direct.D_coeffs)
@printf("  |D_aim - D_direct| / |D_direct| = %.3e\n", rel_D)

# Compute cross sections for both
scat_direct = compute_scattering(basis, res_direct.D_coeffs;
                                  k_hat = k_hat, E0 = E0, k0 = K0,
                                  eps_p = EPS_P, eps_bg = EPS_BG,
                                  csca_method = :farfield, n_theta = 20)
scat_iter = compute_scattering(basis, res_iter.D_coeffs;
                                k_hat = k_hat, E0 = E0, k0 = K0,
                                eps_p = EPS_P, eps_bg = EPS_BG,
                                csca_method = :farfield, n_theta = 20)

@printf("  C_abs  direct = %.6e,  iter = %.6e, rel diff = %.2e\n",
        scat_direct.C_abs, scat_iter.C_abs,
        abs(scat_direct.C_abs - scat_iter.C_abs) / abs(scat_direct.C_abs))
@printf("  C_sca  direct = %.6e,  iter = %.6e, rel diff = %.2e\n",
        scat_direct.C_sca, scat_iter.C_sca,
        abs(scat_direct.C_sca - scat_iter.C_sca) / abs(scat_direct.C_sca))
@printf("  C_ext  direct = %.6e,  iter = %.6e, rel diff = %.2e\n",
        scat_direct.C_ext, scat_iter.C_ext,
        abs(scat_direct.C_ext - scat_iter.C_ext) / abs(scat_direct.C_ext))
println()
println("Speedup (direct / iter): $(round(t_direct / t_iter, digits=2))×")
