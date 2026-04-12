# Phase A.2b: AIM vs dense Z diagnostic with k0 varying.
#
# We observed in A.2 that AIM vs dense gives 27-40% relative error at
# k0=0.316, way above the expected <1%. To diagnose, sweep k0 from
# near-static (1e-4) to moderate (1) and watch the error behavior.
#
# Near-static: G_hat is close to G_0 = 1/(4πR). Moment matching should
# work well. Large error here means the precorrection or the moment
# setup itself is wrong.
#
# Moderate k0: the Helmholtz factor exp(-ik0R) introduces a smooth
# modulation. If static is accurate but moderate is not, the issue is
# multipole truncation (poly_order too low).

using LinearAlgebra: norm, dot
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

mesh = read_msh(MESH_PATH)
basis = build_swg_basis(mesh)
N = n_basis(basis)
h_mean = mean_edge_length(mesh)

const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0
const PITCH  = 0.35 * h_mean
const PAD    = 4

println("=" ^ 78)
println("  Phase A.2b: AIM vs dense Z, k0 sweep")
println("  mesh=Sphere_677 N=$N  pitch=0.35·h̄  padding=$PAD  P=2 M=3")
println("=" ^ 78)

using Random
Random.seed!(42)
x = randn(ComplexF64, N)
x ./= norm(x)

@printf("%9s | %10s %10s | %10s\n",
        "k0", "rel err", "C_abs(x)", "C_abs(dense)")
println("-"^56)

for k0 in [1e-4, 1e-3, 0.01, 0.1, 0.316, 1.0]
    Z = assemble_impedance_matrix(basis; k0 = k0, eps_p = EPS_P,
                                   eps_bg = EPS_BG, symmetrize = false)
    op = build_aim_operator(basis; k0 = k0, eps_p = EPS_P, eps_bg = EPS_BG,
                            pitch = PITCH, padding = PAD,
                            poly_order = 2, stencil_size = 3)
    y_dense = Z * x
    y_aim   = aim_mvp(op, x)
    rel = norm(y_aim - y_dense) / norm(y_dense)
    @printf("%9.1e | %10.2e %10.2e %10.2e\n",
            k0, rel, abs(dot(x, y_aim)), abs(dot(x, y_dense)))
end
println()

# Diagnostic: separate the mass and K contributions.
println("=== Split: mass term alone vs K term alone ===")
k0 = 0.316
Z_full = assemble_impedance_matrix(basis; k0 = k0, eps_p = EPS_P,
                                    eps_bg = EPS_BG, symmetrize = false)
# Mass-only dense: kappa = 0 trick
Z_mass = assemble_impedance_matrix(basis; k0 = k0, eps_p = EPS_P,
                                    eps_bg = EPS_P, symmetrize = false)
# (when eps_bg == eps_p, kappa = 0, so Z reduces to (1/eps_p) M)
Z_K = Z_full - Z_mass    # contains -(kappa/eps_bg) K

op = build_aim_operator(basis; k0 = k0, eps_p = EPS_P, eps_bg = EPS_BG,
                        pitch = PITCH, padding = PAD,
                        poly_order = 2, stencil_size = 3)
y_full_dense = Z_full * x
y_full_aim   = aim_mvp(op, x)
y_mass_dense = Z_mass * x
y_K_dense    = Z_K * x
y_K_aim      = y_full_aim - y_mass_dense   # = (1/eps_p)M x from dense minus everything else

@printf("  |y_full_dense - y_full_aim| / |y_full_dense| = %.2e\n",
        norm(y_full_dense - y_full_aim)/norm(y_full_dense))
@printf("  |y_K_dense    - y_K_aim   | / |y_K_dense  | = %.2e\n",
        norm(y_K_dense - y_K_aim)/norm(y_K_dense))
println()
