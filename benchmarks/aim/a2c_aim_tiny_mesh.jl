# Phase A.2c: AIM vs dense Z on a small Gmsh sphere (lc=0.7).
#
# Rapid diagnostic: uses a small mesh so the dense reference is cheap.
# Sweeps k0, AIM pitch, and poly order to localize the bug.
#
# stdout is flushed aggressively so we see progress in real time.

using LinearAlgebra: norm, dot
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

flush(stdout)
println("=" ^ 78); flush(stdout)
println("  Phase A.2c: AIM diagnostic on small sphere mesh"); flush(stdout)
println("=" ^ 78); flush(stdout)

function small_sphere_mesh(radius, lc)
    path = joinpath(tempdir(), "sphere_tiny_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("tiny$(lc)")
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

function mean_edge_length(mesh)
    total = 0.0; count = 0
    edges = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
    for tet in mesh.tets, (a,b) in edges
        total += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]])
        count += 1
    end
    return total / count
end

path = small_sphere_mesh(1.0, 0.7)
mesh = read_msh(path)
basis = build_swg_basis(mesh)
N = n_basis(basis)
h = mean_edge_length(mesh)
println("  N_tet=$(length(mesh.tets))  N_swg=$N  h̄=$(round(h,digits=4))"); flush(stdout)

using Random
Random.seed!(42)
x = randn(ComplexF64, N); x ./= norm(x)

const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0

# --- 1. k0 sweep at fixed AIM parameters ---
println("\n=== k0 sweep ==="); flush(stdout)
@printf("%10s | %10s | %10s\n", "k0", "rel err", "dense time")
println("-"^40); flush(stdout)

for k0 in (1e-4, 1e-2, 0.1, 0.316, 1.0)
    t_Z = @elapsed Z = assemble_impedance_matrix(basis; k0=k0, eps_p=EPS_P,
                                                  eps_bg=EPS_BG, symmetrize=false)
    op = build_aim_operator(basis; k0=k0, eps_p=EPS_P, eps_bg=EPS_BG,
                            pitch=0.35*h, padding=4,
                            poly_order=2, stencil_size=3)
    y_d = Z * x; y_a = aim_mvp(op, x)
    rel = norm(y_a - y_d) / norm(y_d)
    @printf("%10.1e | %10.2e | %8.2f s\n", k0, rel, t_Z); flush(stdout)
end

# --- 2. Pitch sweep at k0=1e-4 (should be essentially static) ---
println("\n=== Pitch sweep at k0=1e-4 (nearly static) ==="); flush(stdout)
k0 = 1e-4
Z_static = assemble_impedance_matrix(basis; k0=k0, eps_p=EPS_P, eps_bg=EPS_BG,
                                      symmetrize=false)
y_d = Z_static * x
@printf("%10s %6s | %10s\n", "pitch/h̄", "pad", "rel err"); flush(stdout)
println("-"^32)
for (pr, pad) in [(1.0, 3), (0.5, 3), (0.5, 5), (0.25, 5), (0.15, 5), (0.1, 5)]
    op = build_aim_operator(basis; k0=k0, eps_p=EPS_P, eps_bg=EPS_BG,
                            pitch=pr*h, padding=pad,
                            poly_order=2, stencil_size=3)
    y_a = aim_mvp(op, x)
    rel = norm(y_a - y_d) / norm(y_d)
    @printf("%10.3f %6d | %10.2e\n", pr, pad, rel); flush(stdout)
end

# --- 3. Split the error: mass (k=0) vs radiation (k>0) ---
println("\n=== Split error diagnosis at k0=1e-4 ==="); flush(stdout)
k0 = 1e-4
Z = assemble_impedance_matrix(basis; k0=k0, eps_p=EPS_P, eps_bg=EPS_BG,
                               symmetrize=false)
op = build_aim_operator(basis; k0=k0, eps_p=EPS_P, eps_bg=EPS_BG,
                        pitch=0.25*h, padding=5,
                        poly_order=2, stencil_size=3)

# Full dense & aim
y_d_full = Z * x
y_a_full = aim_mvp(op, x)

# Mass-only action (κ=0 → kappa=0 inside AIM operator construction)
# Easiest: use eps_bg = eps_p so kappa = 0
op_mass = build_aim_operator(basis; k0=k0, eps_p=EPS_P, eps_bg=EPS_P,
                              pitch=0.25*h, padding=5,
                              poly_order=2, stencil_size=3)
y_a_mass = aim_mvp(op_mass, x)
Z_mass_only = assemble_impedance_matrix(basis; k0=k0, eps_p=EPS_P,
                                         eps_bg=EPS_P, symmetrize=false)
y_d_mass = Z_mass_only * x

# K-only = Full - Mass
y_d_K = y_d_full - y_d_mass
y_a_K = y_a_full - y_a_mass

@printf("  FULL: |y_d-y_a|/|y_d|  = %.2e\n", norm(y_a_full-y_d_full)/norm(y_d_full))
@printf("  MASS: |y_d-y_a|/|y_d|  = %.2e\n", norm(y_a_mass-y_d_mass)/norm(y_d_mass))
@printf("  K   : |y_d-y_a|/|y_d|  = %.2e   (|y_d_K|=%.2e, |y_d_full|=%.2e)\n",
        norm(y_a_K-y_d_K)/max(norm(y_d_K),1e-30),
        norm(y_d_K), norm(y_d_full))
flush(stdout)
