# Phase A.2: rigorous AIM vs dense Z validation.
#
# Goal: confirm that `aim_mvp(op, x)` reproduces `Z * x` from the dense
# impedance matrix to better than 0.5% relative error on a realistic mesh,
# across a range of AIM parameters (grid pitch, stencil size, poly order).
#
# Runs on the buff-em Sphere_677 mesh (677 tets, 1188 SWG DOFs) with the
# eps=10+1i material at ka=0.316. Same mesh and material as the buff-em
# cross-validation in benchmarks/runs.

using LinearAlgebra: norm, dot
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, _tet_vertices, tet_volume

# Resolve mesh path relative to this script
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const MESH_PATH = joinpath(REPO, "benchmarks", "external", "buff-em",
                            "examples", "MieScattering", "Sphere_677.vmsh")

println("=" ^ 78)
println("  Phase A.2: AIM vs dense Z (SWG/RT0 on Sphere_677)")
println("=" ^ 78)

"""
Mean edge length across all tetrahedra in `mesh` (each edge counted once
per incident tet; sufficient for scaling heuristics).
"""
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
V_mesh = total_volume(mesh)
h_mean = mean_edge_length(mesh)
println("  mesh  = Sphere_677.vmsh")
println("  N_swg = $N")
println("  V     = $(round(V_mesh, digits=4))")
println("  h̄    = $(round(h_mean, digits=4)) (mean edge length)")

# Physical parameters (match buff-em validation)
const K0     = 0.316
const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0

println("  k0    = $K0")
println("  eps_p = $EPS_P")
println()

# Build dense Z once (expensive but only done here)
println("Assembling dense Z matrix ...")
t_dense = @elapsed begin
    Z = assemble_impedance_matrix(basis; k0 = K0, eps_p = EPS_P,
                                   eps_bg = EPS_BG,
                                   symmetrize = false)
end
@printf("  dense assembly: %.2f s\n", t_dense)

# Generate a deterministic test vector: mixture of unit-norm random
# vectors (for generic accuracy) and a plane-wave-like RHS.
using Random
Random.seed!(42)
x_rand = randn(ComplexF64, N)
x_rand ./= norm(x_rand)

E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
k_hat = Vec3(0, 0, 1)
k_bg = ComplexF64(K0) * sqrt(ComplexF64(EPS_BG))
b_pw = project_plane_wave(basis; k_hat = k_hat, E0 = E0, k_bg = k_bg)
b_pw ./= norm(b_pw)

test_vectors = [("random", x_rand), ("plane-wave RHS", b_pw)]

# Reference products
y_dense_rand = Z * x_rand
y_dense_pw   = Z * b_pw
refs = Dict("random" => y_dense_rand, "plane-wave RHS" => y_dense_pw)

# Parameter sweep: (pitch / h̄, padding, poly_order, stencil_size)
# Conservative AIM defaults: poly_order=2, stencil=3 → 10 moments in 27 points.
configs = [
    (pitch_ratio = 1.0,  padding = 3, poly = 2, stencil = 3),
    (pitch_ratio = 0.75, padding = 3, poly = 2, stencil = 3),
    (pitch_ratio = 0.5,  padding = 3, poly = 2, stencil = 3),
    (pitch_ratio = 0.5,  padding = 4, poly = 2, stencil = 3),
    (pitch_ratio = 0.5,  padding = 4, poly = 3, stencil = 4),
    (pitch_ratio = 0.35, padding = 4, poly = 2, stencil = 3),
    (pitch_ratio = 0.35, padding = 4, poly = 3, stencil = 4),
]

println("=== AIM parameter sweep ===")
@printf("%6s %4s %4s %4s | %9s %9s | %10s %10s | %10s\n",
        "p/h̄", "pad", "P", "M", "grid_dims", "Ngrid", "rand rel", "pw rel", "build (s)")
println("-"^90)

best = Dict{String,Tuple{Float64,NamedTuple}}()

for cfg in configs
    pitch = cfg.pitch_ratio * h_mean
    local op, t_build
    try
        t_build = @elapsed begin
            op = build_aim_operator(basis;
                                    k0 = K0, eps_p = EPS_P, eps_bg = EPS_BG,
                                    pitch = pitch, padding = cfg.padding,
                                    poly_order = cfg.poly,
                                    stencil_size = cfg.stencil)
        end
    catch err
        @printf("%6.2f %4d %4d %4d | (build error: %s)\n",
                cfg.pitch_ratio, cfg.padding, cfg.poly, cfg.stencil, sprint(showerror, err))
        continue
    end

    # Accuracy test
    errs = Dict{String,Float64}()
    for (name, x) in test_vectors
        y_aim = aim_mvp(op, x)
        y_ref = refs[name]
        errs[name] = norm(y_aim - y_ref) / norm(y_ref)
    end

    dims = op.projection.grid.dims
    Ngrid = prod(dims)
    @printf("%6.2f %4d %4d %4d | %3d×%3d×%3d | %8d | %10.2e %10.2e | %10.2f\n",
            cfg.pitch_ratio, cfg.padding, cfg.poly, cfg.stencil,
            dims[1], dims[2], dims[3], Ngrid,
            errs["random"], errs["plane-wave RHS"], t_build)

    for name in keys(errs)
        if !haskey(best, name) || errs[name] < best[name][1]
            best[name] = (errs[name], cfg)
        end
    end
end

println()
println("=== Best parameters per test vector ===")
for (name, (err, cfg)) in best
    @printf("  %-18s rel_err = %.2e  pitch/h̄=%.2f pad=%d P=%d M=%d\n",
            name, err, cfg.pitch_ratio, cfg.padding, cfg.poly, cfg.stencil)
end

println()
println("Acceptance: AIM vs dense < 0.5% on both test vectors.")
println()
