#
# A.10: `build_aim_projection` scaling benchmark (Phase-4 S1).
#
# Times `build_aim_projection` at several Julia thread counts; the
# function now partitions the basis-loop into n_chunks = min(nthr, N)
# tasks that each assemble their own COO triplets, so wall time is
# expected to drop roughly linearly up to ~8 threads (limited by BLAS
# nested-threading safety / QR cost per basis).
#
#   julia --project -t 1  benchmarks/aim/a10_projection_build.jl
#   julia --project -t 4  benchmarks/aim/a10_projection_build.jl
#   julia --project -t 10 benchmarks/aim/a10_projection_build.jl

using LinearAlgebra: norm
using Printf
using BlockVIEM

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const MESH_PATH = joinpath(REPO, "benchmarks", "external", "buff-em",
                            "examples", "MieScattering", "Sphere_1675.vmsh")

function mean_edge_length(mesh)
    total = 0.0; count = 0
    edges = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
    for tet in mesh.tets, (a,b) in edges
        total += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]])
        count += 1
    end
    return total / count
end

println("=" ^ 72)
println("  A.10  build_aim_projection scaling")
println("=" ^ 72)
@printf("  Julia threads : %d\n", Threads.nthreads())

mesh  = read_msh(MESH_PATH)
basis = build_swg_basis(mesh)
N     = n_basis(basis)
hmean = mean_edge_length(mesh)
pitch = 0.5 * hmean
grid  = aim_grid(basis.mesh; pitch = pitch, padding = 4)

@printf("  N_swg = %d,  grid = %s,  stencil = 3, poly_order = 2\n",
        N, repr(grid.dims))
println()

# Warmup
_ = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)

ntrials = 3
times = Float64[]
for trial in 1:ntrials
    GC.gc()
    t = @elapsed build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    push!(times, t)
end

t_best = minimum(times)
t_mean = sum(times) / length(times)
@printf("  best  : %.3f s\n", t_best)
@printf("  mean  : %.3f s  (n=%d)\n", t_mean, ntrials)
println()

logfile = joinpath(@__DIR__, "a10_projection_build.csv")
if !isfile(logfile)
    open(logfile, "w") do io
        println(io, "julia_threads,N,t_best_s,t_mean_s")
    end
end
open(logfile, "a") do io
    @printf(io, "%d,%d,%.4f,%.4f\n",
            Threads.nthreads(), N, t_best, t_mean)
end
println("  appended to $(relpath(logfile, REPO))")
