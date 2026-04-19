#
# A.11: `assemble_precorrection` scaling benchmark (Phase-4 S2).
#
# Partitions the outer `near_cols` loop into interleaved column chunks
# and runs them as Threads.@spawn tasks; each task owns its own COO
# buffers and reads the shared stencil cache + Green tensor.  For any
# real-scale problem this is the dominant `build_aim_operator` cost, so
# wall-time drop here drives the end-to-end improvement.
#
#   julia --project -t 1  benchmarks/aim/a11_precorrection_build.jl
#   julia --project -t 4  benchmarks/aim/a11_precorrection_build.jl
#   julia --project -t 10 benchmarks/aim/a11_precorrection_build.jl

using LinearAlgebra: norm
using Printf
using BlockVIEM
using BlockVIEM: assemble_precorrection, build_aim_projection,
                 build_green_toeplitz, precompute_green_fft, aim_grid

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const MESH_PATH = joinpath(REPO, "benchmarks", "external", "buff-em",
                            "examples", "MieScattering", "Sphere_1675.vmsh")

const K0     = 0.316
const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0

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
println("  A.11  assemble_precorrection scaling")
println("=" ^ 72)
@printf("  Julia threads : %d\n", Threads.nthreads())

mesh  = read_msh(MESH_PATH)
basis = build_swg_basis(mesh)
N     = n_basis(basis)
hmean = mean_edge_length(mesh)
pitch = 0.5 * hmean
grid  = aim_grid(basis.mesh; pitch = pitch, padding = 4)

proj  = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
G_tp  = build_green_toeplitz(grid, K0)
G_hat = precompute_green_fft(G_tp)

@printf("  N_swg = %d,  grid = %s\n", N, repr(grid.dims))
println()

# Warmup
_ = assemble_precorrection(basis, proj, G_hat;
                            k0 = K0, eps_p = EPS_P, eps_bg = EPS_BG)

ntrials = 3
times = Float64[]
for trial in 1:ntrials
    GC.gc()
    t = @elapsed assemble_precorrection(basis, proj, G_hat;
                                         k0 = K0, eps_p = EPS_P,
                                         eps_bg = EPS_BG)
    push!(times, t)
end

t_best = minimum(times)
t_mean = sum(times) / length(times)
@printf("  best  : %.3f s\n", t_best)
@printf("  mean  : %.3f s  (n=%d)\n", t_mean, ntrials)
println()

logfile = joinpath(@__DIR__, "a11_precorrection_build.csv")
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
