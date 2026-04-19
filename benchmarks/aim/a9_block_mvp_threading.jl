#
# A.9: Block-MVP column-parallelization benchmark.
#
# Measures the wall-clock cost of `aim_mvp(op, X)` for a matrix X with
# L columns, sweeping:
#   (i)  the Julia thread count            (Threads.nthreads())
#   (ii) the FFTW thread count             (set_fft_threads(n))
#   (iii) the block width L
#
# After the Phase-2 column parallelization, the L-loop in aim_mvp runs
# under `Threads.@threads`, so the best throughput is achieved when
#
#     julia_threads × fftw_threads ≈ physical_cores
#
# This benchmark verifies that scaling.
#
# Usage:
#   julia --project -t 1  benchmarks/aim/a9_block_mvp_threading.jl
#   julia --project -t 4  benchmarks/aim/a9_block_mvp_threading.jl
#   julia --project -t 10 benchmarks/aim/a9_block_mvp_threading.jl

using LinearAlgebra: norm
using Printf
using StaticArrays
using FFTW
using BlockVIEM
using BlockVIEM: Vec3

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

function bench_block(op, X, ntrials)
    _ = aim_mvp(op, X)   # warmup
    t_best = Inf
    for _ in 1:ntrials
        t = @elapsed aim_mvp(op, X)
        t_best = min(t_best, t)
    end
    return t_best
end

println("=" ^ 72)
println("  A.9  Block-MVP column parallelization")
println("=" ^ 72)
@printf("  Julia threads : %d\n", Threads.nthreads())

mesh  = read_msh(MESH_PATH)
basis = build_swg_basis(mesh)
N     = n_basis(basis)
hmean = mean_edge_length(mesh)
pitch = 0.5 * hmean

print("  building AIM operator ... "); flush(stdout)
t_build = @elapsed op = build_aim_operator(basis;
                                            k0 = K0, eps_p = EPS_P,
                                            eps_bg = EPS_BG,
                                            pitch = pitch, padding = 4)
@printf("done (%.2f s)\n", t_build)
Nx, Ny, Nz = op.projection.grid.dims
@printf("  N_swg = %d,  grid = (%d,%d,%d),  doubled FFT = (%d,%d,%d)\n",
        N, Nx, Ny, Nz, 2Nx, 2Ny, 2Nz)
println()

# Sweep block width L and FFTW thread count.
Ls        = (1, 2, 4, 8)
fft_cfgs  = (1, 2, 4, Threads.nthreads())  # unique() done below
fft_cfgs  = Tuple(unique(fft_cfgs))

@printf("  %-4s  %-8s  %-10s  %-10s\n",
        "L", "fft_thr", "t_best [ms]", "ms/column")
println("  " * "-"^44)

for L in Ls
    X = randn(ComplexF64, N, L)
    for fftn in fft_cfgs
        set_fft_threads(fftn)
        t = bench_block(op, X, 15)
        @printf("  %-4d  %-8d  %10.3f  %10.3f\n",
                L, fftn, 1e3*t, 1e3*t / L)
    end
    println("  " * "-"^44)
end
