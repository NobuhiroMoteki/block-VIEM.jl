#
# A.8: FFT-threading microbenchmark for the AIM MVP.
#
# Measures wall time of `aim_far_mvp_weighted` (the FFT-dominated core of
# every Krylov iteration) across the current FFTW thread count. Run with
# `julia -t N` for several N to chart the scaling.
#
#   julia --project -t 1  benchmarks/aim/a8_fft_threading.jl
#   julia --project -t 4  benchmarks/aim/a8_fft_threading.jl
#   julia --project -t 10 benchmarks/aim/a8_fft_threading.jl
#
# Reports:
#   - FFT-only time per MVP (sum over 4 convolutions: Wx, Wy, Wz, scalar)
#   - Full aim_mvp time (FFT + sparse SpMV + precorr)
#   - Speedup relative to 1 thread (if a prior 1-thread row was written
#     to a side-file; otherwise just absolute numbers).

using LinearAlgebra: norm
using Printf
using StaticArrays
using FFTW
using BlockVIEM
using BlockVIEM: Vec3, aim_far_mvp_weighted

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const MESH_PATH = joinpath(REPO, "benchmarks", "external", "buff-em",
                            "examples", "MieScattering", "Sphere_1675.vmsh")

const K0     = 0.316
const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0

function mean_edge_length(mesh)
    total = 0.0
    count = 0
    edges = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
    for tet in mesh.tets, (a,b) in edges
        total += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]])
        count += 1
    end
    return total / count
end

println("=" ^ 72)
println("  A.8  AIM MVP FFT-threading microbenchmark")
println("=" ^ 72)
println("  Julia threads : $(Threads.nthreads())")
println("  FFTW threads  : $(FFTW.get_num_threads())")
println()

mesh  = read_msh(MESH_PATH)
basis = build_swg_basis(mesh)
N     = n_basis(basis)
hmean = mean_edge_length(mesh)
pitch = 0.5 * hmean

println("  mesh  = Sphere_1675.vmsh,  N_swg = $N,  h̄ = $(round(hmean,digits=4))")
println("  pitch = $(round(pitch,digits=4))  (= 0.5 × h̄)")
println()

print("  building AIM operator ... "); flush(stdout)
t_build = @elapsed op = build_aim_operator(basis;
                                            k0 = K0,
                                            eps_p = EPS_P,
                                            eps_bg = EPS_BG,
                                            pitch = pitch,
                                            padding = 4)
@printf("done (%.2f s)\n", t_build)

Nx, Ny, Nz = op.projection.grid.dims
@printf("  grid  = (%d, %d, %d), doubled FFT = (%d, %d, %d)\n",
        Nx, Ny, Nz, 2Nx, 2Ny, 2Nz)
println()

# Representative input vector
x = randn(ComplexF64, N)

# ----- warmup (plan creation, JIT) -----
_ = aim_mvp(op, x)
_ = aim_far_mvp_weighted(op.projection, op.G_hat, op.k0,
                          op.kappa_v, op.kappa_avg, x)

# ----- FFT-only timing -----
function bench_fft(op, x, ntrials)
    t_best = Inf
    for _ in 1:ntrials
        t = @elapsed aim_far_mvp_weighted(op.projection, op.G_hat, op.k0,
                                           op.kappa_v, op.kappa_avg, x)
        t_best = min(t_best, t)
    end
    return t_best
end

function bench_mvp(op, x, ntrials)
    t_best = Inf
    for _ in 1:ntrials
        t = @elapsed aim_mvp(op, x)
        t_best = min(t_best, t)
    end
    return t_best
end

ntrials = 20
t_fft = bench_fft(op, x, ntrials)
t_mvp = bench_mvp(op, x, ntrials)

println("  results (best of $ntrials trials):")
@printf("    aim_far_mvp_weighted  (FFT core)  : %7.2f ms\n", 1e3*t_fft)
@printf("    full aim_mvp          (FFT+SpMV)  : %7.2f ms\n", 1e3*t_mvp)
println()

# Append result row for post-hoc scaling plots.
logfile = joinpath(@__DIR__, "a8_fft_threading.csv")
if !isfile(logfile)
    open(logfile, "w") do io
        println(io, "julia_threads,fftw_threads,t_fft_ms,t_mvp_ms")
    end
end
open(logfile, "a") do io
    @printf(io, "%d,%d,%.4f,%.4f\n",
            Threads.nthreads(), FFTW.get_num_threads(),
            1e3*t_fft, 1e3*t_mvp)
end
println("  appended to $(relpath(logfile, REPO))")
