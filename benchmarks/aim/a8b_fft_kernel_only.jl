#
# A.8b: FFT kernel scaling (synthetic) for the AIM MVP core.
#
# Skips AIM-operator construction entirely and just times the 4 FFT
# convolutions `aim_far_mvp_weighted` performs per MVP, at grid sizes
# that bracket the real R/5 ... R/10 problem range.
#
#   julia --project -t 1  benchmarks/aim/a8b_fft_kernel_only.jl
#   julia --project -t 4  ...
#   julia --project -t 10 ...

using Printf
using FFTW
using StaticArrays
using BlockVIEM: build_green_toeplitz, precompute_green_fft,
                 fft_convolve, AIMGrid, aim_grid, Vec3

# Four representative doubled-FFT sizes (2N on each axis):
#   46³  : Sphere N~3k    (current A.5 test)
#   64³  : doublet R/5
#   96³  : doublet R/7    (the regression case)
#  128³  : doublet R/10   (memory-plan target)
const SIZES = [(46,46,46), (64,64,64), (96,96,96), (128,128,128)]

println("=" ^ 72)
println("  A.8b  FFT-only scaling  —  pure fft_convolve cost per MVP")
println("=" ^ 72)
@printf("  Julia threads : %d\n", Threads.nthreads())
@printf("  FFTW threads  : %d\n", FFTW.get_num_threads())
println()
@printf("  %-14s %-10s %-14s %-14s\n",
        "doubled FFT", "N_grid", "1 conv [ms]", "4 convs [ms]")
println("  " * "-"^54)

for (Dx, Dy, Dz) in SIZES
    Nx, Ny, Nz = Dx÷2, Dy÷2, Dz÷2
    # Fake AIMGrid for build_green_toeplitz
    grid = AIMGrid(Vec3(-1.0, -1.0, -1.0), 0.05, (Nx, Ny, Nz))
    k0 = 0.316 + 0.0im
    G_toep = build_green_toeplitz(grid, k0)
    G_hat  = precompute_green_fft(G_toep)
    u = randn(ComplexF64, Nx, Ny, Nz)

    # warmup
    _ = fft_convolve(G_hat, u)

    ntrials = 30
    t1 = Inf
    for _ in 1:ntrials
        t = @elapsed fft_convolve(G_hat, u)
        t1 = min(t1, t)
    end
    # The real MVP does 4 convolutions (Wx/Wy/Wz + scalar)
    t4 = Inf
    for _ in 1:ntrials
        t = @elapsed begin
            fft_convolve(G_hat, u); fft_convolve(G_hat, u)
            fft_convolve(G_hat, u); fft_convolve(G_hat, u)
        end
        t4 = min(t4, t)
    end
    @printf("  (%3d,%3d,%3d)  %-10d %12.3f  %12.3f\n",
            Dx, Dy, Dz, Nx*Ny*Nz, 1e3*t1, 1e3*t4)
end

