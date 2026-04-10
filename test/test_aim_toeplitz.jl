using Test
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3

@testset "Toeplitz Green kernel and FFT convolution" begin
    @testset "Toeplitz symmetry" begin
        grid = AIMGrid(Vec3(0, 0, 0), 0.5, (4, 4, 4))
        G = build_green_toeplitz(grid, 1.0)
        # Circulant symmetry: G[i,j,k] == G[2Nx-i+2, 2Ny-j+2, 2Nz-k+2]
        # (wrapping, 1-based). More simply, G[2,1,1] == G[8,1,1] for Nx=4.
        @test G[2, 1, 1] ≈ G[8, 1, 1]
        @test G[1, 2, 1] ≈ G[1, 8, 1]
        @test G[1, 1, 2] ≈ G[1, 1, 8]
        @test G[2, 3, 4] ≈ G[8, 7, 6]    # all three axes wrapped
        # Self-term is zero.
        @test G[1, 1, 1] == 0.0
    end

    @testset "kernel values match helmholtz_green" begin
        grid = AIMGrid(Vec3(0, 0, 0), 1.0, (5, 5, 5))
        k0 = 0.7
        G = build_green_toeplitz(grid, k0)
        h = grid.pitch
        # G[2,1,1] corresponds to Δi=1,Δj=0,Δk=0 → R = h = 1.0
        @test G[2, 1, 1] ≈ helmholtz_green(h, k0)
        # G[2,2,1] → Δi=1,Δj=1,Δk=0 → R = h√2
        @test G[2, 2, 1] ≈ helmholtz_green(h * sqrt(2), k0)
        # G[2,2,2] → Δi=1,Δj=1,Δk=1 → R = h√3
        @test G[2, 2, 2] ≈ helmholtz_green(h * sqrt(3), k0)
    end

    @testset "delta convolution recovers Green's function" begin
        grid = AIMGrid(Vec3(0, 0, 0), 1.0, (5, 5, 5))
        k0 = 1.0
        G_toep = build_green_toeplitz(grid, k0)
        G_hat = precompute_green_fft(G_toep)

        # Place a unit delta at the grid centre (3, 3, 3).
        delta = zeros(ComplexF64, 5, 5, 5)
        delta[3, 3, 3] = 1.0
        result = fft_convolve(G_hat, delta)

        h = grid.pitch
        for i in 1:5, j in 1:5, k in 1:5
            R = h * sqrt(Float64((i - 3)^2 + (j - 3)^2 + (k - 3)^2))
            if R > 0
                @test isapprox(result[i, j, k], helmholtz_green(R, k0);
                               rtol = 1e-10)
            else
                @test isapprox(result[i, j, k], 0.0; atol = 1e-12)
            end
        end
    end

    @testset "in-place fft_convolve! matches allocating version" begin
        grid = AIMGrid(Vec3(0, 0, 0), 1.0, (4, 4, 4))
        G_hat = precompute_green_fft(build_green_toeplitz(grid, 0.5))
        u = ComplexF64[sin(i + j + k) for i in 1:4, j in 1:4, k in 1:4]
        ref = fft_convolve(G_hat, u)
        result = similar(ref)
        buf = zeros(ComplexF64, 8, 8, 8)
        fft_convolve!(result, G_hat, u, buf)
        @test result ≈ ref
    end

    @testset "static limit (k0 = 0): real-valued kernel" begin
        grid = AIMGrid(Vec3(0, 0, 0), 1.0, (3, 3, 3))
        G = build_green_toeplitz(grid, 0.0)
        @test maximum(abs.(imag.(G))) < 1e-15
        # All non-self entries should be 1/(4πR).
        @test real(G[2, 1, 1]) ≈ 1 / (4π * 1.0)
    end
end
