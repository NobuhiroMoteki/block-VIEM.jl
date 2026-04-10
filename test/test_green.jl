using Test
using BlockVIEM

@testset "Helmholtz Green's function" begin
    @testset "static limit" begin
        for R in (0.1, 1.0, 10.0, 1e-3)
            @test helmholtz_green_static(R) ≈ 1 / (4π * R)
            @test helmholtz_green(R, 0.0) ≈ helmholtz_green_static(R)
        end
    end

    @testset "phase at integer multiples of π" begin
        R = 1.5
        # k0 R = π  ->  exp(-iπ) = -1.
        k0 = π / R
        G = helmholtz_green(R, k0)
        @test real(G) ≈ -1 / (4π * R) atol = 1e-12
        @test imag(G) ≈ 0.0 atol = 1e-12

        # k0 R = 2π -> exp(-i2π) = 1.
        k0 = 2π / R
        G = helmholtz_green(R, k0)
        @test real(G) ≈ 1 / (4π * R) atol = 1e-12
        @test imag(G) ≈ 0.0 atol = 1e-12

        # k0 R = π/2 -> exp(-iπ/2) = -i.
        k0 = π / (2 * R)
        G = helmholtz_green(R, k0)
        @test real(G) ≈ 0.0 atol = 1e-12
        @test imag(G) ≈ -1 / (4π * R) atol = 1e-12
    end

    @testset "lossy medium: exponential decay" begin
        # k0 = 1 - 0.1im (Im(k0) < 0  =>  decay for outgoing wave).
        k0 = 1.0 - 0.1im
        R1, R2 = 1.0, 10.0
        G1 = helmholtz_green(R1, k0)
        G2 = helmholtz_green(R2, k0)
        # |G(R)| = exp(Im(k0) * R) / (4π R)  for our sign convention.
        @test abs(G1) ≈ exp(imag(k0) * R1) / (4π * R1)
        @test abs(G2) ≈ exp(imag(k0) * R2) / (4π * R2)
        @test abs(G2) < abs(G1)               # decays with distance
    end

    @testset "domain check" begin
        @test_throws ArgumentError helmholtz_green(0.0, 1.0)
        @test_throws ArgumentError helmholtz_green(-1.0, 1.0)
        @test_throws ArgumentError helmholtz_green_static(0.0)
    end

    @testset "type stability" begin
        @inferred helmholtz_green(1.0, 1.0)
        @inferred helmholtz_green(1.0, 1.0 + 0.1im)
        @inferred helmholtz_green_static(1.0)
    end
end
