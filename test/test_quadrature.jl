using Test
using StaticArrays
using LinearAlgebra: norm
using BlockVIEM
using BlockVIEM: Vec3

const REF_TET = (Vec3(0, 0, 0), Vec3(1, 0, 0),
                 Vec3(0, 1, 0), Vec3(0, 0, 1))

# Reference monomial integrals over the standard 3-simplex of volume 1/6:
# ∫ x^a y^b z^c dV = a! b! c! / (a + b + c + 3)!
exact_monomial(a, b, c) = factorial(a) * factorial(b) * factorial(c) /
                          factorial(a + b + c + 3)

@testset "Tetrahedron Gauss quadrature" begin
    @testset "1-point rule (degree 1)" begin
        rule = TET_QUAD_1PT
        @test sum(rule.weights) ≈ 1.0
        @test integrate(rule, REF_TET, r -> 1.0) ≈ 1 / 6
        @test integrate(rule, REF_TET, r -> r[1]) ≈ exact_monomial(1, 0, 0)
        # Quadratic should be inexact for the 1-point rule.
        I_x2_1pt = integrate(rule, REF_TET, r -> r[1]^2)
        @test !isapprox(I_x2_1pt, exact_monomial(2, 0, 0); rtol = 1e-2)
    end

    @testset "4-point rule (degree 2)" begin
        rule = TET_QUAD_4PT
        @test sum(rule.weights) ≈ 1.0
        @test all(>(0), rule.weights)         # all positive weights
        @test integrate(rule, REF_TET, r -> 1.0) ≈ 1 / 6
        @test integrate(rule, REF_TET, r -> r[1]) ≈ exact_monomial(1, 0, 0)
        @test integrate(rule, REF_TET, r -> r[1]^2) ≈ exact_monomial(2, 0, 0)
        @test integrate(rule, REF_TET, r -> r[1] * r[2]) ≈ exact_monomial(1, 1, 0)
        # Cubic should be inexact for the 4-point rule.
        I_x3 = integrate(rule, REF_TET, r -> r[1]^3)
        @test !isapprox(I_x3, exact_monomial(3, 0, 0); rtol = 1e-3)
    end

    @testset "5-point rule (degree 3)" begin
        rule = TET_QUAD_5PT
        @test sum(rule.weights) ≈ 1.0
        @test integrate(rule, REF_TET, r -> 1.0) ≈ 1 / 6
        @test integrate(rule, REF_TET, r -> r[1]^3) ≈ exact_monomial(3, 0, 0)
        @test integrate(rule, REF_TET, r -> r[1] * r[2] * r[3]) ≈
              exact_monomial(1, 1, 1)
    end

end

@testset "Gauss-Legendre 1D on [0,1]" begin
    @testset "node-weight invariants" begin
        for n in 1:8
            nodes, weights = gauss_legendre_unit(n)
            @test length(nodes) == n
            @test length(weights) == n
            @test sum(weights) ≈ 1.0      # ∫_0^1 1 dx
            @test all(>(0), weights)
            @test all(0 .≤ nodes .≤ 1)
        end
    end

    @testset "polynomial precision = 2n - 1" begin
        for n in 1:6
            nodes, weights = gauss_legendre_unit(n)
            for k in 0:(2 * n - 1)
                I = sum(weights .* (nodes .^ k))
                @test isapprox(I, 1 / (k + 1); atol = 1e-12)
            end
        end
    end
end
