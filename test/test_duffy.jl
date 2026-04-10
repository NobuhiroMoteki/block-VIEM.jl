using Test
using StaticArrays
using LinearAlgebra: norm
using BlockVIEM
using BlockVIEM: Vec3

const REF_TET_D = (Vec3(0, 0, 0), Vec3(1, 0, 0),
                   Vec3(0, 1, 0), Vec3(0, 0, 1))

@testset "Vertex Duffy quadrature" begin
    @testset "reference rule structure" begin
        for n in 1:5
            rule = duffy_reference_rule(n)
            @test length(rule.bary) == n^3
            @test length(rule.weights) == n^3
            @test rule.n_per_axis == n
            # All barycentric coordinates lie in [0,1] and sum to 1.
            for λ in rule.bary
                @test all(0 .≤ λ .≤ 1)
                @test sum(λ) ≈ 1.0
            end
            # Weights non-negative on the reference simplex.
            @test all(>=(0), rule.weights)
        end
    end

    @testset "constant integration ∫1 dV = V_T" begin
        for s in 1:4, n in 2:5
            pts, wts = duffy_quadrature(REF_TET_D, s, n)
            @test isapprox(sum(wts), 1 / 6; atol = 1e-12)
            @test all(p -> all(0 .≤ p .≤ 1 + eps()), pts)
        end
    end

    @testset "linear integration (exact for n ≥ 2)" begin
        # Exact value: ∫_T x dV = 1/24 (singular vertex at origin).
        for n in 2:5
            pts, wts = duffy_quadrature(REF_TET_D, 1, n)
            I_x = sum(wts[i] * pts[i][1] for i in eachindex(pts))
            I_y = sum(wts[i] * pts[i][2] for i in eachindex(pts))
            I_z = sum(wts[i] * pts[i][3] for i in eachindex(pts))
            @test isapprox(I_x, 1 / 24; atol = 1e-12)
            @test isapprox(I_y, 1 / 24; atol = 1e-12)
            @test isapprox(I_z, 1 / 24; atol = 1e-12)
        end
    end

    @testset "cubic integrand matches 5-point tet rule" begin
        # Smooth cubic integrand: g(r) = 1 + x + xy + xyz.
        g(r) = 1 + r[1] + r[1] * r[2] + r[1] * r[2] * r[3]
        I_tet = integrate(TET_QUAD_5PT, REF_TET_D, g)
        rule = duffy_reference_rule(5)
        pts, wts = duffy_quadrature(REF_TET_D, 1, rule)
        I_duffy = sum(wts[i] * g(pts[i]) for i in eachindex(pts))
        @test isapprox(I_duffy, I_tet; rtol = 1e-12)
    end

    @testset "translation/rotation: shifted regular tet" begin
        # Same tet shifted by (10, -3, 7); ∫1 dV must remain 1/6.
        shift = Vec3(10, -3, 7)
        v = (REF_TET_D[1] + shift, REF_TET_D[2] + shift,
             REF_TET_D[3] + shift, REF_TET_D[4] + shift)
        rule = duffy_reference_rule(4)
        for s in 1:4
            _, wts = duffy_quadrature(v, s, rule)
            @test isapprox(sum(wts), 1 / 6; atol = 1e-12)
        end
    end

    @testset "1/R singular integrand: Cauchy convergence" begin
        # Singular vertex at origin; integrand 1/|r| has integrable singularity.
        # Verify that successive Duffy refinements converge to a stable value.
        f(r) = 1 / norm(r)
        Is = Float64[]
        for n in (2, 4, 6, 8, 10)
            rule = duffy_reference_rule(n)
            pts, wts = duffy_quadrature(REF_TET_D, 1, rule)
            push!(Is, sum(wts[i] * f(pts[i]) for i in eachindex(pts)))
        end
        # All values are finite (Duffy cancels the 1/R singularity).
        @test all(isfinite, Is)
        # Cauchy convergence: differences shrink monotonically.
        diffs = abs.(diff(Is))
        for i in 1:(length(diffs) - 1)
            @test diffs[i + 1] ≤ diffs[i] + 1e-12
        end
        # Final relative gap < 1e-3.
        @test abs(Is[end] - Is[end - 1]) / abs(Is[end]) < 1e-3
    end

    @testset "polynomial: independence from singular-vertex choice" begin
        # For a polynomial integrand of degree ≤ 3, Duffy with n ≥ 4 is
        # exact, so the result must be identical (up to round-off) for all
        # four choices of "singular" vertex. This validates the local-vertex
        # permutation logic in `_duffy_permutation`.
        g(r) = 1 + r[1] + r[1] * r[2] + r[1] * r[2] * r[3]
        I_exact = integrate(TET_QUAD_5PT, REF_TET_D, g)
        rule = duffy_reference_rule(5)
        for s in 1:4
            pts, wts = duffy_quadrature(REF_TET_D, s, rule)
            I = sum(wts[i] * g(pts[i]) for i in eachindex(pts))
            @test isapprox(I, I_exact; rtol = 1e-12)
        end
    end
end
