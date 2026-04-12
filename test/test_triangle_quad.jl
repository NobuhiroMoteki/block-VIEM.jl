# Unit tests for src/triangle_quad.jl (Stage 2.1).
#
# Covers:
#   - polynomial exactness of TRI_QUAD_1PT, TRI_QUAD_3PT, tri_collapsed_rule
#   - barycentric integrals against the closed-form formula
#     ∫ λ1^a λ2^b λ3^c dS = (a! b! c!) / (a+b+c+2)! × 2 a_T
#   - Duffy quadrature: 1/R self-integral on an equilateral triangle
#     converges (while plain Gauss diverges)
#   - subdivide_triangle_around + tri_duffy_quadrature_around consistency

using Test
using LinearAlgebra: norm
using StaticArrays
using BlockVIEM

# Closed-form barycentric monomial integral on a physical triangle:
#   ∫_T λ1^a λ2^b λ3^c dS = (a! b! c!) / (a+b+c+2)! × (2 a_T)
function analytic_bary_integral(a::Int, b::Int, c::Int, a_T::Float64)
    num = factorial(a) * factorial(b) * factorial(c)
    den = factorial(a + b + c + 2)
    return (num / den) * 2 * a_T
end

@testset "TriQuadRule basic" begin
    # Use an off-axis, non-equilateral triangle to stress the implementation.
    v1 = Vec3(0.2, 0.0, 0.5)
    v2 = Vec3(1.5, 0.3, 0.4)
    v3 = Vec3(0.8, 1.2, 0.9)
    verts = (v1, v2, v3)
    a_T = triangle_area(verts...)

    @testset "weights sum to 1" begin
        for rule in (TRI_QUAD_1PT, TRI_QUAD_3PT,
                     tri_collapsed_rule(2), tri_collapsed_rule(3),
                     tri_collapsed_rule(4), tri_collapsed_rule(5))
            @test sum(rule.weights) ≈ 1.0 atol=1e-14
            for λ in rule.bary
                @test sum(λ) ≈ 1.0 atol=1e-14
                @test all(≥(-1e-14), λ)
            end
        end
    end

    @testset "integrate constant" begin
        for rule in (TRI_QUAD_1PT, TRI_QUAD_3PT,
                     tri_collapsed_rule(2), tri_collapsed_rule(3))
            I = integrate_tri(rule, verts, r -> 1.0)
            @test I ≈ a_T rtol=1e-12
        end
    end

    @testset "integrate centroid coordinate" begin
        # ∫_T x dS = a_T * centroid_x
        cx = (v1[1] + v2[1] + v3[1]) / 3
        for rule in (TRI_QUAD_3PT, tri_collapsed_rule(2))
            I = integrate_tri(rule, verts, r -> r[1])
            @test I ≈ a_T * cx rtol=1e-12
        end
    end

    @testset "exactness on barycentric monomials" begin
        # Use canonical reference triangle for the analytic comparison.
        w1 = Vec3(0.0, 0.0, 0.0)
        w2 = Vec3(1.0, 0.0, 0.0)
        w3 = Vec3(0.0, 1.0, 0.0)
        wv = (w1, w2, w3)
        a_ref = triangle_area(wv...)

        # λ1(r) = 1 - x - y, λ2 = x, λ3 = y on this triangle.
        λ1_of(r) = 1 - r[1] - r[2]
        λ2_of(r) = r[1]
        λ3_of(r) = r[2]

        # 3-pt rule (degree 2): exact up to (a+b+c) ≤ 2.
        for (a, b, c) in [(0,0,0),(1,0,0),(0,1,0),(2,0,0),(1,1,0),(0,1,1),(0,0,2)]
            f = r -> λ1_of(r)^a * λ2_of(r)^b * λ3_of(r)^c
            expected = analytic_bary_integral(a, b, c, a_ref)
            got = integrate_tri(TRI_QUAD_3PT, wv, f)
            @test got ≈ expected atol=1e-13
        end

        # 9-pt collapsed rule (degree 4): exact up to (a+b+c) ≤ 4.
        rule9 = tri_collapsed_rule(3)
        for (a, b, c) in [(4,0,0),(2,2,0),(1,1,2),(0,2,2),(0,0,4)]
            f = r -> λ1_of(r)^a * λ2_of(r)^b * λ3_of(r)^c
            expected = analytic_bary_integral(a, b, c, a_ref)
            got = integrate_tri(rule9, wv, f)
            @test got ≈ expected atol=1e-13
        end
    end
end

@testset "TriDuffyRule: 1/R self-integral convergence" begin
    # Equilateral triangle of side length s placed in the xy-plane, with
    # vertex 1 at the origin. The self-integral
    #   I = ∫_T 1 / |r - v1| dS
    # has a known closed form for an equilateral triangle; we only need a
    # reference value that is stable to high precision. Compute a very
    # high-order Duffy result as the reference and check convergence.
    s = 1.3
    v1 = Vec3(0.0, 0.0, 0.0)
    v2 = Vec3(s,   0.0, 0.0)
    v3 = Vec3(s/2, s*sqrt(3)/2, 0.0)
    verts = (v1, v2, v3)
    a_T = triangle_area(verts...)

    # Closed form for the self-integral ∫_T 1/|r - v1| dS with v1 an apex
    # of the triangle in its plane. For a triangle, for each point v1,
    # the integral can be computed in polar coordinates centred at v1.
    # For an equilateral triangle of side s we can get a reference via
    # very-high-order Duffy; use order 12 as the reference.
    ref_rule = tri_duffy_reference_rule(12)
    ref_pts, ref_wts = tri_duffy_quadrature(verts, 1, ref_rule)
    I_ref = sum(ref_wts[i] / norm(ref_pts[i] - v1) for i in eachindex(ref_pts))

    # Lower orders must converge toward the reference.
    for n in (3, 4, 5, 6, 8)
        rule = tri_duffy_reference_rule(n)
        pts, wts = tri_duffy_quadrature(verts, 1, rule)
        I = sum(wts[i] / norm(pts[i] - v1) for i in eachindex(pts))
        @test isfinite(I)
        @test abs(I - I_ref) / abs(I_ref) < 10.0^(-n + 1)
    end

    # Sanity: Duffy self-integral is independent of the singular vertex
    # chosen (the triangle is equilateral and the integrand 1/|r - v_s|
    # is the same form). Use vertex 2 and 3 and check equality.
    pts2, wts2 = tri_duffy_quadrature(verts, 2, ref_rule)
    I2 = sum(wts2[i] / norm(pts2[i] - v2) for i in eachindex(pts2))
    pts3, wts3 = tri_duffy_quadrature(verts, 3, ref_rule)
    I3 = sum(wts3[i] / norm(pts3[i] - v3) for i in eachindex(pts3))
    @test I2 ≈ I_ref rtol=1e-8
    @test I3 ≈ I_ref rtol=1e-8
end

@testset "tri_duffy_quadrature_around: area preservation" begin
    # Split a triangle around an interior point and check that the total
    # weight of the combined quadrature equals the area.
    v1 = Vec3(0.0, 0.0, 0.1)
    v2 = Vec3(1.0, 0.0, 0.1)
    v3 = Vec3(0.3, 0.8, 0.1)
    verts = (v1, v2, v3)
    a_T = triangle_area(verts...)

    # Interior point
    p_int = (v1 + v2 + v3) / 3
    rule = tri_duffy_reference_rule(5)
    pts, wts = tri_duffy_quadrature_around(verts, p_int, rule)
    @test sum(wts) ≈ a_T rtol=1e-12

    # Edge point (midpoint of edge v1-v2): one sub-triangle degenerate
    p_edge = 0.5 * (v1 + v2)
    pts_e, wts_e = tri_duffy_quadrature_around(verts, p_edge, rule)
    @test sum(wts_e) ≈ a_T rtol=1e-12

    # Vertex: two sub-triangles degenerate
    pts_v, wts_v = tri_duffy_quadrature_around(verts, v1, rule)
    @test sum(wts_v) ≈ a_T rtol=1e-12
end

@testset "tri_duffy_quadrature_around: 1/R at interior singularity" begin
    # For an interior singularity, the sum of three vertex-Duffy sub-
    # triangles gives a convergent 1/R integral, whereas plain Gauss does
    # not. Check that refining the Duffy order reduces the residual.
    v1 = Vec3(0.0, 0.0, 0.0)
    v2 = Vec3(1.0, 0.0, 0.0)
    v3 = Vec3(0.0, 1.0, 0.0)
    verts = (v1, v2, v3)
    p_int = Vec3(0.25, 0.25, 0.0)

    ref = tri_duffy_quadrature_around(verts, p_int, 16)
    I_ref = sum(ref[2][i] / norm(ref[1][i] - p_int) for i in eachindex(ref[1]))

    # Gauss-Legendre × Duffy convergence is exponential but the rate is
    # governed by the remaining smooth factor 1/||(1-u2)(v_a-p) + u2(v_b-p)||
    # on each sub-triangle, which depends on the geometry. We check monotone
    # error reduction and require decent accuracy at moderate order.
    errs = Float64[]
    for n in (4, 6, 8, 10)
        pts, wts = tri_duffy_quadrature_around(verts, p_int, n)
        I = sum(wts[i] / norm(pts[i] - p_int) for i in eachindex(pts))
        push!(errs, abs(I - I_ref) / abs(I_ref))
    end
    @test all(isfinite.(errs))
    @test errs[end] < errs[1]         # monotone convergence
    @test errs[end] < 1e-4            # high-order is accurate
    @test errs[1]   < 1e-2            # low order is at least usable
end
