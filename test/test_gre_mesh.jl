using Test
using BlockVIEM
using Random: MersenneTwister

@testset "GRE mesh" begin
    # ── GREParams validation ─────────────────────────────────────────────
    @testset "GREParams construction & validation" begin
        p = GREParams(0.2, 3.0, 1.5, 0.1)
        @test p.r_v_base == 0.2
        @test p.bc_ratio == 3.0
        @test p.ab_ratio == 1.5
        @test p.beta == 0.1

        @test_throws ArgumentError GREParams(-0.1, 1.0, 1.0, 0.0)
        @test_throws ArgumentError GREParams(0.1, 0.5, 1.0, 0.0)
        @test_throws ArgumentError GREParams(0.1, 1.0, 0.5, 0.0)
        @test_throws ArgumentError GREParams(0.1, 1.0, 1.0, -0.1)
    end

    # ── Semi-axes ────────────────────────────────────────────────────────
    @testset "gre_semi_axes volume identity" begin
        for (rv, bc, ab) in [(0.2, 1.0, 1.0), (0.3, 3.0, 1.5), (0.5, 7.0, 2.0)]
            p = GREParams(rv, bc, ab, 0.0)
            a, b, c = gre_semi_axes(p)
            @test a >= b >= c
            V_ellipsoid = (4π / 3) * a * b * c
            V_target    = (4π / 3) * rv^3
            @test isapprox(V_ellipsoid, V_target; rtol=1e-12)
        end
    end

    # ── Adaptive lc ──────────────────────────────────────────────────────
    @testset "adaptive_lc" begin
        p = GREParams(0.2, 3.0, 1.0, 0.0)
        _, _, c = gre_semi_axes(p)
        lc_geo = adaptive_lc(p)
        @test lc_geo ≈ c / 3.0

        # with wavelength constraint
        lc_wl = adaptive_lc(p; wl_0=0.638, m_p_max=1.5, N_pw=10)
        @test lc_wl <= 0.638 / (1.5 * 10) + 1e-15
        @test lc_wl <= c / 3.0 + 1e-15

        # beta > 0 activates correlation-length constraint
        p2 = GREParams(0.2, 3.0, 1.0, 0.2)
        lc_def = adaptive_lc(p2)
        _, _, c2 = gre_semi_axes(p2)
        @test lc_def <= 0.3 * c2 / 3.0 + 1e-15
    end

    # ── Surface generation (β = 0) ───────────────────────────────────────
    @testset "gre_surface β=0 recovers ellipsoid" begin
        p = GREParams(0.3, 3.0, 1.5, 0.0)
        a, b, c = gre_semi_axes(p)
        rng = MersenneTwister(42)
        pts, Nθ, Nφ = gre_surface(p, rng; interp_factor=1)

        @test size(pts) == (3, Nθ * Nφ)
        # every point must lie on the ellipsoid: (x/a)² + (y/b)² + (z/c)² ≈ 1
        for k in 1:size(pts, 2)
            x, y, z = pts[1, k], pts[2, k], pts[3, k]
            @test isapprox((x / a)^2 + (y / b)^2 + (z / c)^2, 1.0; atol=1e-12)
        end
    end

    # ── Surface generation (β > 0) ───────────────────────────────────────
    @testset "gre_surface β>0 is reproducible" begin
        p = GREParams(0.2, 2.0, 1.0, 0.15)
        rng1 = MersenneTwister(123)
        rng2 = MersenneTwister(123)
        pts1, _, _ = gre_surface(p, rng1; interp_factor=2)
        pts2, _, _ = gre_surface(p, rng2; interp_factor=2)
        @test pts1 ≈ pts2
    end

    @testset "gre_surface β>0 points are near the ellipsoid" begin
        p = GREParams(0.2, 2.0, 1.5, 0.2)
        a, b, c = gre_semi_axes(p)
        rng = MersenneTwister(999)
        pts, _, _ = gre_surface(p, rng; interp_factor=2)

        # points should deviate from the ellipsoid, but not by too much
        for k in 1:size(pts, 2)
            x, y, z = pts[1, k], pts[2, k], pts[3, k]
            r_ell = (x / a)^2 + (y / b)^2 + (z / c)^2
            # should be within a factor of ~2 of the ellipsoid surface
            @test 0.3 < r_ell < 3.0
        end
    end

    # ── Mesh generation: β = 0 (OCC path) ────────────────────────────────
    @testset "gre_mesh β=0 sphere" begin
        p = GREParams(0.2, 1.0, 1.0, 0.0)
        rng = MersenneTwister(1)
        mesh, r_ve = gre_mesh(p, rng; lc=0.08)

        V_target = (4π / 3) * 0.2^3
        @test isapprox(total_volume(mesh), V_target; rtol=0.10)
        @test isapprox(r_ve, 0.2; rtol=0.05)
        @test n_tets(mesh) > 10
        @test all(>(0), mesh.tet_volumes)
    end

    @testset "gre_mesh β=0 oblate spheroid" begin
        p = GREParams(0.2, 3.0, 1.0, 0.0)
        rng = MersenneTwister(1)
        mesh, r_ve = gre_mesh(p, rng; lc=0.05)

        @test isapprox(r_ve, 0.2; rtol=0.03)
        @test n_tets(mesh) > 50
    end

    @testset "gre_mesh β=0 triaxial ellipsoid" begin
        p = GREParams(0.3, 2.0, 1.5, 0.0)
        a, b, c = gre_semi_axes(p)
        rng = MersenneTwister(1)
        mesh, r_ve = gre_mesh(p, rng; lc=0.06)

        V_target = (4π / 3) * a * b * c
        @test isapprox(total_volume(mesh), V_target; rtol=0.05)
        @test isapprox(r_ve, 0.3; rtol=0.03)
    end

    # ── Mesh generation: β > 0 (STL path) ────────────────────────────────
    @testset "gre_mesh β>0 deformed ellipsoid" begin
        p = GREParams(0.2, 2.0, 1.0, 0.1)
        rng = MersenneTwister(42)
        mesh, r_ve = gre_mesh(p, rng; lc=0.05)

        # volume should be in the right ballpark (deformation preserves
        # volume approximately for small β)
        @test isapprox(r_ve, 0.2; rtol=0.15)
        @test n_tets(mesh) > 50
        @test all(>(0), mesh.tet_volumes)
    end

    @testset "gre_mesh β>0 reproducible" begin
        p = GREParams(0.2, 1.5, 1.0, 0.15)
        mesh1, rv1 = gre_mesh(p, MersenneTwister(77); lc=0.06)
        mesh2, rv2 = gre_mesh(p, MersenneTwister(77); lc=0.06)
        @test isapprox(rv1, rv2; rtol=1e-10)
        @test n_tets(mesh1) == n_tets(mesh2)
    end

    # ── Adaptive lc integration ──────────────────────────────────────────
    @testset "gre_mesh with adaptive lc" begin
        p = GREParams(0.2, 3.0, 1.0, 0.0)
        rng = MersenneTwister(1)
        # let adaptive_lc choose; provide wavelength info
        mesh, r_ve = gre_mesh(p, rng; wl_0=0.638, m_p_max=1.5, N_pw=10)
        @test isapprox(r_ve, 0.2; rtol=0.05)
        @test n_tets(mesh) > 10
    end

    # ── Wide parameter sweep (smoke test) ────────────────────────────────
    @testset "parameter sweep smoke test" begin
        for bc in [1.0, 3.0, 7.0], ab in [1.0, 2.0], β in [0.0, 0.15]
            p = GREParams(0.2, bc, ab, β)
            rng = MersenneTwister(1)
            a, _, c = gre_semi_axes(p)
            lc_test = min(c / 2.0, 0.08)   # coarse mesh for speed
            mesh, r_ve = gre_mesh(p, rng; lc=lc_test)
            @test n_tets(mesh) > 5
            @test r_ve > 0
            # volume should be in reasonable range
            @test 0.5 * 0.2 < r_ve < 2.0 * 0.2
        end
    end
end
