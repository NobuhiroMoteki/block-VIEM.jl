using Test
using BlockVIEM
using LinearAlgebra: norm
using HDF5

# Analytical volume of two equal overlapping spheres of radius R with
# centre-distance d (d ≤ 2R): V = 2·(4π/3)R³ − V_lens, with lens volume
# V_lens = (π/12)(4R + d)(2R − d)².
function _two_sphere_volume(R::Real, d::Real)
    V_sph = (4π / 3) * R^3
    V_lens = (π / 12) * (4R + d) * (2R - d)^2
    return 2 * V_sph - V_lens
end

function _nearest_neighbour_distance(centers::AbstractMatrix{<:Real})
    N = size(centers, 2)
    N >= 2 || return Inf
    dmin = Inf
    @inbounds for i in 1:N-1, j in i+1:N
        dx = centers[1,i] - centers[1,j]
        dy = centers[2,i] - centers[2,j]
        dz = centers[3,i] - centers[3,j]
        d = sqrt(dx*dx + dy*dy + dz*dz)
        if d < dmin
            dmin = d
        end
    end
    return dmin
end

@testset "Aggregate mesh" begin

    # ── SphereAggregate construction & invariants ────────────────────────
    @testset "SphereAggregate construction" begin
        centers = [0.0 1.0 ; 0.0 0.0 ; 0.0 0.0]   # shape (3, 2)
        radii = [0.5, 0.5]
        agg = SphereAggregate(centers, radii)
        @test n_monomers(agg) == 2
        @test size(agg.centers) == (3, 2)
        @test monomer_volume_sum(agg) ≈ 2 * (4π / 3) * 0.5^3

        @test_throws ArgumentError SphereAggregate([1.0 2.0; 3.0 4.0], [0.5, 0.5])
        @test_throws ArgumentError SphereAggregate(centers, [0.5])
        @test_throws ArgumentError SphereAggregate(centers, [0.5, -0.1])
    end

    # ── Neck-radius parameterisation ─────────────────────────────────────
    @testset "neck_ratio / overlap_factor round-trip" begin
        for eps in (0.01, 0.02, 0.05, 0.10, 0.25)
            nr = overlap_to_neck_ratio(eps)
            @test nr ≈ sqrt((1 + eps)^2 - 1)
            eps_back = neck_ratio_to_overlap(nr)
            @test eps_back ≈ eps rtol=1e-12
        end
        @test_throws ArgumentError neck_ratio_to_overlap(-0.1)
        @test_throws ArgumentError neck_ratio_to_overlap(1.0)
        @test_throws ArgumentError overlap_to_neck_ratio(-0.01)
    end

    # ── Linear chain geometry ────────────────────────────────────────────
    @testset "make_linear_chain" begin
        agg = make_linear_chain(5, 0.02)
        @test n_monomers(agg) == 5
        # centres equispaced along x, centred at origin
        xs = agg.centers[1, :]
        @test xs ≈ 2 * 0.02 .* (-2:2)
        @test all(agg.centers[2, :] .== 0.0)
        @test all(agg.centers[3, :] .== 0.0)
        @test all(agg.radii .== 0.02)

        # with gap
        agg2 = make_linear_chain(3, 0.02; gap=0.01)
        step = 2 * 0.02 + 0.01
        @test agg2.centers[1, :] ≈ step .* (-1:1)
    end

    # ── Planar array geometry ────────────────────────────────────────────
    @testset "make_planar_array square" begin
        agg = make_planar_array(3, 2, 0.02; lattice=:square)
        @test n_monomers(agg) == 6
        @test all(agg.centers[3, :] .== 0.0)
        # expected in-plane spacing 2r = 0.04
        # uniqueness of x-coordinates up to 3 values
        xs_unique = unique(round.(agg.centers[1, :], digits=10))
        ys_unique = unique(round.(agg.centers[2, :], digits=10))
        @test length(xs_unique) == 3
        @test length(ys_unique) == 2
    end

    @testset "make_planar_array triangular" begin
        agg = make_planar_array(4, 3, 0.02; lattice=:triangular)
        @test n_monomers(agg) == 12
        @test all(agg.centers[3, :] .== 0.0)
        # Even rows at x = 0, a, 2a, 3a; odd rows at x = a/2, 3a/2, 5a/2, 7a/2.
        # → 8 distinct x-values for nx = 4 with at least one even and one odd row.
        xs = sort(unique(round.(agg.centers[1, :], digits=10)))
        @test length(xs) == 2 * 4
    end

    @testset "make_planar_array invalid lattice" begin
        @test_throws ArgumentError make_planar_array(2, 2, 0.02; lattice=:weird)
    end

    # ── FCC / BCC / HCP nearest-neighbour distances ──────────────────────
    @testset "FCC cluster nearest-neighbour distance" begin
        r = 0.02
        agg = make_fcc_cluster(r; cluster_radius=4r)
        @test n_monomers(agg) >= 4
        d_nn = _nearest_neighbour_distance(agg.centers)
        @test isapprox(d_nn, 2r; rtol=1e-12)
    end

    @testset "BCC cluster nearest-neighbour distance" begin
        r = 0.02
        agg = make_bcc_cluster(r; cluster_radius=5r)
        @test n_monomers(agg) >= 2
        d_nn = _nearest_neighbour_distance(agg.centers)
        @test isapprox(d_nn, 2r; rtol=1e-12)
    end

    @testset "HCP cluster nearest-neighbour distance" begin
        r = 0.02
        agg = make_hcp_cluster(r; cluster_radius=5r)
        @test n_monomers(agg) >= 2
        d_nn = _nearest_neighbour_distance(agg.centers)
        @test isapprox(d_nn, 2r; rtol=1e-6)
    end

    # ── PTSA HDF5 round-trip via a temporary file ────────────────────────
    @testset "ptsa_h5_key string format" begin
        key = ptsa_h5_key(0.02, 0.15, 0.90, 2.40, 100, 0)
        @test key == "0.0200/0.15/0.900/2.40/00100/0"
    end

    @testset "load_ptsa_h5 round-trip" begin
        h5_path = tempname() * ".h5"
        # HDF5.jl round-trip: Julia (3, Np) ↔ on-disk (Np, 3) ↔ Python (Np, 3)
        xp = [0.0 0.05 ; 0.0 0.0 ; 0.0 0.0]   # (3, 2) in Julia
        rp = [0.025, 0.025]
        key = ptsa_h5_key(0.025, 0.10, 0.95, 2.50, 2, 3)
        h5open(h5_path, "w") do h5f
            g = create_group(h5f, key)
            g["xp"] = xp
            g["rp"] = rp
        end

        agg = load_ptsa_h5(h5_path;
                    mean_rp=0.025, rel_std_rp=0.10, k=0.95,
                    Df=2.50, Np=2, agg_num=3)
        @test n_monomers(agg) == 2
        @test agg.metadata["source"] == "PTSA_HDF5"
        @test agg.metadata["h5_key"] == key
        # after recenter, sum of (r^3-weighted) positions should be zero
        cx, cy, cz = aggregate_centroid(agg)
        @test isapprox(cx, 0.0; atol=1e-14)
        @test isapprox(cy, 0.0; atol=1e-14)
        @test isapprox(cz, 0.0; atol=1e-14)

        # list_ptsa_keys finds the written key
        keys_found = list_ptsa_keys(h5_path)
        @test key in keys_found

        rm(h5_path; force=true)
    end

    # ── Gmsh meshing: single sphere reproduces a sphere mesh ─────────────
    @testset "mesh_sphere_aggregate single sphere" begin
        agg = SphereAggregate([0.0; 0.0; 0.0;;], [0.05])
        mesh, path = mesh_sphere_aggregate(agg;
                         overlap_factor=0.0, lc=0.015, verbosity=0)
        V = total_volume(mesh)
        V_true = (4π / 3) * 0.05^3
        # Linear-tet faceting of a sphere underestimates volume by a few %
        # (boundary triangle inscribed in the true sphere).
        @test isapprox(V, V_true; rtol=8e-2)
        @test isfile(path)
        rm(path; force=true)
    end

    # ── pair_neck_radius: exact formula for polydisperse pairs ───────────
    @testset "pair_neck_radius" begin
        # Equal-radius case reduces to overlap_to_neck_ratio
        r = 0.03
        eps = 0.05
        a_eq = pair_neck_radius(r, r, 2r; overlap_factor=eps)
        @test isapprox(a_eq, r * overlap_to_neck_ratio(eps); rtol=1e-12)

        # Polydisperse case, small ε: a_neck ≈ sqrt(2ε r_i r_j)
        r_i = 0.020;  r_j = 0.030;  d = r_i + r_j
        eps2 = 0.01
        a_exact = pair_neck_radius(r_i, r_j, d; overlap_factor=eps2)
        a_approx = sqrt(2 * eps2 * r_i * r_j)
        @test isapprox(a_exact, a_approx; rtol=2e-2)

        # No overlap (gap > 0 and ε = 0) → a_neck = 0
        @test pair_neck_radius(r_i, r_j, d + 1e-6; overlap_factor=0.0) == 0.0

        # Containment (very large ε with small j relative to i): capped at min R
        a_cap = pair_neck_radius(0.01, 0.001, 0.005; overlap_factor=1.0)
        @test a_cap == 0.002   # 2·r_j after inflation factor 2
    end

    # ── Polydisperse PTSA-like aggregate meshes correctly ────────────────
    @testset "mesh_sphere_aggregate polydisperse pair" begin
        r_i = 0.020;  r_j = 0.030
        d = r_i + r_j   # exact contact (PTSA convention)
        centers = reshape([0.0, 0.0, 0.0, d, 0.0, 0.0], 3, 2)
        agg = SphereAggregate(centers, [r_i, r_j])

        eps_eff = 0.05
        R_i = r_i * (1 + eps_eff)
        R_j = r_j * (1 + eps_eff)
        # General two-sphere overlap volume (polydisperse):
        V_sph = (4π / 3) * (R_i^3 + R_j^3)
        # Lens (spherical cap sum) formula for two unequal spheres at distance d:
        # V_lens = π(R_i + R_j − d)² [d² + 2d(R_i+R_j) − 3(R_i−R_j)²] / (12 d)
        V_lens = π * (R_i + R_j - d)^2 *
                 (d^2 + 2d*(R_i + R_j) - 3*(R_i - R_j)^2) / (12 * d)
        V_analytic = V_sph - V_lens

        # `rescale_to_target_volume=false` keeps the gmsh-faceted
        # overlap geometry intact for this analytical-volume check
        # (paper-production passes the default `true` to enforce
        # V_mesh = monomer_volume_sum exactly; v0.7.7+).
        mesh, path = mesh_sphere_aggregate(agg;
                         overlap_factor=eps_eff, lc=0.012, verbosity=0,
                         rescale_to_target_volume=false)
        V_mesh = total_volume(mesh)
        @test isapprox(V_mesh, V_analytic; rtol=8e-2)
        rm(path; force=true)
    end

    # ── Two-sphere neck volume ───────────────────────────────────────────
    @testset "mesh_sphere_aggregate 2-sphere neck volume" begin
        r = 0.05
        # Touching configuration: d = 2r.  overlap_factor inflates each radius.
        centers = reshape([0.0, 0.0, 0.0, 2r, 0.0, 0.0], 3, 2)
        agg = SphereAggregate(centers, [r, r])

        eps_eff = 0.05
        # effective inflated radius, effective separation (unchanged)
        R_eff = r * (1 + eps_eff)
        d_eff = 2r
        V_analytic = _two_sphere_volume(R_eff, d_eff)

        # See note above: disable rescale to inspect the raw faceted
        # overlap volume.
        mesh, path = mesh_sphere_aggregate(agg;
                         overlap_factor=eps_eff, lc=0.015, verbosity=0,
                         rescale_to_target_volume=false)
        V_mesh = total_volume(mesh)
        # Tet discretisation error ~ few %; rtol=0.08 is generous but robust.
        @test isapprox(V_mesh, V_analytic; rtol=8e-2)
        rm(path; force=true)
    end

    # ── Volume-preserving rescale enforces V_mesh == monomer_volume_sum ──
    @testset "mesh_sphere_aggregate rescale to target volume (v0.7.7+)" begin
        # Equal-radius doublet, paper-production geometry: R = a_eq/2^(1/3),
        # gap = 0.1·R, default overlap_factor=0.02.
        a_eq = 0.05
        R    = a_eq / 2^(1/3)
        gap  = 0.1 * R
        d    = 2R + gap
        centers = reshape([0.0, 0.0, -d/2, 0.0, 0.0, d/2], 3, 2)
        agg = SphereAggregate(centers, [R, R])

        # Default rescale=true → V_mesh == monomer_volume_sum to machine precision
        mesh, path = mesh_sphere_aggregate(agg;
                         overlap_factor=0.02, lc=0.012, verbosity=0)
        V_target = monomer_volume_sum(agg)
        @test isapprox(total_volume(mesh), V_target; rtol=1e-12)
        # Volume-equivalent radius == a_eq exactly
        r_ve = cbrt(3 * total_volume(mesh) / (4π))
        @test isapprox(r_ve, a_eq; rtol=1e-12)
        rm(path; force=true)

        # rescale=false → V_mesh ≈ inflated faceted volume (~few % off target)
        mesh2, path2 = mesh_sphere_aggregate(agg;
                          overlap_factor=0.02, lc=0.012, verbosity=0,
                          rescale_to_target_volume=false)
        @test !isapprox(total_volume(mesh2), V_target; rtol=1e-6)
        rm(path2; force=true)
    end

end # @testset "Aggregate mesh"
