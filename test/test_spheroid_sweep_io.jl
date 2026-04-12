using Test
using BlockVIEM
using HDF5

@testset "expand_alpha_from_alpha0" begin
    # For a sphere-like degenerate case (S_θ = S_φ at α = 0), the α-expansion
    # should give a constant in α (since B = 0).
    S_th0 = 0.123 + 0.045im
    S_ph0 = S_th0
    α = collect(0:π/8:π)
    Sth, Sph = expand_alpha_from_alpha0(S_th0, S_ph0, α)
    @test all(isapprox.(Sth, S_th0; atol = 1e-15))
    @test all(isapprox.(Sph, S_ph0; atol = 1e-15))

    # General case: at α = 0 the expansion must reproduce the inputs.
    S_th0 = 0.2 + 0.1im
    S_ph0 = 0.15 + 0.08im
    Sth, Sph = expand_alpha_from_alpha0(S_th0, S_ph0, [0.0])
    @test isapprox(Sth[1], S_th0; atol = 1e-15)
    @test isapprox(Sph[1], S_ph0; atol = 1e-15)

    # At α = π/2:  exp(2j·π/2) = -1, so S_θ(π/2) = A - B = S_φ(0).
    Sth, Sph = expand_alpha_from_alpha0(S_th0, S_ph0, [π/2])
    @test isapprox(Sth[1], S_ph0; atol = 1e-14)
    @test isapprox(Sph[1], S_th0; atol = 1e-14)

    # Periodicity in α with period π:  exp(+2j(α+π)) = exp(+2jα).
    Sth_a, _ = expand_alpha_from_alpha0(S_th0, S_ph0, [0.7])
    Sth_b, _ = expand_alpha_from_alpha0(S_th0, S_ph0, [0.7 + π])
    @test isapprox(Sth_a[1], Sth_b[1]; atol = 1e-14)
end

@testset "spheroid_sweep HDF5 round trip" begin
    # Minimal grid for fast testing
    N_Dve, N_RI, N_AR, N_u, N_ph = 2, 2, 3, 4, 5
    grids = SpheroidSweepGrids(
        collect(range(0.30, 0.50, length = N_Dve)),
        collect(range(1.30, 1.60, length = N_RI)),
        collect(range(-0.5, 0.5, length = N_AR)),
        collect(range(0.0, 1.0, length = N_u)),
        collect(range(0.0, π, length = N_ph)),
    )

    # Two wavelengths, distinct synthetic data so we can verify per-group order
    function make_data(wl_0, m_m, scale)
        S_theta = Array{ComplexF64}(undef, N_Dve, N_RI, N_AR, N_u, N_ph)
        S_phi   = similar(S_theta)
        for i in 1:N_Dve, j in 1:N_RI, k in 1:N_AR, m in 1:N_u, n in 1:N_ph
            S_theta[i,j,k,m,n] = scale * (i + 0.1j + 0.01k) + im * scale * (0.001m + 0.0001n)
            S_phi[i,j,k,m,n]   = -S_theta[i,j,k,m,n]
        end
        converged = trues(N_Dve, N_RI, N_AR)
        # Mark one cell as non-converged with NaNs
        converged[1,1,1] = false
        S_theta[1,1,1,:,:] .= NaN + NaN*im
        S_phi[1,1,1,:,:]   .= NaN + NaN*im
        return SpheroidSweepData(wl_0, m_m, S_theta, S_phi, converged)
    end

    data1 = make_data(0.638, 1.0, 1.0)
    data2 = make_data(0.834, 1.0, 2.0)

    path = joinpath(tempdir(), "spheroid_sweep_test_$(getpid()).h5")
    write_spheroid_sweep_h5(path, grids, [data1, data2];
                            block_viem_version = "0.1.1-test",
                            solver_tol = 1.0e-6,
                            extra_root_attrs = Dict("rng_seed" => 42))

    # Schema-level checks
    h5open(path, "r") do f
        @test haskey(f, "D_ve_grid")
        @test haskey(f, "RI_real_grid")
        @test haskey(f, "log_AR_grid")
        @test haskey(f, "cos_theta_o_half_grid")
        @test haskey(f, "phi_o_grid")
        @test haskey(f, "wl_0p638")
        @test haskey(f, "wl_0p834")

        @test attrs(f)["m_m"] == 1.0
        @test attrs(f)["solver_tol"] == 1.0e-6
        @test attrs(f)["block_viem_version"] == "0.1.1-test"
        @test attrs(f)["rng_seed"] == 42

        grp = f["wl_0p638"]
        @test attrs(grp)["wl_0"] == 0.638
        @test attrs(grp)["m_m"] == 1.0

        @test size(read(grp["S_fw_theta_re"])) == (N_Dve, N_RI, N_AR, N_u, N_ph)
        @test size(read(grp["S_fw_theta_im"])) == (N_Dve, N_RI, N_AR, N_u, N_ph)
        @test size(read(grp["S_fw_phi_re"]))   == (N_Dve, N_RI, N_AR, N_u, N_ph)
        @test size(read(grp["S_fw_phi_im"]))   == (N_Dve, N_RI, N_AR, N_u, N_ph)
        @test size(read(grp["converged"]))     == (N_Dve, N_RI, N_AR)
    end

    # Round-trip via the public reader
    grids2, data_per_wl2 = read_spheroid_sweep_h5(path)
    @test grids2.D_ve              == grids.D_ve
    @test grids2.RI_real           == grids.RI_real
    @test grids2.log_AR            == grids.log_AR
    @test grids2.cos_theta_o_half  == grids.cos_theta_o_half
    @test grids2.phi_o             == grids.phi_o

    @test length(data_per_wl2) == 2
    @test data_per_wl2[1].wl_0 == 0.638
    @test data_per_wl2[2].wl_0 == 0.834

    # Compare entries that are not NaN-marked
    function _eq_with_nan_skip(A, B)
        size(A) == size(B) || return false
        @inbounds for i in eachindex(A)
            if !(isnan(real(A[i])) || isnan(real(B[i])))
                A[i] == B[i] || return false
            end
        end
        return true
    end
    @test _eq_with_nan_skip(data_per_wl2[1].S_fw_theta, data1.S_fw_theta)
    @test _eq_with_nan_skip(data_per_wl2[1].S_fw_phi,   data1.S_fw_phi)
    @test _eq_with_nan_skip(data_per_wl2[2].S_fw_theta, data2.S_fw_theta)
    @test _eq_with_nan_skip(data_per_wl2[2].S_fw_phi,   data2.S_fw_phi)
    @test data_per_wl2[1].converged == data1.converged
    @test data_per_wl2[2].converged == data2.converged

    rm(path; force = true)
end

@testset "non-equidistant grids rejected" begin
    bad_grids = SpheroidSweepGrids(
        [0.3, 0.5, 1.0],   # not equidistant
        [1.3, 1.5],
        [-0.3, 0.0, 0.3],
        [0.0, 0.5, 1.0],
        [0.0, π],
    )
    data = SpheroidSweepData(
        0.638, 1.0,
        zeros(ComplexF64, 3, 2, 3, 3, 2),
        zeros(ComplexF64, 3, 2, 3, 3, 2),
        trues(3, 2, 3),
    )
    path = joinpath(tempdir(), "spheroid_sweep_bad_$(getpid()).h5")
    @test_throws ErrorException write_spheroid_sweep_h5(path, bad_grids, [data])
    rm(path; force = true)
end

@testset "shape mismatch rejected" begin
    grids = SpheroidSweepGrids([0.3, 0.5], [1.3, 1.5], [-0.3, 0.0, 0.3],
                               [0.0, 0.5, 1.0], [0.0, π])
    data_bad_shape = SpheroidSweepData(
        0.638, 1.0,
        zeros(ComplexF64, 2, 2, 3, 3, 99),  # wrong N_ph
        zeros(ComplexF64, 2, 2, 3, 3, 99),
        trues(2, 2, 3),
    )
    path = joinpath(tempdir(), "spheroid_sweep_shape_$(getpid()).h5")
    @test_throws ErrorException write_spheroid_sweep_h5(path, grids, [data_bad_shape])
    rm(path; force = true)
end
