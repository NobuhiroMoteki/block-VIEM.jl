using Test
using Random
using LinearAlgebra
using BlockVIEM
using StaticArrays

include(joinpath(@__DIR__, "mie_reference.jl"))

# Backward scattering amplitude S_bk (S_bak of CAS-v2, `S_bk_OCBS` in the
# HDF5 files). The definition is the corrected Eq. (28) of Moteki and Adachi,
# Opt. Express 32, 36500 (2024), given in the erratum to that paper:
#
#     S_bak = (-S_11 + S_22) / 2   at the backward direction.
#
# The printed Eq. (28) has 1/sqrt(2) in place of 1/2, and this code followed
# the printed equation until 2026-11-19. The tests below fail for a factor
# sqrt(2) in the VIEM value, in the Mie value, or in one of the two.
# block-DDA_Py has the same tests (tests/test_backward_amplitude.py, v0.8.2).

@testset "backward scattering amplitude S_bk" begin
    wl_0 = 0.638                                    # vacuum wavelength [um]
    m_p = 1.5 + 0.0im

    @testset "Mie: Rayleigh limit and sphere identity" begin
        # A dipole radiates equally forward and backward: S_bak -> S(0°) for x -> 0.
        mie = mie_cas_observables(; wl_0 = wl_0, m_m = 1.0, r_p = 0.001, m_p = m_p)
        @test abs(mie.S_bk / mie.S_fw_mean - 1) < 1e-3
        # For a sphere S11(π) = -S22(π), so S_bak = S22(π) = S1(π)/(-ik).
        for r_p in (0.001, 0.1, 0.4), m in (1.5 + 0.0im, 2.9 + 0.1im, 0.17525 + 3.4830im)
            mie = mie_cas_observables(; wl_0 = wl_0, m_m = 1.33, r_p = r_p, m_p = m)
            @test mie.S_bk ≈ mie.S1_bk / (-im * mie.k) rtol = 1e-12
            @test mie.S_bk ≈ -mie.S2_bk / (-im * mie.k) rtol = 1e-9
        end
    end

    mesh = read_msh(joinpath(@__DIR__, "meshes", "sphere.msh"))
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    r_ve = cbrt(3 * total_volume(mesh) / (4π))      # volume-equivalent radius
    wl_x = 2π * r_ve / 0.6                          # size parameter 0.6

    @testset "definition: half the difference of the two components" begin
        rng = MersenneTwister(1)
        D = randn(rng, ComplexF64, n_basis(basis))
        for eu in ((0.0, 0.0, 0.0), (0.2, 0.7, 0.4))
            cas = compute_cas_observables(basis, D; orientation = cas_orientation(eu...),
                                          k0 = 2π / wl_x, eps_p = m_p^2)
            @test cas.S_bk == (-cas.S_bk_theta + cas.S_bk_phi) / 2
        end
    end

    @testset "VIEM small sphere against Mie" begin
        eulers = [(0.0, 0.0, 0.0), (0.3, 0.9, 0.2), (1.1, 2.0, 0.5)]
        res = solve_cas_v2_orientations(basis, eulers; wl_0 = wl_x, m_m = 1.0, m_p = m_p,
                                        tol = 1e-8, maxiter = 200)
        mie = mie_cas_observables(; wl_0 = wl_x, m_m = 1.0, r_p = r_ve, m_p = m_p)
        for r in res
            @test abs(r.S_bk / mie.S_bk - 1) < 0.05
            @test abs(r.S_fw_mean / mie.S_fw_mean - 1) < 0.05
            # Ratio of backward to forward amplitude, independent of the Mie code
            # in its absolute scale.
            @test abs(abs(r.S_bk) / abs(r.S_fw_mean) /
                      (abs(mie.S_bk) / abs(mie.S_fw_mean)) - 1) < 0.05
            # In this regime the two amplitudes are of the same size. A factor
            # sqrt(2) in S_bk would move the ratio outside these bounds.
            @test 0.8 < abs(r.S_bk) / abs(r.S_fw_mean) < 1.1
        end

        # The shifted sweep uses the same definition and gives the same value.
        out, info = solve_cas_v2_orientations_shifted(
            basis, eulers; wl_0 = wl_x, m_m = 1.0, m_p_list = [m_p, 1.7 + 0.0im],
            tol = 1e-8, maxiter = 200)
        @test all(info.converged)
        for i in eachindex(eulers)
            @test out[1, i].S_bk == (-out[1, i].S_bk_theta + out[1, i].S_bk_phi) / 2
            @test out[2, i].S_bk == (-out[2, i].S_bk_theta + out[2, i].S_bk_phi) / 2
            @test abs(out[1, i].S_bk / res[i].S_bk - 1) < 1e-6
        end
    end
end
