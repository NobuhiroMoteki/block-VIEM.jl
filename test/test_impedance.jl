using Test
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, TetVerts

# ----------------------------------------------------------------------------
# Helper meshes (mirrored from test_swg.jl).
# ----------------------------------------------------------------------------
function bipyramid_mesh_z()
    nodes = Vec3[
        Vec3(0, 0, 0), Vec3(1, 0, 0),
        Vec3(0, 1, 0), Vec3(0, 0, 1),
        Vec3(1, 1, 1),
    ]
    tets = TetVerts[
        TetVerts(1, 2, 3, 4),
        TetVerts(5, 2, 3, 4),
    ]
    return TetMesh(nodes, tets)
end

function unit_cube_mesh_z()
    nodes = Vec3[
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
        Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1),
    ]
    tets = TetVerts[
        TetVerts(1, 2, 3, 7),
        TetVerts(1, 3, 4, 7),
        TetVerts(1, 5, 6, 7),
        TetVerts(1, 6, 2, 7),
        TetVerts(1, 4, 8, 7),
        TetVerts(1, 8, 5, 7),
    ]
    return TetMesh(nodes, tets)
end

@testset "impedance_element" begin
    @testset "bipyramid: analytic mass-only Z_11 (κ = 0)" begin
        # With eps_p == eps_bg == 1 the contrast vanishes, so Z_mn reduces
        # to the geometric inner product (1/eps_bg) ∫ f_m·f_n dV. For the
        # canonical bipyramid the single internal face has
        #
        #   ∫_T1 |f|² dV = 3/20      (T1 = reference unit tet, V = 1/6)
        #   ∫_T2 |f|² dV = 9/40      (T2 = (1,1,1)-pyramid, V = 1/3)
        #
        # giving M_11 = 3/20 + 9/40 = 3/8.  See technical_note.md §1 and the
        # hand calculation in the Phase 2 review.
        mesh = bipyramid_mesh_z()
        basis = build_swg_basis(mesh)
        Z = impedance_element(basis, 1, 1; k0 = 0.0, eps_p = 1.0, eps_bg = 1.0)
        @test isapprox(real(Z), 3 / 8; atol = 1e-12)
        @test isapprox(imag(Z), 0.0; atol = 1e-14)
    end

    @testset "type stability and return type" begin
        mesh = bipyramid_mesh_z()
        basis = build_swg_basis(mesh)
        Z = @inferred impedance_element(basis, 1, 1; k0 = 1.0, eps_p = 2.0)
        @test Z isa ComplexF64
        Zc = @inferred impedance_element(basis, 1, 1;
                                         k0 = 1.0 + 0.1im,
                                         eps_p = 2.0 + 0.05im)
        @test Zc isa ComplexF64
    end

    @testset "real k0=0 with real ε ⇒ imag(Z) ≈ 0" begin
        mesh = bipyramid_mesh_z()
        basis = build_swg_basis(mesh)
        Z = impedance_element(basis, 1, 1; k0 = 0.0, eps_p = 2.5, eps_bg = 1.0)
        @test isapprox(imag(Z), 0.0; atol = 1e-14)
    end

    @testset "k0 → 0 smooth limit" begin
        mesh = bipyramid_mesh_z()
        basis = build_swg_basis(mesh)
        Z0 = impedance_element(basis, 1, 1; k0 = 0.0, eps_p = 2.0)
        Zε = impedance_element(basis, 1, 1; k0 = 1e-6, eps_p = 2.0)
        @test isapprox(real(Zε), real(Z0); rtol = 1e-8)
        # Imag part of Zε should be O(k0) — small but not necessarily zero.
        @test abs(imag(Zε)) < 1e-3
    end

    @testset "cube mesh: κ=0 reciprocity (mass-only, exact)" begin
        # With eps_p == eps_bg the radiation kernel is multiplied by κ = 0
        # and Z_mn = (1/eps_bg) M_mn. The mass term uses the 5-point degree-3
        # tet rule which is exact for the degree-2 product f_m·f_n, so
        # reciprocity must hold to floating-point precision.
        mesh = unit_cube_mesh_z()
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        for m in 1:N, n in 1:N
            Zmn = impedance_element(basis, m, n; k0 = 0.0,
                                    eps_p = 1.0, eps_bg = 1.0)
            Znm = impedance_element(basis, n, m; k0 = 0.0,
                                    eps_p = 1.0, eps_bg = 1.0)
            @test isapprox(Zmn, Znm; atol = 1e-13)
        end
    end

    @testset "cube mesh: κ≠0, k0>0 reciprocity (quadrature accuracy)" begin
        # The asymmetric outer-Gauss / inner-Duffy scheme introduces a small
        # quadrature-error gap between Z_mn and Z_nm. With the default 5-point
        # and order-5 rules the worst-case relative gap is a few × 1e-6;
        # tightening would require higher-order rules.
        mesh = unit_cube_mesh_z()
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        k0 = 0.5
        eps_p = 2.0 + 0.1im
        max_diff = 0.0
        for m in 1:N, n in (m + 1):N
            Zmn = impedance_element(basis, m, n; k0 = k0, eps_p = eps_p)
            Znm = impedance_element(basis, n, m; k0 = k0, eps_p = eps_p)
            d = abs(Zmn - Znm) / max(abs(Zmn), abs(Znm), 1.0)
            max_diff = max(max_diff, d)
        end
        @test max_diff < 1e-5
    end

    @testset "self-term convergence under Duffy refinement" begin
        mesh = bipyramid_mesh_z()
        basis = build_swg_basis(mesh)
        Zs = ComplexF64[]
        for n in (3, 5, 7, 9)
            rule = duffy_reference_rule(n)
            Z = impedance_element(basis, 1, 1;
                                  k0 = 0.5, eps_p = 2.0,
                                  duffy_rule = rule)
            push!(Zs, Z)
        end
        @test all(isfinite, Zs)
        # Successive differences shrink (Cauchy convergence).
        diffs = abs.(diff(Zs))
        @test diffs[end] < diffs[1]
        # Order-9 vs order-7 gap is ~3e-4 of |Z| for the bipyramid singular
        # vertex; loosen the threshold to leave headroom.
        @test diffs[end] / abs(Zs[end]) < 1e-3
    end

    @testset "diagonal positivity in the static limit" begin
        # For k0 = 0 and real eps_p > 1, Z_mm collects a positive mass term
        # and a positive electrostatic-energy contribution from -κ * (-div²)
        # ∫∫ G dV'dV. Hence real(Z_mm) > 0 for every basis function.
        mesh = unit_cube_mesh_z()
        basis = build_swg_basis(mesh)
        for m in 1:n_basis(basis)
            Z = impedance_element(basis, m, m;
                                  k0 = 0.0, eps_p = 2.0, eps_bg = 1.0)
            @test real(Z) > 0
            @test isapprox(imag(Z), 0.0; atol = 1e-13)
        end
    end
end
