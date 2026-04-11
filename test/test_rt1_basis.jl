using Test
using StaticArrays
using LinearAlgebra: norm, dot, cross, det, inv
using BlockVIEM
using BlockVIEM: Vec3, TetVerts, AbstractDivBasis,
                 _rt1_local_info, _ref_face_phys_nodes, _find_sub,
                 REF_FACE_VERTS, LOCAL_FACE_TO_REF_FACE,
                 rt1_ref_evaluate, rt1_ref_divergence

# ---------------------------------------------------------------------------
# Test meshes (reused from test_swg.jl)
# ---------------------------------------------------------------------------
function bipyramid_mesh()
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

function unit_cube_mesh()
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

@testset "RT1Basis" begin
    @testset "type hierarchy" begin
        @test RT1Basis <: AbstractDivBasis
        @test SWGBasis <: AbstractDivBasis
    end

    @testset "construction: bipyramid" begin
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)

        # 1 internal face → 3 face DOFs; 2 tets → 6 interior DOFs; total = 9
        @test basis.n_face_dofs == 3
        @test n_basis(basis) == 9

        # T+ / T- assignment
        for i in 1:3
            @test basis.face_dof_tet_plus[i] == 1
            @test basis.face_dof_tet_minus[i] == 2
        end

        # Jacobian data
        @test length(basis.tet_J) == 2
        @test length(basis.tet_Jinv) == 2
        @test length(basis.tet_detJ) == 2
        # det(J) = 6V_signed for the reference-like first tet
        @test basis.tet_detJ[1] ≈ 6 * mesh.tet_volumes[1]  atol=1e-14
    end

    @testset "construction: unit cube" begin
        mesh = unit_cube_mesh()
        basis = build_rt1_basis(mesh)

        # 6 internal faces → 18 face DOFs; 6 tets → 18 interior DOFs; total = 36
        @test basis.n_face_dofs == 18
        @test n_basis(basis) == 36
    end

    @testset "support_tets" begin
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)

        # Face DOFs: 2 supporting tets
        for i in 1:3
            st = support_tets(basis, i)
            @test length(st) == 2
            @test 1 in st
            @test 2 in st
        end

        # Interior DOFs of tet 1 (DOFs 4,5,6)
        for i in 4:6
            st = support_tets(basis, i)
            @test length(st) == 1
            @test st[1] == 1
        end

        # Interior DOFs of tet 2 (DOFs 7,8,9)
        for i in 7:9
            st = support_tets(basis, i)
            @test length(st) == 1
            @test st[1] == 2
        end
    end

    @testset "_rt1_local_info" begin
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)

        # Face DOFs: sign +1 in T+, -1 in T-
        for i in 1:3
            idx_p, s_p = _rt1_local_info(basis, i, 1)
            @test s_p == 1.0
            @test 1 <= idx_p <= 12

            idx_m, s_m = _rt1_local_info(basis, i, 2)
            @test s_m == -1.0
            @test 1 <= idx_m <= 12
        end

        # Interior DOFs: sign always +1, local idx 13..15
        for i in 4:6
            idx, s = _rt1_local_info(basis, i, 1)
            @test s == 1.0
            @test idx in (13, 14, 15)
            # Not supported in tet 2
            idx2, s2 = _rt1_local_info(basis, i, 2)
            @test idx2 == 0
        end

        # Outside support
        idx, s = _rt1_local_info(basis, 1, 999)
        @test idx == 0
    end

    @testset "Piola: reference tet identity" begin
        # For the standard reference tetrahedron, J = I, det(J) = 1,
        # so the Piola transform is the identity.
        nodes = Vec3[Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1), Vec3(2,0,0)]
        # Two tets sharing face (2,3,4): one is the reference tet, the other is arbitrary
        tets = TetVerts[TetVerts(1,2,3,4), TetVerts(5,2,3,4)]
        mesh = TetMesh(nodes, tets)
        basis = build_rt1_basis(mesh)

        # For interior DOFs of tet 1 (reference tet): Piola = identity
        # Interior DOFs of tet 1: n_face_dofs + 1, +2, +3
        nfd = basis.n_face_dofs
        @test basis.tet_detJ[1] ≈ 1.0  atol=1e-14

        for sub in 1:3
            dof = nfd + sub
            local_idx = 12 + sub
            # Evaluate at a few test points inside the reference tet
            for (ξ, η, ζ) in [(0.1, 0.2, 0.3), (0.25, 0.25, 0.25), (0.05, 0.05, 0.05)]
                r = Vec3(ξ, η, ζ)
                val = evaluate(basis, dof, r, 1)
                ref_val = rt1_ref_evaluate(local_idx, ξ, η, ζ)
                @test val ≈ ref_val  atol=1e-13

                div_val = divergence(basis, dof, r, 1)
                ref_div = rt1_ref_divergence(local_idx, ξ, η, ζ)
                @test div_val ≈ ref_div  atol=1e-12
            end
        end
    end

    @testset "Piola: scaled tet" begin
        # Tet with J = 2I → det(J) = 8. Check Piola scaling.
        nodes = Vec3[Vec3(0,0,0), Vec3(2,0,0), Vec3(0,2,0), Vec3(0,0,2), Vec3(3,0,0)]
        tets = TetVerts[TetVerts(1,2,3,4), TetVerts(5,2,3,4)]
        mesh = TetMesh(nodes, tets)
        basis = build_rt1_basis(mesh)

        @test basis.tet_detJ[1] ≈ 8.0  atol=1e-14

        nfd = basis.n_face_dofs
        # Interior DOF: evaluate at r = (0.2, 0.4, 0.6) → ξ = (0.1, 0.2, 0.3)
        r = Vec3(0.2, 0.4, 0.6)
        for sub in 1:3
            dof = nfd + sub
            local_idx = 12 + sub
            val = evaluate(basis, dof, r, 1)
            ξ = Vec3(0.1, 0.2, 0.3)
            ref_val = rt1_ref_evaluate(local_idx, ξ[1], ξ[2], ξ[3])
            # φ^phys = (1/det J) J φ^ref = (1/8) * 2I * φ^ref = (1/4) φ^ref
            @test val ≈ (1.0/4.0) * ref_val  atol=1e-13

            div_val = divergence(basis, dof, r, 1)
            ref_div = rt1_ref_divergence(local_idx, ξ[1], ξ[2], ξ[3])
            # div^phys = (1/det J) div^ref = (1/8) div^ref
            @test div_val ≈ ref_div / 8.0  atol=1e-12
        end
    end

    @testset "H(div) continuity: face DOF normal component" begin
        # On the shared face, the normal component of a face DOF must be
        # continuous (same value when evaluated from T+ and T-).
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)

        # The shared face is (node2, node3, node4) = (1,0,0), (0,1,0), (0,0,1)
        # Outward normal from T1 toward T2: (1,1,1)/sqrt(3)
        n_hat = Vec3(1, 1, 1) / sqrt(3)

        # Test at multiple points on the shared face
        # Points on the face: r = α v2 + β v3 + γ v4 with α+β+γ=1, α,β,γ ≥ 0
        v2, v3, v4 = mesh.nodes[2], mesh.nodes[3], mesh.nodes[4]
        face_pts = [
            (1/3) * v2 + (1/3) * v3 + (1/3) * v4,  # centroid
            0.5 * v2 + 0.25 * v3 + 0.25 * v4,
            0.1 * v2 + 0.1 * v3 + 0.8 * v4,
        ]

        for i in 1:basis.n_face_dofs
            for r in face_pts
                val_plus = evaluate(basis, i, r, 1)    # from T+
                val_minus = evaluate(basis, i, r, 2)   # from T-
                # Normal component must match
                normal_plus = dot(val_plus, n_hat)
                normal_minus = dot(val_minus, n_hat)
                @test normal_plus ≈ normal_minus  atol=1e-12
            end
        end
    end

    @testset "interior DOFs vanish on faces" begin
        # Interior (bubble) DOFs must have zero normal component on ALL faces.
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)
        nfd = basis.n_face_dofs

        # Test interior DOFs of tet 1 (DOFs nfd+1..nfd+3)
        tet = mesh.tets[1]
        verts = [mesh.nodes[tet[k]] for k in 1:4]

        for lf in 1:4
            # Face nodes
            fi = BlockVIEM.TET_LOCAL_FACES[lf]
            fv = [verts[fi[1]], verts[fi[2]], verts[fi[3]]]
            # Outward normal (unnormalized)
            n_face = cross(fv[2] - fv[1], fv[3] - fv[1])
            n_face = n_face / norm(n_face)
            # Point on face
            r_face = (fv[1] + fv[2] + fv[3]) / 3

            for sub in 1:3
                dof = nfd + sub
                val = evaluate(basis, dof, r_face, 1)
                @test abs(dot(val, n_face)) < 1e-12
            end
        end
    end

    @testset "divergence theorem" begin
        # ∫_K div(φ) dV = ∫_∂K φ·n dA  for each basis function on each tet.
        # Use high-order quadrature for volume (degree 7 rule) and surface.
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)
        vol_rule = TET_QUAD_125PT

        for tet_idx in 1:2
            tet = mesh.tets[tet_idx]
            verts = ntuple(k -> mesh.nodes[tet[k]], Val(4))
            V = tet_volume(verts...)

            for dof in 1:n_basis(basis)
                # Volume integral of divergence
                vol_int = 0.0
                for i in 1:vol_rule.n
                    r = bary_to_point(vol_rule.bary[i], verts)
                    vol_int += vol_rule.weights[i] * V * divergence(basis, dof, r, tet_idx)
                end

                # Surface integral of normal flux
                surf_int = 0.0
                for lf in 1:4
                    fi = BlockVIEM.TET_LOCAL_FACES[lf]
                    fv = (verts[fi[1]], verts[fi[2]], verts[fi[3]])
                    # Outward normal × area (cross product gives 2× triangle area with normal)
                    nA = cross(fv[2] - fv[1], fv[3] - fv[1]) / 2
                    # Ensure outward: dot with (face_centroid - opposite_vertex) > 0
                    opp_vert = verts[lf]
                    fc = (fv[1] + fv[2] + fv[3]) / 3
                    if dot(nA, fc - opp_vert) < 0
                        nA = -nA
                    end

                    # 7-point triangle quadrature (degree 5)
                    tri_pts = [
                        (1/3, 1/3, 1/3),
                        (0.059715871789770, 0.470142064105115, 0.470142064105115),
                        (0.470142064105115, 0.059715871789770, 0.470142064105115),
                        (0.470142064105115, 0.470142064105115, 0.059715871789770),
                        (0.797426985353087, 0.101286507323456, 0.101286507323456),
                        (0.101286507323456, 0.797426985353087, 0.101286507323456),
                        (0.101286507323456, 0.101286507323456, 0.797426985353087),
                    ]
                    tri_wts = [0.225,
                               0.132394152788506, 0.132394152788506, 0.132394152788506,
                               0.125939180544827, 0.125939180544827, 0.125939180544827]
                    A = norm(cross(fv[2]-fv[1], fv[3]-fv[1])) / 2

                    for (idx, (b1, b2, b3)) in enumerate(tri_pts)
                        r_tri = b1 * fv[1] + b2 * fv[2] + b3 * fv[3]
                        val = evaluate(basis, dof, r_tri, tet_idx)
                        n_unit = nA / norm(nA)
                        surf_int += tri_wts[idx] * A * dot(val, n_unit)
                    end
                end

                @test vol_int ≈ surf_int  atol=1e-10
            end
        end
    end

    @testset "evaluate/divergence outside support return zero" begin
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)
        r = Vec3(0.1, 0.1, 0.1)

        # Interior DOF of tet 1, evaluated in tet 2 → zero
        @test evaluate(basis, basis.n_face_dofs + 1, r, 2) == zero(Vec3)
        @test divergence(basis, basis.n_face_dofs + 1, r, 2) == 0.0

        # Any DOF evaluated in a non-existent tet → zero
        @test evaluate(basis, 1, r, 999) == zero(Vec3)
        @test divergence(basis, 1, r, 999) == 0.0
    end

    @testset "type stability" begin
        mesh = bipyramid_mesh()
        basis = build_rt1_basis(mesh)
        r = Vec3(0.1, 0.1, 0.1)
        @inferred evaluate(basis, 1, r, 1)
        @inferred divergence(basis, 1, r, 1)
        @inferred n_basis(basis)
    end

    @testset "SWGBasis 4-arg divergence" begin
        # Verify the uniform calling convention works for SWG
        mesh = bipyramid_mesh()
        basis = build_swg_basis(mesh)
        r = Vec3(0.1, 0.1, 0.1)
        @test divergence(basis, 1, r, 1) == divergence(basis, 1, 1)
        @test divergence(basis, 1, r, 2) == divergence(basis, 1, 2)
        @inferred divergence(basis, 1, r, 1)
    end
end
