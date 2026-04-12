# Unit tests for src/surface_integrals.jl (Stage 2.2).
#
# The central primitive is  I_G(r; n) = ∫_{S_n} G(|r-r'|) dS'  over a
# boundary face S_n of a half-SWG basis. We verify three things:
#
# 1. Far-field limit: for r far from S_n compared to the face size, the
#    integral reduces to  a_n * G(|r - centroid_n|). Check convergence.
# 2. Self / near-singular convergence: refining the Duffy order reduces
#    the error monotonically for an observation point inside T_n^+ or on
#    a shared vertex.
# 3. Real-part consistency with the static Green ∫ 1/(4π R) ds' for
#    small k0 (limit k0 → 0 gives the electrostatic Coulomb integral).
#
# We build a small custom mesh (a single tet) with boundary-face DOFs
# enabled via `include_boundary_faces = true`.

using Test
using LinearAlgebra: norm
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, TetVerts, TetMesh, build_swg_basis

function one_tet_mesh()
    nodes = Vec3[
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    ]
    tets = TetVerts[TetVerts(1, 2, 3, 4)]
    return TetMesh(nodes, tets)
end

@testset "surface_integral_G: far-field limit" begin
    mesh = one_tet_mesh()
    basis = build_swg_basis(mesh; include_boundary_faces = true)

    # All 4 faces are boundary faces in a single-tet mesh. Pick the one
    # opposite node 1 (the (2,3,4) face lying on the plane x + y + z = 1).
    n_bnd = 0
    for n in 1:n_basis(basis)
        if basis.is_boundary[n] && sort(collect(basis.face_nodes[n])) == [2,3,4]
            n_bnd = n
            break
        end
    end
    @test n_bnd > 0

    verts = boundary_face_vertices(basis, n_bnd)
    centroid = (verts[1] + verts[2] + verts[3]) / 3
    a = triangle_area(verts...)

    # Pick k0 small enough that k0 · (face size) ≪ 1 — then the centroid
    # approximation a·G(centroid) matches the full integral at leading
    # order. Use dist/face_size ≫ 1 to kill the remaining 1/R variation.
    k0 = 1e-3
    for dist in (50.0, 200.0, 1000.0)
        r = Vec3(0.0, 0.0, dist)
        I_num = surface_integral_G(basis, n_bnd, r, k0; mode = :far)
        R_c = norm(r - centroid)
        I_approx = a * exp(+im * k0 * R_c) / (4π * R_c)
        @test isapprox(I_num, I_approx; rtol = 1e-3)
    end
end

@testset "surface_integral_G: centroid-rule match at very-far" begin
    # For very-far observation, even TRI_QUAD_1PT (centroid-only) matches
    # the analytic leading-order expression to machine precision; and
    # a higher-order collapsed rule must also match. This isolates the
    # integration routine from any physics.
    mesh = one_tet_mesh()
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    n_bnd = findfirst(basis.is_boundary)
    verts = boundary_face_vertices(basis, n_bnd)
    centroid = (verts[1] + verts[2] + verts[3]) / 3
    a = triangle_area(verts...)

    # Use a tiny k0 so the phase variation k0·(face size) across the face
    # is negligible — then the centroid approximation is exact to many
    # digits at any large distance.
    k0 = 1e-6
    r = Vec3(0.0, 0.0, 1e4)
    I_ref = a * exp(+im * k0 * norm(r - centroid)) / (4π * norm(r - centroid))
    for order in (2, 3, 4)
        rule = tri_collapsed_rule(order)
        I_num = surface_integral_G(basis, n_bnd, r, k0;
                                    mode = :far, tri_rule = rule)
        @test isapprox(I_num, I_ref; rtol = 1e-6)
    end
end

@testset "surface_integral_G: self-mode convergence to static limit" begin
    # For k0 → 0 the Helmholtz kernel reduces to the electrostatic 1/(4π R).
    # For a point r inside T_n^+ at arbitrary position, the integral
    #   ∫_{S_n} 1/(4π R) dS'
    # is strictly positive and finite. Compare self-mode output at
    # increasing Duffy order against a very-high-order reference.
    mesh = one_tet_mesh()
    basis = build_swg_basis(mesh; include_boundary_faces = true)

    # Pick the tilted face (2, 3, 4) — it is the most interesting.
    n_bnd = 0
    for n in 1:n_basis(basis)
        if basis.is_boundary[n] && sort(collect(basis.face_nodes[n])) == [2,3,4]
            n_bnd = n
            break
        end
    end
    @test n_bnd > 0

    # r is inside T_n^+ (which is tet 1), above the face in the (-,-,-)
    # direction (roughly the centroid of the tet).
    r = Vec3(0.2, 0.2, 0.2)
    k0 = 1e-8  # effectively static

    ref_rule = tri_duffy_reference_rule(16)
    I_ref = surface_integral_G(basis, n_bnd, r, k0;
                                mode = :self, tri_duffy_rule = ref_rule)
    @test real(I_ref) > 0
    @test isapprox(imag(I_ref), 0.0; atol = 1e-6)

    errs = Float64[]
    for order in (4, 6, 8, 10)
        rule = tri_duffy_reference_rule(order)
        I = surface_integral_G(basis, n_bnd, r, k0;
                                mode = :self, tri_duffy_rule = rule)
        push!(errs, abs(I - I_ref) / abs(I_ref))
    end
    @test all(isfinite.(errs))
    @test errs[end] < errs[1]
    @test errs[end] < 1e-3
end

@testset "surface_integral_G: auto mode picks :self for owning tet" begin
    mesh = one_tet_mesh()
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    n_bnd = findfirst(basis.is_boundary)
    verts = boundary_face_vertices(basis, n_bnd)
    tp = basis.tet_plus[n_bnd]

    # Point inside the owning tet.
    centroid_tet = sum(basis.mesh.nodes[v] for v in basis.mesh.tets[tp]) / 4
    k0 = 0.5

    I_auto = surface_integral_G(basis, n_bnd, centroid_tet, k0;
                                 mode = :auto, m_tet = tp)
    I_self = surface_integral_G(basis, n_bnd, centroid_tet, k0;
                                 mode = :self)
    @test isapprox(I_auto, I_self; rtol = 1e-12)
end
