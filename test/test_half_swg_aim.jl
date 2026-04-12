# Stage 2.7: AIM operator with half-SWG enabled must match the dense
# reference Z × x for all-boundary and mixed-boundary basis sets.
#
# We compare `aim_mvp(op, x)` vs `Z * x` where `Z` is the dense matrix
# from `assemble_impedance_matrix(basis; include_boundary_faces = true)`.

using Test
using LinearAlgebra: norm
using SparseArrays: nnz
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, TetVerts, TetMesh
import Gmsh: gmsh

function small_sphere(lc)
    path = joinpath(tempdir(), "sph_aim_halfswg_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sph_aim_$(lc)")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path); gmsh.finalize()
    return path
end

@testset "Half-SWG AIM MVP == dense Z*x" begin
    path = small_sphere(0.6)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N = n_basis(basis)
    @test N > 0
    @test any(basis.is_boundary)
    @test !all(basis.is_boundary)        # some interior DOFs expected too

    k0 = 0.4
    eps_p = 2.25 + 0.01im
    eps_bg = 1.0

    # Dense reference
    Z = assemble_impedance_matrix(basis;
                                   k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                                   symmetrize = true)

    # AIM operator with half-SWG correction
    h_bar = 0.0; c = 0
    for tet in mesh.tets, (a, b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        h_bar += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
    end
    h_bar /= c
    pitch = 0.5 * h_bar
    op = build_aim_operator(basis;
                             k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                             pitch = pitch, padding = 4)

    # Random complex test vector
    rng = 1:N
    x = ComplexF64.(rng) + im * ComplexF64.(reverse(rng))
    y_dense = Z * x
    y_aim = aim_mvp(op, x)

    rel = norm(y_dense - y_aim) / norm(y_dense)
    @info "AIM vs dense relative MVP error (half-SWG)" rel
    @test rel < 5e-2   # AIM's far-field approx + precorrection tolerance
end

@testset "Half-SWG correction is zero for interior-only basis" begin
    path = small_sphere(0.6)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = false)
    extra = assemble_half_swg_correction(basis;
                                          k0 = 0.4, eps_p = 2.25 + 0.01im)
    @test size(extra) == (n_basis(basis), n_basis(basis))
    @test nnz(extra) == 0
end
