# Phase A.4b: tune near_radius for AIM scalability + accuracy.
#
# Goal: find the smallest near_radius that keeps AIM vs dense error
# below a target (e.g., 0.1%) on a realistic mesh. This controls the
# memory / time cost of the precorrection matrix and is the dominant
# factor in AIM scalability.

using LinearAlgebra: norm
using Printf
using Random
using SparseArrays: nnz
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh
import Base: summarysize

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_near_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sphNR$(lc)")
    gmsh.model.occ.addSphere(0.,0.,0.,1.0,1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3,[1],1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path)
    gmsh.finalize()
    return path
end

function mean_h(mesh)
    s = 0.0; c = 0
    for tet in mesh.tets, (a,b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
    end
    s/c
end

const K0     = 0.316
const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0

flush(stdout)
println("=" ^ 88); flush(stdout)
println("  Phase A.4b: near_radius tuning (Sphere meshes, k0=$K0, eps=$EPS_P)"); flush(stdout)
println("=" ^ 88); flush(stdout)

for lc in (0.35, 0.25)
    println("\n--- lc = $lc ---"); flush(stdout)
    mesh = read_msh(sphere_mesh(lc))
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    h = mean_h(mesh)
    @printf("  N=%d  h̄=%.4f\n", N, h); flush(stdout)

    # Dense reference
    print("  assembling dense Z ..."); flush(stdout)
    t_Z = @elapsed Z = assemble_impedance_matrix(basis; k0=K0, eps_p=EPS_P,
                                                  eps_bg=EPS_BG, symmetrize=false)
    @printf(" %.1f s\n", t_Z); flush(stdout)
    Random.seed!(42)
    x = randn(ComplexF64, N); x ./= norm(x)
    y_dense = Z * x

    @printf("%10s | %10s | %10s %8s | %8s | %12s\n",
            "r_near/h̄", "#near/N", "rel err", "err %", "t_build", "Pc mem(MB)")
    println("  " * "-"^72); flush(stdout)

    for ratio in (0.5, 1.0, 1.5, 2.0, 3.0)
        r_near = ratio * h
        pitch = 0.5 * h

        t_build = @elapsed op = build_aim_operator(basis;
            k0=K0, eps_p=EPS_P, eps_bg=EPS_BG,
            pitch=pitch, padding=4,
            poly_order=2, stencil_size=3,
            near_radius=r_near)

        y_aim = aim_mvp(op, x)
        rel = norm(y_aim - y_dense) / norm(y_dense)

        n_near = nnz(op.precorrection)
        Pc_mb = Base.summarysize(op.precorrection) / 1024^2

        @printf("%10.2f | %10.2f | %10.2e %7.2f%% | %7.1fs | %10.2f\n",
                ratio, n_near/N, rel, 100*rel, t_build, Pc_mb)
        flush(stdout)
    end
end
