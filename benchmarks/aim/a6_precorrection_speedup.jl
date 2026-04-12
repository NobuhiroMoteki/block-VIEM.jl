# Phase A.6: measure the precorrection build speedup from
# fft_convolve_plan! (pre-allocated FFT workspace + plans).
#
# Before optimization: ~40 s at lc=0.25 (N=2600) from Phase A.4.
# Target: 3–10× speedup by reusing FFT plan and buffer.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_a6_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("a6_$(lc)")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1)
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

const K0 = 0.316
const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0

flush(stdout)
println("=" ^ 78); flush(stdout)
println("  Phase A.6: precorrection build speedup via planned FFT"); flush(stdout)
println("=" ^ 78); flush(stdout)
@printf("%6s %5s %6s | %10s %10s %10s | %10s\n",
        "lc", "Ntet", "N", "t_grid+proj", "t_Ghat", "t_precorr", "t_total(s)")
println("-"^74); flush(stdout)

for lc in [0.7, 0.5, 0.35, 0.25]
    mesh = read_msh(sphere_mesh(lc))
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    h = mean_h(mesh)
    pitch = 0.5 * h

    # Time the full build_aim_operator and break down
    t_total = @elapsed begin
        t_setup = @elapsed begin
            grid = BlockVIEM.aim_grid(mesh; pitch=pitch, padding=4)
            proj = BlockVIEM.build_aim_projection(basis, grid;
                                                    poly_order=2, stencil=3)
        end
        t_Ghat = @elapsed begin
            G_toep = BlockVIEM.build_green_toeplitz(grid, ComplexF64(K0))
            G_hat  = BlockVIEM.precompute_green_fft(G_toep)
        end
        mass = BlockVIEM.assemble_mass_matrix(basis)
        t_precorr = @elapsed precorr = BlockVIEM.assemble_precorrection(
            basis, proj, G_hat; k0=K0)
    end

    @printf("%6.3f %5d %6d | %10.2f %10.2f %10.2f | %10.2f\n",
            lc, length(mesh.tets), N, t_setup, t_Ghat, t_precorr, t_total)
    flush(stdout)
end
