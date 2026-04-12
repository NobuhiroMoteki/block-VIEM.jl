# Phase A.4: AIM scalability benchmark.
#
# Builds the AIM operator at several mesh resolutions and reports:
#   - setup time (projection + Green FFT + precorrection)
#   - MVP time (steady-state, averaged over several applications)
#   - memory footprint of the sparse mass, sparse precorrection,
#     and FFT kernel tables
#   - cost per iteration if used inside a Krylov solver
#
# Goal: characterize the largest sphere mesh we can reasonably afford
# to run inside a BiCGSTAB loop for CAS-v2 sweeps (Phase B).

using LinearAlgebra: norm
using Printf
using StaticArrays
using SparseArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh
import Base: summarysize

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

function generate_sphere_mesh(radius::Float64, lc::Float64)
    path = joinpath(tempdir(), "sphere_aim_scale_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere_scale_$(lc)")
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

function mean_edge_length(mesh)
    total = 0.0
    count = 0
    edges = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
    for tet in mesh.tets
        for (a, b) in edges
            total += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]])
            count += 1
        end
    end
    return total / count
end

const K0     = 0.316
const EPS_P  = 10.0 + 1.0im
const EPS_BG = 1.0
const PITCH_RATIO = 0.5     # tuned in A.2; re-evaluate if A.2 suggests otherwise
const PADDING = 4
const POLY = 2
const STENCIL = 3

println("=" ^ 78); flush(stdout)
println("  Phase A.4: AIM scalability (RT0 / SWG)"); flush(stdout)
println("  k0=$K0, eps_p=$EPS_P, pitch=$PITCH_RATIO·h̄, pad=$PADDING, P=$POLY, M=$STENCIL"); flush(stdout)
println("=" ^ 78); flush(stdout)
@printf("%7s %6s %6s %8s | %8s %8s %8s | %8s %8s %8s | %8s\n",
        "lc", "Ntet", "N", "h̄",
        "M(MB)", "Pc(MB)", "Gĥ(MB)",
        "t_grid", "t_proj", "t_prec",
        "t_mvp(s)")
println("-"^92); flush(stdout)

for lc in [0.7, 0.5, 0.35, 0.25]
    path = generate_sphere_mesh(1.0, lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    h_mean = mean_edge_length(mesh)
    pitch = PITCH_RATIO * h_mean

    # Time the individual stages of operator construction.
    t_grid = @elapsed grid = BlockVIEM.aim_grid(mesh; pitch = pitch, padding = PADDING)
    t_proj = @elapsed proj = BlockVIEM.build_aim_projection(
        basis, grid; poly_order = POLY, stencil = STENCIL)
    t_toep = @elapsed begin
        G_toep = BlockVIEM.build_green_toeplitz(grid, ComplexF64(K0))
        G_hat  = BlockVIEM.precompute_green_fft(G_toep)
    end
    t_mass = @elapsed mass = BlockVIEM.assemble_mass_matrix(basis)
    t_prec = @elapsed precorr = BlockVIEM.assemble_precorrection(
        basis, proj, G_hat; k0 = K0)

    eps_p_c = ComplexF64(EPS_P)
    kappa = (eps_p_c - ComplexF64(EPS_BG)) / eps_p_c
    inv_eps = 1 / eps_p_c
    op = BlockVIEM.AIMOperator(proj, G_hat, mass, precorr,
                               ComplexF64(K0), eps_p_c,
                               ComplexF64(EPS_BG), kappa, inv_eps)

    # Memory footprints (MB). summarysize gives total bytes including
    # pointers and buffers.
    mass_MB  = summarysize(mass) / 1024^2
    prec_MB  = summarysize(precorr) / 1024^2
    Ghat_MB  = summarysize(G_hat) / 1024^2

    # Time the MVP (steady state, amortize JIT)
    x = randn(ComplexF64, N)
    aim_mvp(op, x)    # warm up
    n_trials = 5
    t_mvp = @elapsed for _ in 1:n_trials
        aim_mvp(op, x)
    end
    t_mvp /= n_trials

    @printf("%7.3f %6d %6d %8.4f | %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f | %8.4f\n",
            lc, length(mesh.tets), N, h_mean,
            mass_MB, prec_MB, Ghat_MB,
            t_grid + t_proj + t_toep + t_mass, t_proj, t_prec,
            t_mvp)
    flush(stdout)
end
println()
