# Low-contrast dielectric sphere benchmark for block-VIEM.jl vs block-DDA_Py.
#
# Reproduces the low-contrast entry of the "When to use block-VIEM.jl vs
# block-DDA_Py" comparison table in README.md:
#   r = 1 μm, m_p = 1.5 + 0.01i, wl_0 = 10 μm, x = k0·r ≈ 0.628,
#   single orientation.
#
# For each mesh refinement the script reports
#   N, C_abs error vs Mie, t_setup (AIM operator build), t_solve
#   (block-BiCGSTAB, L=1), and `Base.summarysize(AIMOperator)` as the
#   in-memory footprint of the assembled operator.
#
# Run single-threaded (apples-to-apples with the README caveat) as
#     JULIA_NUM_THREADS=1 julia --project=. -t 1 \
#         benchmarks/cas_v2/low_contrast_sphere.jl
#
# Memory accounting matches the Phase A doublet table in README §
# "Phase A memory scaling": `summarysize(op)` covers every sparse
# matrix, the FFT Green kernel, and the mass matrix held by the AIM
# operator.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(radius::Float64, lc::Float64)
    path = joinpath(tempdir(), "low_contrast_sphere_r$(radius)_lc$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dielec_sphere")
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

# ── Physical parameters ──────────────────────────────────────────────────────
const WL_0   = 10.0
const M_M    = 1.0
const M_P    = 1.5 + 0.01im
const RADIUS = 1.0
const k0_bg  = ComplexF64(2π * M_M / WL_0)
const eps_p  = M_P^2
const eps_bg = M_M^2

println("=" ^ 96)
println("Low-contrast dielectric sphere (block-VIEM.jl vs DDA comparison)")
println("=" ^ 96)
@printf("  λ_0 = %.2f μm,  m_p = %.3f + %.3f i,  m_m = %.3f,  r = %.3f μm,  x = %.3f\n",
        WL_0, real(M_P), imag(M_P), M_M, RADIUS, real(k0_bg) * RADIUS)

mie = mie_cross_sections(; wl_0 = WL_0, m_m = M_M, r_p = RADIUS, m_p = M_P)
@printf("\n  Mie reference (r = %.3f μm):  C_abs = %.6e μm²  (Q_abs = %.5f)\n",
        RADIUS, mie.C_abs, mie.Q_abs)

# Target lc values to hit roughly N ≈ 600, 2000, 7900 half-SWG DOFs.
const LC_LIST = (0.125, 0.085, 0.055)
const ORI     = (0.0, 0.0, 0.0)
const DUFFY   = duffy_reference_rule(5)
const TOL     = 1e-8
const MAXIT   = 400
const PADDING = 4

println("\n" * "-" ^ 96)
@printf("  %-7s %-6s %-11s %-9s %-9s %-10s\n",
        "lc", "N", "C_abs err", "t_setup", "t_solve", "memory")
println("  " * "-" ^ 92)

for lc in LC_LIST
    path  = sphere_mesh(RADIUS, lc)
    mesh  = read_msh(path)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N     = n_basis(basis)

    k_hat = Vec3(0, 0, 1)
    E0    = SVector{3,ComplexF64}(1/√2, 1im/√2, 0.0)

    GC.gc()

    # ── t_setup: build AIM operator + project RHS ────────────────────────────
    pitch = 0.5 * BlockVIEM.mean_edge_length(mesh)
    t_setup = @elapsed begin
        op = build_aim_operator(basis; k0 = k0_bg, eps_p = eps_p,
                                eps_bg = eps_bg,
                                pitch = pitch, padding = PADDING,
                                outer_rule = BlockVIEM.TET_QUAD_5PT,
                                duffy_rule = DUFFY)
        b = project_plane_wave(basis; k_hat = k_hat, E0 = E0, k_bg = k0_bg)
    end

    mem_bytes = Base.summarysize(op)

    # ── t_solve: single-RHS block-BiCGSTAB ───────────────────────────────────
    B = reshape(b, :, 1)
    A = BlockVIEM._AIMLinOp(op, N)
    t_solve = @elapsed begin
        res = block_bicgstab(A, B; tol = TOL, maxiter = MAXIT)
    end
    D = res.X[:, 1]

    # ── Accuracy vs Mie at actual mesh volume-equivalent radius ──────────────
    V_mesh = total_volume(mesh)
    r_ve   = (3V_mesh / (4π))^(1/3)
    mie_e  = mie_cross_sections(; wl_0 = WL_0, m_m = M_M, r_p = r_ve, m_p = M_P)
    scat   = compute_scattering(basis, D;
                                k_hat = k_hat, E0 = E0,
                                k0 = k0_bg, eps_p = eps_p, eps_bg = eps_bg,
                                csca_method = :farfield)
    err_Cabs = abs(scat.C_abs - mie_e.C_abs) / mie_e.C_abs

    # Report memory in MB (decimal, matching README convention)
    mem_mb = mem_bytes / 1024 / 1024
    mem_str = mem_mb < 10 ? @sprintf("%.1f MB", mem_mb) :
                            @sprintf("%.0f MB",  mem_mb)
    @printf("  %-7.4f %-6d %-11.3e %7.1fs %8.1fs %10s\n",
            lc, N, err_Cabs, t_setup, t_solve, mem_str)
    flush(stdout)
end

println("\nDone.")
