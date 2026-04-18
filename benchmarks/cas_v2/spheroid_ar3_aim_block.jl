# AR=3 oblate spheroid CAS-v2 benchmark through the AIM block-Krylov path.
#
# Mirrors `benchmarks/cas_v2/spheroid_ar3.jl` but solves all orientations
# at once with `solve_cas_v2_orientations(..., method = :aim_bicgstab)`.
# Verifies:
#   1. the analytical  S_θ(α) = A + B·exp(+2jα)  symmetry (C∞ about c),
#   2. agreement with the dense LU reference to within AIM's intrinsic
#      MVP approximation error.
#
# Also compares block-BiCGSTAB and block-GMRES end-to-end.
#
# Run with:
#     julia --project=. benchmarks/cas_v2/spheroid_ar3_aim_block.jl

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function spheroid_mesh(a::Float64, c::Float64, lc::Float64)
    path = joinpath(tempdir(), "spheroid_aim_a$(a)_c$(c)_lc$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("spheroid_aim")
        sph_tag = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, sph_tag)], 0.0, 0.0, 0.0, a, a, c)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [sph_tag], 1)
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
const WL_0 = 0.638
const M_M  = 1.0
const M_P  = 1.5 + 0.0im
const D_VE = 0.40
const AR   = 3.0

const r_ve = D_VE / 2
const b_eq = r_ve * AR^( 1/3)
const c_ax = r_ve * AR^(-2/3)

const eps_p  = M_P^2
const eps_bg = M_M^2
const k0     = 2π * M_M / WL_0

println("=" ^ 70)
println("CAS-v2 AR=$AR oblate spheroid via AIM block Krylov")
println("=" ^ 70)
@printf("  D_ve=%.3f μm, (a,b,c)=(%.4f, %.4f, %.4f) μm, x=k0·r_ve=%.4f\n",
        D_VE, b_eq, b_eq, c_ax, k0 * r_ve)

const LC = 0.06
mesh_path = spheroid_mesh(b_eq, c_ax, LC)
mesh = read_msh(mesh_path)
basis = build_swg_basis(mesh; include_boundary_faces = true)
N = n_basis(basis)
@printf("  Mesh: %d tets, N=%d SWG DOFs\n", n_tets(mesh), N)

# mean edge length → AIM pitch
function _mean_edge_length(mesh)
    s = 0.0; c = 0
    for tet in mesh.tets, (a, b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
    end
    return s / c
end
const H_BAR = _mean_edge_length(mesh)
const PITCH = 0.5 * H_BAR
@printf("  h̄=%.4f μm, AIM pitch=%.4f μm\n", H_BAR, PITCH)

# ── Orientations ─────────────────────────────────────────────────────────────
const BETA = π/4
const ALPHAS = (0.0, π/8, π/4, 3π/8, π/2)
euler_list = [(α, BETA, 0.0) for α in ALPHAS]
L = length(euler_list)
@printf("  L=%d orientations at β=π/4\n", L)

# ── Dense LU reference ───────────────────────────────────────────────────────
println("\n[1/3] Dense LU reference")
t_dense = @elapsed ref = solve_cas_v2_orientations(
    basis, euler_list;
    k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
    duffy_rule = duffy_reference_rule(7))
@printf("  elapsed: %.2f s\n", t_dense)

# ── AIM + Block BiCGSTAB ─────────────────────────────────────────────────────
println("\n[2/3] AIM + Block BiCGSTAB (multi-orientation)")
t_bcg = @elapsed out_bcg = solve_cas_v2_orientations(
    basis, euler_list;
    k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
    duffy_rule = duffy_reference_rule(7),
    method = :aim_bicgstab, pitch = PITCH, padding = 4,
    tol = 1e-8, maxiter = 400, verbose = true)
@printf("  elapsed: %.2f s\n", t_bcg)

# ── AIM + Block GMRES ────────────────────────────────────────────────────────
println("\n[3/3] AIM + Block GMRES (multi-orientation)")
t_gmr = @elapsed out_gmr = solve_cas_v2_orientations(
    basis, euler_list;
    k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
    duffy_rule = duffy_reference_rule(7),
    method = :aim_gmres, pitch = PITCH, padding = 4,
    tol = 1e-8, maxiter = 400, verbose = true)
@printf("  elapsed: %.2f s\n", t_gmr)

# ── Agreement checks ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("Comparison")
println("=" ^ 70)
println("  α           S_fw_mean(dense)           S_fw_mean(AIM bicgstab)    err")
for k in 1:L
    ref_S = ref[k].S_fw_mean
    bcg_S = out_bcg[k].S_fw_mean
    err = abs(ref_S - bcg_S) / abs(ref_S)
    @printf("  α=%-6.4f  %+8.5f%+8.5fim   %+8.5f%+8.5fim   %.2e\n",
            ALPHAS[k], real(ref_S), imag(ref_S),
            real(bcg_S), imag(bcg_S), err)
end

function _max_bcg_vs_gmr(out_bcg, out_gmr, L)
    m = 0.0
    for k in 1:L
        d = abs(out_bcg[k].S_fw_mean - out_gmr[k].S_fw_mean) / abs(out_bcg[k].S_fw_mean)
        m = max(m, d)
    end
    return m
end
@printf("\n  max |S_fw_mean(bicgstab) − S_fw_mean(gmres)| / |S_fw_mean| = %.2e\n",
        _max_bcg_vs_gmr(out_bcg, out_gmr, L))

# ── Symmetry verification on the AIM block-BiCGSTAB solution ─────────────────
println("\n  Spheroid symmetry: S_θ(α) ?= A + B exp(+2jα)  (AIM block-BiCGSTAB)")
S_theta_0 = out_bcg[1].S_fw_theta
S_phi_0   = out_bcg[1].S_fw_phi
A = (S_theta_0 + S_phi_0) / 2
B = (S_theta_0 - S_phi_0) / 2
function _max_sym_err(out, A, B, ALPHAS)
    e = 0.0
    for (k, α) in enumerate(ALPHAS)
        pred_θ = A + B * exp(+2im * α)
        pred_φ = A - B * exp(+2im * α)
        err_θ = abs(out[k].S_fw_theta - pred_θ) / abs(out[k].S_fw_theta)
        err_φ = abs(out[k].S_fw_phi   - pred_φ) / abs(out[k].S_fw_phi)
        e = max(e, err_θ, err_φ)
    end
    return e
end
@printf("  Max symmetry-test relative error: %.3e\n",
        _max_sym_err(out_bcg, A, B, ALPHAS))

println("\nDone.")
