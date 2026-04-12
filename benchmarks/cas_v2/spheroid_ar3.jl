# CAS-v2 spheroid benchmark (Phase 5.3).
#
# Build a prolate spheroid with aspect ratio AR = c/a = 3 and verify
# that VIEM's CAS-v2 observables obey the analytical phi-expansion that
# block-DDA_Py exploits in `run_dda_spheroid_sweep.py`:
#
#     S_θ(α, β, γ=0) = A(β) + B(β) · exp(+2j·α)
#     S_φ(α, β, γ=0) = A(β) - B(β) · exp(+2j·α)
#
# where  A(β) = (S_θ(α=0, β) + S_φ(α=0, β)) / 2
#        B(β) = (S_θ(α=0, β) - S_φ(α=0, β)) / 2
#
# This identity follows from the C∞ rotational symmetry of the spheroid
# about its symmetry axis. Verifying it for VIEM is a strong correctness
# test for the multi-orientation CAS pipeline.
#
# CONVENTION NOTE: block-DDA_Py uses `+2j·α` because it works in the
# physics convention (e^{-iωt}). VIEM internally uses the engineering
# convention (e^{+jωt}) and `compute_cas_observables` converts the final
# S to physics convention via complex conjugation. After that conversion,
# the spheroid identity should hold with the SAME sign as DDA. Below we
# verify both signs and report which holds.
#
# Run with:
#     julia --project=. benchmarks/cas_v2/spheroid_ar3.jl

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

"""
Build a prolate spheroid mesh with semi-axes (a, a, c) and target
mesh size `lc`. The symmetry axis is +z.
"""
function spheroid_mesh(a::Float64, c::Float64, lc::Float64)
    path = joinpath(tempdir(), "spheroid_a$(a)_c$(c)_lc$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("spheroid")
        # Build a unit sphere then dilate to (a, a, c).
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

# ── Physical parameters (matching block-DDA_Py defaults) ─────────────────────
const WL_0  = 0.638               # vacuum wavelength [μm]
const M_M   = 1.0                  # background medium RI (air)
const M_P   = 1.5 + 0.0im           # particle RI (non-absorbing)
const D_VE  = 0.40                  # volume-equivalent diameter [μm]
const AR    = 3.0                   # aspect ratio c/a (prolate)

# DDA convention (`shape_model/gaussian_ellipsoid.py`):
#   bc_ratio = b/c, range [1.0, 7.0]   ⇒  b ≥ c
#   ab_ratio = a/b, range [1.0, 2.0]   ⇒  a ≥ b
# For an axisymmetric spheroid, ab_ratio = 1 (a = b are equatorial radii)
# and bc_ratio = AR. AR > 1 means a = b > c, i.e. an OBLATE spheroid (a
# discus with the symmetry axis = c along z, the THIN direction).
const r_ve = D_VE / 2
const b_eq = r_ve * AR^( 1/3)        # equatorial semi-axes a = b
const c_ax = r_ve * AR^(-2/3)        # polar semi-axis c (= symmetry axis)

const eps_p   = M_P^2
const eps_bg  = M_M^2
const k0      = 2π * M_M / WL_0
const k0_eff  = k0 * abs(M_P)       # in-medium wavenumber, for kh diagnostics

println("=" ^ 70)
println("CAS-v2 spheroid benchmark (AR=$AR, oblate)")
println("=" ^ 70)
@printf("  D_ve=%.3f μm, AR=b/c=%.1f → semi-axes (a,b,c) = (%.4f, %.4f, %.4f) μm\n",
        D_VE, AR, b_eq, b_eq, c_ax)
@printf("  λ_0=%.3f μm, m_p=%s, k0=%.4f μm⁻¹, x=k0·r_ve=%.4f\n",
        WL_0, M_P, k0, k0 * r_ve)

# ── Build mesh ────────────────────────────────────────────────────────────────
const LC = 0.06                      # target mesh size [μm] — refine for accuracy
mesh_path = spheroid_mesh(b_eq, c_ax, LC)
mesh = read_msh(mesh_path)
basis = build_swg_basis(mesh; include_boundary_faces = true)
V_mesh = total_volume(mesh)
r_ve_mesh = (3 * V_mesh / (4π))^(1/3)
@printf("  Mesh: %d tets, %d SWG basis functions, V=%.4f μm³, r_ve_mesh=%.4f μm\n",
        n_tets(mesh), n_basis(basis), V_mesh, r_ve_mesh)
@printf("  Mesh r_ve / target r_ve = %.4f (volume error %.2f%%)\n",
        r_ve_mesh/r_ve, abs(r_ve_mesh - r_ve)/r_ve * 100)

# ── Multi-orientation solve at fixed β, varying α ────────────────────────────
const BETA = π/4                       # fixed polar Euler angle
const ALPHAS = (0.0, π/8, π/4, 3π/8, π/2)

euler_list = [(α, BETA, 0.0) for α in ALPHAS]

println("\n  Solving VIEM at $(length(euler_list)) orientations (β=π/4)…")
results = @time solve_cas_v2_orientations(basis, euler_list;
                                           k0 = k0, eps_p = eps_p,
                                           eps_bg = eps_bg,
                                           duffy_rule = duffy_reference_rule(7))

# ── Verify the analytical phi expansion ──────────────────────────────────────
S_theta_0 = results[1].S_fw_theta
S_phi_0   = results[1].S_fw_phi
A = (S_theta_0 + S_phi_0) / 2
B = (S_theta_0 - S_phi_0) / 2

println("\n  Spheroid symmetry test: S_θ(α) ?= A + B exp(2jα)")
println("  " * "-" ^ 65)
@printf("  %-10s %-22s %-22s %-10s\n", "α", "S_θ_observed", "S_θ_predicted", "rel.err")
global err_pos = 0.0
global err_neg = 0.0
for (k, α) in enumerate(ALPHAS)
    obs_θ = results[k].S_fw_theta
    obs_φ = results[k].S_fw_phi
    pred_θ_pos = A + B * exp(+2im * α)
    pred_φ_pos = A - B * exp(+2im * α)
    pred_θ_neg = A + B * exp(-2im * α)
    pred_φ_neg = A - B * exp(-2im * α)
    e_pos = max(abs(obs_θ - pred_θ_pos)/abs(obs_θ), abs(obs_φ - pred_φ_pos)/abs(obs_φ))
    e_neg = max(abs(obs_θ - pred_θ_neg)/abs(obs_θ), abs(obs_φ - pred_φ_neg)/abs(obs_φ))
    global err_pos = max(err_pos, e_pos)
    global err_neg = max(err_neg, e_neg)
    pred_θ = e_neg < e_pos ? pred_θ_neg : pred_θ_pos
    pred_φ = e_neg < e_pos ? pred_φ_neg : pred_φ_pos
    err_θ = abs(obs_θ - pred_θ) / abs(obs_θ)
    err_φ = abs(obs_φ - pred_φ) / abs(obs_φ)
    @printf("  α=%-6.4f  θ: %-+7.4f%+7.4fim   pred %-+7.4f%+7.4fim   %.2e\n",
            α, real(obs_θ), imag(obs_θ), real(pred_θ), imag(pred_θ), err_θ)
    @printf("           φ: %-+7.4f%+7.4fim   pred %-+7.4f%+7.4fim   %.2e\n",
            real(obs_φ), imag(obs_φ), real(pred_φ), imag(pred_φ), err_φ)
end
@printf("\n  Max symmetry-test relative error:\n")
@printf("    DDA-style  S_θ(α) = A + B exp(+2jα): %.3e\n", err_pos)
@printf("    VIEM-style S_θ(α) = A + B exp(-2jα): %.3e\n", err_neg)
err_min = min(err_pos, err_neg)
if err_min < 1e-3
    println("  ✓ PASS (< 1e-3)")
elseif err_min < 1e-2
    println("  ~ MARGINAL (< 1e-2, likely mesh-error-limited)")
else
    println("  ✗ FAIL (> 1e-2)")
end

# ── Print PCAS observables (S_fw) for each orientation ───────────────────────
println("\n  PCAS forward observables S_fw = (S_θ + S_φ)/2:")
println("  " * "-" ^ 65)
@printf("  %-10s %-12s %-22s %-22s\n", "α", "β", "S_fw", "S_bk")
for (k, α) in enumerate(ALPHAS)
    @printf("  α=%-6.4f  β=%-6.4f  %+8.5f%+8.5fim   %+8.5f%+8.5fim\n",
            α, BETA,
            real(results[k].S_fw), imag(results[k].S_fw),
            real(results[k].S_bk), imag(results[k].S_bk))
end

# ── Sphere-degenerate sanity check (AR ≈ 1) ──────────────────────────────────
println("\n" * "=" ^ 70)
println("Degenerate sanity check: AR=1 ⇒ should reduce to Mie sphere")
println("=" ^ 70)
mesh_path_sph = spheroid_mesh(r_ve, r_ve, LC)
mesh_sph = read_msh(mesh_path_sph)
basis_sph = build_swg_basis(mesh_sph; include_boundary_faces = true)
@printf("  Sphere mesh: N=%d basis functions\n", n_basis(basis_sph))

results_sph = solve_cas_v2_orientations(basis_sph, [(0.0, 0.0, 0.0)];
                                         k0 = k0, eps_p = eps_p,
                                         eps_bg = eps_bg,
                                         duffy_rule = duffy_reference_rule(7))
mie = mie_cas_observables(; wl_0 = WL_0, m_m = M_M,
                            r_p = (3 * total_volume(mesh_sph) / (4π))^(1/3),
                            m_p = M_P)
S_VIEM = results_sph[1].S_fw
S_Mie  = mie.S_fw
@printf("  S_fw VIEM:  %+9.6f %+9.6fim\n", real(S_VIEM), imag(S_VIEM))
@printf("  S_fw Mie :  %+9.6f %+9.6fim\n", real(S_Mie),  imag(S_Mie))
@printf("  Re rel.err = %.2e,  Im rel.err = %.2e\n",
        abs(real(S_VIEM) - real(S_Mie))/abs(real(S_Mie)),
        abs(imag(S_VIEM) - imag(S_Mie))/abs(imag(S_Mie)))

println("\nDone.")
