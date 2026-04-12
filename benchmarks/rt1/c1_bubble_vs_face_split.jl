# Phase C.1: Decompose RT1 solution into face-DOF and bubble-DOF
# contributions to diagnose the 37% D†MD overshoot.
#
# Hypothesis: bubble DOFs carry spurious intra-element oscillations that
# don't contribute to far-field (C_sca OK) but inflate D†MD (C_abs wrong).
#
# Experiment: split D = D_face + D_bubble, report contributions to D†MD
# and also try solving with bubble DOFs forcibly zeroed.

using LinearAlgebra: norm, dot
using StaticArrays
using Printf
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

include(joinpath(@__DIR__, "..", "..", "test", "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_rt1c1_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("c1_$(lc)")
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

flush(stdout)
println("=" ^ 78); flush(stdout)
println("  Phase C.1: RT1 face vs bubble DOF decomposition"); flush(stdout)
println("=" ^ 78); flush(stdout)

mesh = read_msh(sphere_mesh(0.7))
basis = build_rt1_basis(mesh)
N = n_basis(basis)
n_face = basis.n_face_dofs
n_bubble = N - n_face
V_mesh = total_volume(mesh)
r_ve = (3V_mesh / (4π))^(1/3)

println("  mesh: Ntet=$(length(mesh.tets))  N_RT1=$N  (face=$n_face, bubble=$n_bubble)  r_ve=$(round(r_ve,digits=4))"); flush(stdout)

# Parameters
const M_P    = 1.5 + 0.01im
const EPS_P  = ComplexF64(M_P)^2
const EPS_BG = 1.0
const K0     = 2π / 10.0  # wl_0 = 10

k_hat = Vec3(0, 0, 1)
E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)
mie = mie_cross_sections(; wl_0=10.0, m_m=1.0, r_p=r_ve, m_p=M_P)

println("  k0=$(round(K0,digits=4))  m_p=$M_P  eps_p=$EPS_P"); flush(stdout)
println(); flush(stdout)

# Solve full RT1 VIEM
println("Assembling RT1 Z matrix ..."); flush(stdout)
outer_rule = TET_QUAD_64PT
dr = duffy_reference_rule(7)
t_Z = @elapsed Z = assemble_impedance_matrix(basis; k0=K0, eps_p=EPS_P, eps_bg=EPS_BG,
                                              outer_rule=outer_rule, duffy_rule=dr, symmetrize=true)
println("  t_Z = $(round(t_Z, digits=1)) s"); flush(stdout)

k_bg = ComplexF64(K0) * sqrt(ComplexF64(EPS_BG))
b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg, rule=outer_rule)
D_full = Z \ b
residual = norm(Z*D_full - b) / norm(b)
println("  residual = $(round(residual, sigdigits=2))"); flush(stdout)

# Mass matrix for L² computations
M_mat = BlockVIEM.assemble_mass_matrix(basis; rule=outer_rule)

# Split: face DOFs (indices 1..n_face) vs bubble DOFs (n_face+1..N)
D_face = copy(D_full)
D_face[n_face+1:end] .= 0.0

D_bubble = copy(D_full)
D_bubble[1:n_face] .= 0.0

DhMD_full    = real(dot(D_full, M_mat * D_full))
DhMD_face    = real(dot(D_face, M_mat * D_face))
DhMD_bubble  = real(dot(D_bubble, M_mat * D_bubble))
DhMD_cross   = real(dot(D_face, M_mat * D_bubble)) + real(dot(D_bubble, M_mat * D_face))

println(); flush(stdout)
println("=== L² decomposition (D†MD) ==="); flush(stdout)
@printf("  full          = %.4f\n", DhMD_full); flush(stdout)
@printf("  face only     = %.4f  (%.1f%% of full)\n", DhMD_face, 100*DhMD_face/DhMD_full)
@printf("  bubble only   = %.4f  (%.1f%% of full)\n", DhMD_bubble, 100*DhMD_bubble/DhMD_full)
@printf("  cross term    = %.4f  (%.1f%% of full)\n", DhMD_cross, 100*DhMD_cross/DhMD_full)
@printf("  sum check     = %.4f\n", DhMD_face + DhMD_bubble + DhMD_cross)
flush(stdout)

# Expected D†MD from Clausius-Mossotti
D_CM = 3 * EPS_BG * EPS_P / (EPS_P + 2*EPS_BG)
DhMD_CM = abs2(D_CM) * V_mesh
@printf("  Clausius-Mossotti   = %.4f\n", DhMD_CM); flush(stdout)
@printf("  ratio full/CM       = %.4f\n", DhMD_full / DhMD_CM); flush(stdout)
@printf("  ratio face_only/CM  = %.4f\n", DhMD_face / DhMD_CM); flush(stdout)
flush(stdout)

# C_abs from full vs face-only D
function c_abs_from_dhmd(dhmd)
    real(K0) * imag(EPS_P) / (real(EPS_BG) * abs2(EPS_P) * 1.0) * dhmd
end

println(); flush(stdout)
println("=== C_abs comparison ==="); flush(stdout)
C_abs_full = c_abs_from_dhmd(DhMD_full)
C_abs_face = c_abs_from_dhmd(DhMD_face)
@printf("  C_abs_Mie        = %.4e\n", mie.C_abs)
@printf("  C_abs_full       = %.4e  (err %.1f%%)\n", C_abs_full,
        100*abs(C_abs_full-mie.C_abs)/mie.C_abs)
@printf("  C_abs_face_only  = %.4e  (err %.1f%%)\n", C_abs_face,
        100*abs(C_abs_face-mie.C_abs)/mie.C_abs)
flush(stdout)

# --- Experiment 2: Solve the system restricted to face DOFs only ---
println(); flush(stdout)
println("=== Solving with bubble DOFs suppressed (Z_ff D_f = b_f) ==="); flush(stdout)
Z_ff = Z[1:n_face, 1:n_face]
b_f = b[1:n_face]
D_f_only = Z_ff \ b_f
D_ff = zeros(ComplexF64, N)
D_ff[1:n_face] .= D_f_only
DhMD_ff = real(dot(D_ff, M_mat * D_ff))
C_abs_ff = c_abs_from_dhmd(DhMD_ff)
@printf("  D†MD_ff          = %.4f\n", DhMD_ff)
@printf("  C_abs_ff         = %.4e  (err %.1f%%)\n", C_abs_ff,
        100*abs(C_abs_ff-mie.C_abs)/mie.C_abs)
flush(stdout)

# --- Experiment 3: far-field scattering from the full vs face-only solution ---
println(); flush(stdout)
println("=== C_sca (far-field) comparison ==="); flush(stdout)
scat_full = compute_scattering(basis, D_full;
                                k_hat=k_hat, E0=E0, k0=K0,
                                eps_p=EPS_P, eps_bg=EPS_BG,
                                rule=outer_rule, csca_method=:farfield, n_theta=15)
scat_face = compute_scattering(basis, D_face;
                                k_hat=k_hat, E0=E0, k0=K0,
                                eps_p=EPS_P, eps_bg=EPS_BG,
                                rule=outer_rule, csca_method=:farfield, n_theta=15)
@printf("  C_sca_Mie        = %.4e\n", mie.C_sca)
@printf("  C_sca_full       = %.4e  (err %.1f%%)\n", scat_full.C_sca,
        100*abs(scat_full.C_sca-mie.C_sca)/mie.C_sca)
@printf("  C_sca_face_only  = %.4e  (err %.1f%%)\n", scat_face.C_sca,
        100*abs(scat_face.C_sca-mie.C_sca)/mie.C_sca)
flush(stdout)
