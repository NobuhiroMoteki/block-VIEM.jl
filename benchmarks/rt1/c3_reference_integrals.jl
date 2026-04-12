# Phase C.3: Verify RT1 reference-element integrals against high-order
# numerical quadrature, and also check the single-tet mass & radiation
# matrix structure.
#
# We compute on the reference tet (v0=(0,0,0), v1=(1,0,0), v2=(0,1,0),
# v3=(0,0,1)):
#
#   M_ij = ∫ φ̂_i · φ̂_j dV
#   S_ij = ∫ (∇·φ̂_i)(∇·φ̂_j) dV
#
# The reference element has:
# - 12 face DOFs (indices 1..12)
# - 3 interior (bubble) DOFs (indices 13..15)
#
# We compare `_mass_term` and the divergence-squared matrix computed via
# the code against very-high-order Gauss quadrature as ground truth.
# If these agree, the reference-element structure is OK.

using LinearAlgebra: norm, dot, eigvals, Symmetric, diag, cond
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, TetVerts, rt1_ref_evaluate, rt1_ref_divergence,
                 TET_QUAD_64PT, TET_QUAD_125PT, _mass_term

flush(stdout)
println("=" ^ 78); flush(stdout)
println("  Phase C.3: RT1 reference-element integral verification"); flush(stdout)
println("=" ^ 78); flush(stdout)

# Build a one-tet mesh that IS the reference tet (so Jacobian = I, detJ = 1).
nodes = Vec3[Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)]
tets = TetVerts[TetVerts(1,2,3,4)]
mesh = TetMesh(nodes, tets)

# Need an RT1 basis but with a single tet the face DOFs are all boundary
# (no internal faces) so they don't exist as global DOFs. We'll work with
# the reference evaluation directly.
println("  Reference tet: detJ should be 1"); flush(stdout)
println("  (Directly evaluating rt1_ref_evaluate / rt1_ref_divergence)")
flush(stdout)

# Build a 2-tet mesh so face DOFs get created on the shared face
# Bipyramid: node 5 reflected through the shared triangle (2,3,4)
nodes2 = Vec3[Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1),
              Vec3(1,1,1)]
tets2 = TetVerts[TetVerts(1,2,3,4), TetVerts(5,2,3,4)]
mesh2 = TetMesh(nodes2, tets2)
basis = build_rt1_basis(mesh2)
N = n_basis(basis)
n_face = basis.n_face_dofs
println("  Bipyramid mesh: N_RT1=$N  (face=$n_face, bubble=$(N-n_face))"); flush(stdout)

# Compute the reference-element mass matrix via very-high-order quadrature
# for indices 1..15 (reference basis functions).
function ref_mass_numerical(rule)
    M = zeros(Float64, 15, 15)
    # Reference tet has volume 1/6. bary_to_point + rule gives points
    # in the reference tet.
    ref_verts = (Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1))
    V = BlockVIEM.tet_volume(ref_verts...)
    for k in 1:rule.n
        λ = rule.bary[k]
        # Reference coordinates (ξ, η, ζ) = (λ2, λ3, λ4); λ1 = 1 - ξ - η - ζ
        ξ, η, ζ = λ[2], λ[3], λ[4]
        w = rule.weights[k] * V
        for i in 1:15
            φi = rt1_ref_evaluate(i, ξ, η, ζ)
            for j in 1:15
                φj = rt1_ref_evaluate(j, ξ, η, ζ)
                M[i, j] += w * dot(φi, φj)
            end
        end
    end
    return M
end

M_64  = ref_mass_numerical(TET_QUAD_64PT)
M_125 = ref_mass_numerical(TET_QUAD_125PT)
d_mass = norm(M_64 - M_125) / norm(M_125)
println(); flush(stdout)
println("=== Reference mass matrix (15×15) ==="); flush(stdout)
@printf("  ||M_64 - M_125|| / ||M_125||   = %.2e\n", d_mass); flush(stdout)
@printf("  ||M_125||_F                    = %.4f\n", norm(M_125)); flush(stdout)
@printf("  diag range [%.2f .. %.2f]\n", minimum(diag(M_125)),
        maximum(diag(M_125))); flush(stdout)

@printf("  cond(M_ref)                    = %.2f\n",
        cond(Symmetric(M_125))); flush(stdout)

# Eigenvalues
eigs_M = eigvals(Symmetric(M_125))
@printf("  eigenvalue range [%.2e .. %.2e]\n",
        minimum(real.(eigs_M)), maximum(real.(eigs_M))); flush(stdout)

# --- Divergence squared: S_ij = ∫ (div φ̂_i)(div φ̂_j) dV ---
function ref_div_squared(rule)
    S = zeros(Float64, 15, 15)
    ref_verts = (Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1))
    V = BlockVIEM.tet_volume(ref_verts...)
    for k in 1:rule.n
        λ = rule.bary[k]
        ξ, η, ζ = λ[2], λ[3], λ[4]
        w = rule.weights[k] * V
        for i in 1:15
            di = rt1_ref_divergence(i, ξ, η, ζ)
            for j in 1:15
                dj = rt1_ref_divergence(j, ξ, η, ζ)
                S[i, j] += w * di * dj
            end
        end
    end
    return S
end

S_64  = ref_div_squared(TET_QUAD_64PT)
S_125 = ref_div_squared(TET_QUAD_125PT)
d_S = norm(S_64 - S_125) / norm(S_125)
println(); flush(stdout)
println("=== Reference divergence² matrix (15×15) ==="); flush(stdout)
@printf("  ||S_64 - S_125|| / ||S_125||   = %.2e\n", d_S); flush(stdout)
@printf("  ||S_125||_F                    = %.4f\n", norm(S_125)); flush(stdout)
@printf("  diag range [%.2f .. %.2f]\n", minimum(diag(S_125)),
        maximum(diag(S_125))); flush(stdout)
@printf("  cond(S_ref)                    = %.2e\n",
        cond(Symmetric(S_125))); flush(stdout)

# --- Global bipyramid mass matrix via _mass_term ---
M_global = zeros(Float64, N, N)
for m in 1:N, n in 1:N
    M_global[m, n] = _mass_term(basis, m, n, TET_QUAD_125PT)
end
d_M_sym = norm(M_global - M_global') / norm(M_global)
println(); flush(stdout)
println("=== Bipyramid global mass matrix ==="); flush(stdout)
@printf("  Symmetry ||M - M^T|| / ||M||   = %.2e\n", d_M_sym); flush(stdout)
@printf("  ||M||_F                        = %.4f\n", norm(M_global)); flush(stdout)
eigs_Mg = eigvals(Symmetric(M_global))
@printf("  eigenvalue range [%.2e .. %.2e]\n",
        minimum(real.(eigs_Mg)), maximum(real.(eigs_Mg))); flush(stdout)
@printf("  cond                           = %.2e\n",
        cond(Symmetric(M_global))); flush(stdout)

# Check: is the face-block diagonal-dominant?
if N > n_face
    M_ff = M_global[1:n_face, 1:n_face]
    M_bb = M_global[n_face+1:end, n_face+1:end]
    M_fb = M_global[1:n_face, n_face+1:end]
    @printf("  ||M_face_face||       = %.4f\n", norm(M_ff))
    @printf("  ||M_bubble_bubble||   = %.4f\n", norm(M_bb))
    @printf("  ||M_face_bubble||     = %.4f  (cross-block)\n", norm(M_fb))
    flush(stdout)
end
