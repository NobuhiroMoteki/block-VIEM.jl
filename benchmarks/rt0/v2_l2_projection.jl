# Verification 2: L² projection error of a uniform D field onto the SWG basis.
#
# Question: how much accuracy is lost JUST by representing a uniform D field
# in the SWG (RT0) basis, before any solver / quadrature error?
#
# Setup: take D_exact(r) = D_0 (constant vector) inside a sphere mesh.
# Find SWG coefficients c_n that minimize ||D_exact - Σ_n c_n f_n||_L²:
#     M c = b,  where M_mn = ∫ f_m·f_n dV,  b_m = ∫ f_m·D_exact dV
# Then compute the L² projection error:
#     err² = ||D_exact||² - c^T b
# and the C_abs that this projected D would give.
#
# Compares against:
#  - The exact C_abs of a uniform D field (no Mie correction):
#       C_abs_uniform = k₀ Im(ε_p) / |ε_p|² × |D₀|² × V_mesh
#  - The VIEM-solved D†MD value
#  - The Mie reference value
#
# This isolates the BASIS APPROXIMATION error from the SOLVER error.

using LinearAlgebra: norm, dot
using Printf
using StaticArrays
using SparseArrays
using BlockVIEM
using BlockVIEM: Vec3, _tet_vertices, tet_volume, bary_to_point, evaluate
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh_v2(lc)
    path = joinpath(tempdir(), "sph_v2_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v2_$(lc)")
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

"""
Compute the L² projection of a constant vector field D₀ onto the SWG basis.
Returns (c, projected_L2_squared, exact_L2_squared, projection_error_L2).
"""
function project_constant_field(basis, D0::SVector{3,ComplexF64}; rule=TET_QUAD_64PT)
    N = n_basis(basis)
    M = BlockVIEM.assemble_mass_matrix(basis; rule=rule)

    # b_m = ∫ f_m · D₀ dV  (integrand has a smooth f_m and constant D₀)
    b = zeros(ComplexF64, N)
    for m in 1:N
        s = zero(ComplexF64)
        for tet in BlockVIEM.support_tets(basis, m)
            tet == 0 && continue
            verts = _tet_vertices(basis.mesh, tet)
            V = tet_volume(verts...)
            for i in 1:rule.n
                r = bary_to_point(rule.bary[i], verts)
                fm = evaluate(basis, m, r, tet)
                s += rule.weights[i] * V * dot(SVector{3,ComplexF64}(fm), D0)
            end
        end
        b[m] = s
    end

    # Solve M c = b for the projection coefficients
    M_dense = Matrix(M)  # M is real & symmetric, fine to densify for small N
    c = M_dense \ b

    # ||P_h D||² = c† M c
    proj_L2_sq = real(dot(c, M_dense * c))

    # ||D_exact||² = |D₀|² × V_mesh
    V_mesh = total_volume(basis.mesh)
    exact_L2_sq = real(dot(D0, D0)) * V_mesh

    # Projection error in L²: ||D_exact - P_h D||² = ||D_exact||² - 2 Re<D_exact, P_h D> + ||P_h D||²
    # Since P_h is the L² projector, <D_exact - P_h D, P_h D> = 0,
    # so ||D_exact - P_h D||² = ||D_exact||² - ||P_h D||²
    proj_err_sq = exact_L2_sq - proj_L2_sq

    return c, proj_L2_sq, exact_L2_sq, proj_err_sq
end

flush(stdout)
println("=" ^ 88); flush(stdout)
println("  Verification 2: SWG L² projection error of a uniform D field"); flush(stdout)
println("=" ^ 88); flush(stdout)
println()

# Test with low-contrast and high-contrast cases
for (m_p, label, k0) in [
        (1.5 + 0.01im, "low contrast", 2π/10.0),
        (sqrt(10.0+1.0im), "high contrast", 0.316),
    ]
    eps_p = ComplexF64(m_p)^2

    # Clausius-Mossotti uniform internal D field for a sphere
    D_CM = 3 * eps_p / (eps_p + 2)
    D0 = SVector{3,ComplexF64}(D_CM, 0, 0)

    println("=" ^ 88); flush(stdout)
    @printf("  %s: m=%s   k0=%.4f   D_CM=%s\n", label, string(m_p), k0, string(D_CM))
    println("=" ^ 88); flush(stdout)

    @printf("%6s %6s %8s | %12s %12s %12s | %10s %10s\n",
            "lc", "N", "h̄", "||D||²V", "||P_h D||²", "||err||²", "rel_L2",
            "C_abs_proj/Mie")
    println("-"^110); flush(stdout)

    for lc in [0.7, 0.5, 0.35, 0.25]
        mesh = read_msh(sphere_mesh_v2(lc))
        basis = build_swg_basis(mesh)
        N = n_basis(basis)
        V_mesh = total_volume(mesh)
        r_ve = (3V_mesh/(4π))^(1/3)

        # mean edge length
        h_sum = 0.0; cnt = 0
        for tet in mesh.tets, (a,b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
            h_sum += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); cnt += 1
        end
        h_bar = h_sum / cnt

        c, proj_L2_sq, exact_L2_sq, proj_err_sq = project_constant_field(basis, D0)
        rel_L2 = sqrt(max(proj_err_sq, 0.0) / exact_L2_sq)

        # C_abs from the projected coefficients
        C_abs_proj = real(k0) * imag(eps_p) / (1.0 * abs2(eps_p) * 1.0) * proj_L2_sq

        # Mie reference at r_ve
        mie = mie_cross_sections(; wl_0 = 2π / k0, m_m = 1.0, r_p = r_ve, m_p = m_p)
        ratio = C_abs_proj / mie.C_abs

        @printf("%6.3f %6d %8.4f | %12.4e %12.4e %12.4e | %9.2f%% %12.4f\n",
                lc, N, h_bar, exact_L2_sq, proj_L2_sq, max(proj_err_sq, 0.0),
                100*rel_L2, ratio)
        flush(stdout)
    end
    println()
end

println("=" ^ 88)
println("  Interpretation:")
println("  - rel_L2  = relative L² error of the SWG projection of a uniform D₀")
println("              (lower bound on any solver error: even a perfect solver")
println("              cannot beat this for representing a uniform field)")
println("  - C_abs_proj/Mie = ratio of the projected uniform-D C_abs to the Mie value")
println("              if this ratio ≈ 1, the SWG basis CAN represent the Mie")
println("              answer; if it's far from 1, the basis is the bottleneck")
println("=" ^ 88)
