# Diagnostic: pointwise comparison of D_VIEM(r) vs D_Mie(r) inside a sphere.
#
# Purpose: determine whether the VIEM-solved D field itself is correct,
# independently of the far-field Fourier integral. If D is correct but
# F_fw is wrong, the bug is in postprocess.jl. If D is wrong, trace back
# to the impedance matrix or RHS.
#
# Usage: julia --project test/diagnose_D_field.jl

using LinearAlgebra: norm, dot
using StaticArrays
using Printf

# Load BlockVIEM
using BlockVIEM
using BlockVIEM: Vec3, TetMesh, SWGBasis, n_basis, evaluate,
                 assemble_impedance_matrix, project_plane_wave,
                 compute_scattering, assemble_mass_matrix,
                 _tet_vertices, tet_volume, bary_to_point, TET_QUAD_5PT
import Gmsh: gmsh

include(joinpath(@__DIR__, "mie_reference.jl"))
include(joinpath(@__DIR__, "mie_internal_field.jl"))

# ============================================================================
# Evaluate D_VIEM(r) at a point inside a given tetrahedron
# ============================================================================

"""
    evaluate_D_at_point(basis, D_coeffs, r, tet_idx) -> SVector{3,ComplexF64}

Evaluate D(r) = Σ_n D_n f_n(r) at point `r` inside tetrahedron `tet_idx`.
Only basis functions with support on `tet_idx` contribute.
"""
function evaluate_D_at_point(basis::SWGBasis, D_coeffs::AbstractVector,
                             r::Vec3, tet_idx::Int)
    D = SVector{3,ComplexF64}(0, 0, 0)
    N = n_basis(basis)
    @inbounds for n in 1:N
        if basis.tet_plus[n] == tet_idx || basis.tet_minus[n] == tet_idx
            fn = evaluate(basis, n, r, tet_idx)
            D += D_coeffs[n] * SVector{3,ComplexF64}(fn)
        end
    end
    return D
end

"""
    evaluate_D_at_centroids(basis, D_coeffs) -> (centroids, D_values)

Evaluate D_VIEM at every tet centroid. Returns vectors of positions and
D field values.
"""
function evaluate_D_at_centroids(basis::SWGBasis, D_coeffs::AbstractVector)
    mesh = basis.mesh
    ntet = length(mesh.tets)
    centroids = Vector{Vec3}(undef, ntet)
    D_vals = Vector{SVector{3,ComplexF64}}(undef, ntet)
    for t in 1:ntet
        centroids[t] = mesh.tet_centroids[t]
        D_vals[t] = evaluate_D_at_point(basis, D_coeffs, centroids[t], t)
    end
    return centroids, D_vals
end

# ============================================================================
# Sphere mesh generation
# ============================================================================

function generate_sphere_mesh(radius::Float64, lc::Float64)
    path = joinpath(tempdir(), "sphere_diag_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere_diag")
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

# ============================================================================
# Main diagnostic
# ============================================================================

function run_diagnostic(; m_p, label, lc=0.5)
    radius = 1.0
    wl_0 = 10.0
    m_m = 1.0
    eps_bg = m_m^2
    eps_p = ComplexF64(m_p)^2
    k0 = 2π * m_m / wl_0

    k_hat = Vec3(0, 0, 1)
    E0 = SVector{3,ComplexF64}(1.0, 0.0, 0.0)

    println("=" ^ 72)
    println("DIAGNOSTIC: $label")
    println("  m_p = $m_p, ε_p = $eps_p")
    println("  radius = $radius, wl_0 = $wl_0, k0 = $k0")
    println("  lc = $lc")
    println("=" ^ 72)

    # --- Generate mesh and solve VIEM ---
    path = generate_sphere_mesh(radius, lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    V_mesh = sum(mesh.tet_volumes)
    r_ve = (3V_mesh / (4π))^(1 / 3)
    println("  N_basis = $N, N_tet = $(length(mesh.tets)), V_mesh = $V_mesh")
    println("  r_ve = $r_ve (nominal = $radius)")

    dr = BlockVIEM.duffy_reference_rule(7)
    println("  Assembling Z matrix...")
    Z = assemble_impedance_matrix(basis; k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                  duffy_rule=dr, symmetrize=true)
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
    b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg)
    println("  Solving Z D = b...")
    D_coeffs = Z \ b
    residual = norm(Z * D_coeffs - b) / norm(b)
    println("  Residual = $residual")

    # --- Cross sections for context ---
    scat = compute_scattering(basis, D_coeffs;
                              k_hat=k_hat, E0=E0, k0=k0,
                              eps_p=eps_p, eps_bg=eps_bg)
    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
    println("\n  Cross sections (VIEM vs Mie at r_ve):")
    @printf("    C_abs: VIEM = %.6e, Mie = %.6e, rel_err = %.3f%%\n",
            scat.C_abs, mie.C_abs, 100 * abs(scat.C_abs - mie.C_abs) / mie.C_abs)
    @printf("    C_ext: VIEM = %.6e, Mie = %.6e, rel_err = %.3f%%\n",
            scat.C_ext, mie.C_ext, 100 * abs(scat.C_ext - mie.C_ext) / mie.C_ext)

    # --- Evaluate D_VIEM at tet centroids ---
    println("\n  Evaluating D_VIEM at tet centroids...")
    centroids, D_viem = evaluate_D_at_centroids(basis, D_coeffs)

    # --- Evaluate D_Mie at the same points ---
    println("  Evaluating D_Mie (BH convention → VIEM convention)...")
    D_mie = mie_D_field_viem_convention(centroids; wl_0=wl_0, m_m=m_m,
                                        r_p=radius, m_p=m_p)

    # --- Pointwise comparison ---
    println("\n  Pointwise comparison (D_VIEM vs D_Mie):")
    println("  " * "-" ^ 68)
    @printf("  %6s %8s %8s | %22s | %22s | %8s\n",
            "tet", "r/a", "z/a", "D_VIEM_x (re, im)", "D_Mie_x (re, im)", "rel_err")
    println("  " * "-" ^ 68)

    # Sort by z-coordinate for readability
    order = sortperm(centroids, by=c -> c[3])

    # Print a subset: 20 evenly spaced tets along z
    ntet = length(order)
    step = max(1, ntet ÷ 20)
    indices = order[1:step:end]

    rel_errors = Float64[]
    for t in order
        dv = D_viem[t]
        dm = D_mie[t]
        err = norm(dv - dm) / max(norm(dm), 1e-30)
        push!(rel_errors, err)
    end

    for t in indices
        c = centroids[t]
        r_norm = norm(c) / radius
        z_norm = c[3] / radius
        dv = D_viem[t]
        dm = D_mie[t]
        err = norm(dv - dm) / max(norm(dm), 1e-30)
        @printf("  %6d %8.3f %8.3f | (%+.4e,%+.4e) | (%+.4e,%+.4e) | %8.2f%%\n",
                t, r_norm, z_norm,
                real(dv[1]), imag(dv[1]),
                real(dm[1]), imag(dm[1]),
                100 * err)
    end

    # --- Summary statistics ---
    println("\n  Summary statistics over all $(length(rel_errors)) tets:")
    @printf("    Mean  |D_VIEM - D_Mie| / |D_Mie| = %.4f%%\n", 100 * sum(rel_errors) / length(rel_errors))
    @printf("    Max   |D_VIEM - D_Mie| / |D_Mie| = %.4f%%\n", 100 * maximum(rel_errors))
    @printf("    Min   |D_VIEM - D_Mie| / |D_Mie| = %.4f%%\n", 100 * minimum(rel_errors))
    p50 = sort(rel_errors)[length(rel_errors) ÷ 2]
    p90 = sort(rel_errors)[min(length(rel_errors), floor(Int, 0.9 * length(rel_errors)) + 1)]
    @printf("    Median                            = %.4f%%\n", 100 * p50)
    @printf("    90th percentile                   = %.4f%%\n", 100 * p90)

    # --- Phase comparison along z-axis ---
    println("\n  Phase of D_x along z-axis (centroids near x≈0, y≈0):")
    println("  " * "-" ^ 60)
    @printf("  %8s | %12s %12s | %12s %12s\n",
            "z/a", "arg(Dv_x)°", "|Dv_x|", "arg(Dm_x)°", "|Dm_x|")
    println("  " * "-" ^ 60)
    # Select centroids near the z-axis
    axis_tets = filter(t -> sqrt(centroids[t][1]^2 + centroids[t][2]^2) < 0.3 * radius,
                       1:ntet)
    sort!(axis_tets, by=t -> centroids[t][3])
    for t in axis_tets[1:max(1, length(axis_tets) ÷ 15):end]
        c = centroids[t]
        dv_x = D_viem[t][1]
        dm_x = D_mie[t][1]
        @printf("  %8.3f | %12.2f %12.4e | %12.2f %12.4e\n",
                c[3] / radius,
                rad2deg(angle(dv_x)), abs(dv_x),
                rad2deg(angle(dm_x)), abs(dm_x))
    end

    println()
    return (; centroids, D_viem, D_mie, rel_errors, scat, mie)
end

# ============================================================================
# Run for multiple contrast levels
# ============================================================================

println("\n" * "=" ^ 72)
println("   D-FIELD DIAGNOSTIC: VIEM vs Mie internal field")
println("=" ^ 72)

# Low contrast (should be easiest)
result_low = run_diagnostic(m_p=1.5 + 0.01im, label="Low contrast (m=1.5+0.01i)", lc=0.5)

# Iron oxide (the problem case)
result_fe = run_diagnostic(m_p=2.5 + 0.5im, label="Iron oxide (m=2.5+0.5i)", lc=0.5)

println("\n" * "=" ^ 72)
println("   DIAGNOSTIC COMPLETE")
println("=" ^ 72)
