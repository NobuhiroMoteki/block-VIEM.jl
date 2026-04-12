# V7: boundary-tet vs interior-tet L2 error of D(r) (investigation #3).
#
# For the VIEM-SWG solution D_VIEM(r), compute the pointwise L2 error
# against the exact Mie internal field D_Mie(r), tetrahedron by
# tetrahedron. Each tet is classified as
#
#   interior   -- all 4 faces are internal (shared with a neighbor tet),
#                 i.e. every face carries an SWG DOF
#   boundary   -- at least one face is a global boundary face (no DOF)
#
# If investigation #3 (SWG structural limit on boundary normal D) is the
# dominant error source, we expect:
#
#   (a) boundary tets to show much larger L2 error than interior tets
#   (b) removing/down-weighting them in a volume-averaged metric would
#       drop the error substantially
#
# A null outcome (interior tets ~ boundary tets error) would rule out
# this hypothesis and force us to look elsewhere.

using LinearAlgebra: norm, dot
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3, SWGBasis, _tet_vertices, bary_to_point,
                 TET_QUAD_5PT, n_basis, evaluate, support_tets
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))
include(joinpath(TEST_DIR, "mie_internal_field.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v7_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v7_$(lc)")
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

# Interior tet = all 4 faces are shared (= 4 SWG DOFs reference it).
# Boundary tet = at least one face on the global exterior surface.
function classify_tets(basis::SWGBasis)
    ntet = length(basis.mesh.tets)
    face_count = zeros(Int, ntet)
    for n in 1:n_basis(basis)
        tp, tm = support_tets(basis, n)
        tp > 0 && (face_count[tp] += 1)
        tm > 0 && (face_count[tm] += 1)
    end
    is_boundary = face_count .< 4
    return is_boundary, face_count
end

function evaluate_D_viem(basis::SWGBasis, D_coeffs, r::Vec3, tet::Int)
    D = SVector{3, ComplexF64}(0, 0, 0)
    @inbounds for n in 1:n_basis(basis)
        if basis.tet_plus[n] == tet || basis.tet_minus[n] == tet
            fn = evaluate(basis, n, r, tet)
            D += D_coeffs[n] * SVector{3, ComplexF64}(fn)
        end
    end
    return D
end

# Volume-weighted L2 error over a subset of tets
function tet_l2(basis::SWGBasis, D_coeffs, tet_indices, radius, wl_0, m_m, m_p)
    eps_p = ComplexF64(m_p)^2
    coeffs = mie_internal_coefficients(; wl_0=wl_0, m_m=m_m, r_p=radius, m_p=m_p)
    mesh = basis.mesh
    rule = TET_QUAD_5PT

    err2 = 0.0
    mie2 = 0.0
    V_sub = 0.0
    @inbounds for t in tet_indices
        verts = _tet_vertices(mesh, t)
        V = mesh.tet_volumes[t]
        V_sub += V
        for i in 1:rule.n
            r = bary_to_point(rule.bary[i], verts)
            w = rule.weights[i] * V
            Dv = evaluate_D_viem(basis, D_coeffs, r, t)
            E_bh = mie_internal_field(SVector{3, Float64}(r);
                                      wl_0=wl_0, m_m=m_m, r_p=radius,
                                      m_p=m_p, coeffs=coeffs)
            Dm = eps_p * conj(E_bh)
            diff = Dv - Dm
            err2 += w * real(dot(diff, diff))
            mie2 += w * real(dot(Dm, Dm))
        end
    end
    return (; err2, mie2, V_sub, rel=sqrt(err2 / mie2))
end

const m_p    = 1.5 + 0.01im
const wl_0   = 10.0
const m_m    = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0     = 2π * m_m / wl_0
const k_hat  = Vec3(0, 0, 1)
const E0     = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

println("=" ^ 100); flush(stdout)
println("  V7: boundary vs interior L2 error of D_VIEM vs D_Mie (investigation #3)")
println("  m=1.5+0.01i, wl=10, ka_ve~0.628")
println("=" ^ 100); flush(stdout)

@printf("%6s %7s %6s %6s %7s %7s | %10s %10s %10s | %7s %7s %7s | %7s\n",
        "lc", "N_swg", "N_tet", "N_int", "N_bnd", "V_bnd%",
        "L2_all", "L2_int", "L2_bnd",
        "errC_a%", "errC_s%", "errC_e%", "solve(s)")
println("-"^128); flush(stdout)

for lc in (0.50, 0.35, 0.25, 0.18)
    path = sphere_mesh(lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    ntet = length(mesh.tets)
    V_mesh = total_volume(mesh)
    r_ve = (3V_mesh / (4π))^(1/3)

    is_bnd, _ = classify_tets(basis)
    bnd_tets = findall(is_bnd)
    int_tets = findall(.!is_bnd)
    V_bnd = sum(mesh.tet_volumes[t] for t in bnd_tets; init=0.0)

    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)

    # Solve: dense for all sizes here (lc>=0.18 -> N<=7000 still dense-feasible
    # with symmetrize=true thread-parallel assemble)
    if N <= 1500
        t_solve = @elapsed res = solve_direct(basis;
                                               k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                               k_hat=k_hat, E0=E0)
    else
        # Use AIM iterative for larger meshes
        h_bar = 0.0; c = 0
        for tet in mesh.tets, (a,b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
            h_bar += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
        end
        h_bar /= c
        pitch = 0.5 * h_bar
        t_solve = @elapsed res = solve_iterative(basis;
                                                  k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                                  k_hat=k_hat, E0=E0,
                                                  pitch=pitch, padding=4,
                                                  tol=1e-7, maxiter=600)
    end
    D = res.D_coeffs

    scat = compute_scattering(basis, D;
                               k_hat=k_hat, E0=E0, k0=k0,
                               eps_p=eps_p, eps_bg=eps_bg,
                               csca_method=:farfield, n_theta=20)

    l2_all = tet_l2(basis, D, 1:ntet, r_ve, wl_0, m_m, m_p)
    l2_int = tet_l2(basis, D, int_tets, r_ve, wl_0, m_m, m_p)
    l2_bnd = tet_l2(basis, D, bnd_tets, r_ve, wl_0, m_m, m_p)

    ea = 100 * (scat.C_abs - mie.C_abs) / mie.C_abs
    es = 100 * (scat.C_sca - mie.C_sca) / mie.C_sca
    ex = 100 * (scat.C_ext - mie.C_ext) / mie.C_ext

    @printf("%6.3f %7d %6d %6d %7d %6.1f%% | %10.2e %10.2e %10.2e | %+6.2f%% %+6.2f%% %+6.2f%% | %7.1f\n",
            lc, N, ntet, length(int_tets), length(bnd_tets),
            100 * V_bnd / V_mesh,
            l2_all.rel, l2_int.rel, l2_bnd.rel,
            ea, es, ex, t_solve)
    flush(stdout)
end

println()
println("  Interpretation:")
println("    L2_bnd >> L2_int  -> boundary tets dominate the error (investigation #3 confirmed)")
println("    L2_bnd ~= L2_int  -> error is uniform in space (rule out #3, inspect mass term)")
flush(stdout)
