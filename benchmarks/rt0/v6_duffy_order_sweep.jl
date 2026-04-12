# V6: does the self-term Duffy quadrature order affect the result?
#
# Investigation #2 from memory/v2_v3_dda_comparison.md.
#
# The default self-term integration is 5-pt outer x 5-pt Duffy reference
# rule (1D order -> tensor-product on a cube). If 5-pt is under-resolved,
# raising the Duffy order should change C_abs / C_sca / C_ext measurably.
# If it does NOT, the self-term quadrature is converged and the per-DOF
# gap to DDA is NOT coming from self-term quadrature error.
#
# We use a small enough mesh that a dense assemble is feasible with
# high-order Duffy rules, and do not go through solve_direct (which does
# not expose duffy_rule). Instead we call assemble_impedance_matrix
# directly with the chosen duffy order.
#
# Run 3 duffy orders (5, 7, 10) on the SAME mesh. Any movement in the
# output = under-resolved. No movement = fine.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v6_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v6_$(lc)")
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
    for tet in mesh.tets, (a, b) in ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        s += norm(mesh.nodes[tet[a]] - mesh.nodes[tet[b]]); c += 1
    end
    s / c
end

function solve_with_duffy_order(basis, k0, eps_p, eps_bg, k_hat, E0, duffy_n)
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
    b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg)
    duffy_rule = duffy_reference_rule(duffy_n)
    Z = assemble_impedance_matrix(basis;
                                   k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                   duffy_rule=duffy_rule,
                                   symmetrize=true)
    x = Z \ b
    return x
end

const m_p    = 1.5 + 0.01im
const wl_0   = 10.0
const m_m    = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0     = 2π * m_m / wl_0
const k_hat  = Vec3(0, 0, 1)
const E0     = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

println("=" ^ 104); flush(stdout)
println("  V6: self-term Duffy quadrature order sweep (investigation #2)")
println("  m=1.5+0.01i, wl=10, ka_ve~0.628. Dense assemble at each Duffy order.")
println("=" ^ 104); flush(stdout)

for lc in (0.5, 0.35)
    path = sphere_mesh(lc)
    mesh = read_msh(path)
    basis = build_swg_basis(mesh)
    N = n_basis(basis)
    V_mesh = total_volume(mesh)
    r_ve = (3V_mesh / (4π))^(1/3)
    h_bar = mean_h(mesh)

    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)

    println()
    @printf("  --- lc=%.2f, N_swg=%d, h_bar=%.3f ---\n", lc, N, h_bar)
    @printf("      Mie refs: C_abs=%.4e  C_sca=%.4e  C_ext=%.4e\n",
            mie.C_abs, mie.C_sca, mie.C_ext)
    @printf("%8s %8s | %10s %7s | %10s %7s | %10s %7s\n",
            "duffy_n", "t_asm(s)",
            "C_abs", "err%", "C_sca", "err%", "C_ext", "err%")
    println("  ", "-"^90)
    flush(stdout)

    results = NamedTuple[]
    for duffy_n in (5, 7, 10)
        t = @elapsed D = solve_with_duffy_order(basis, k0, eps_p, eps_bg, k_hat, E0, duffy_n)
        scat = compute_scattering(basis, D;
                                   k_hat=k_hat, E0=E0, k0=k0,
                                   eps_p=eps_p, eps_bg=eps_bg,
                                   csca_method=:farfield, n_theta=20)
        ea = 100 * (scat.C_abs - mie.C_abs) / mie.C_abs
        es = 100 * (scat.C_sca - mie.C_sca) / mie.C_sca
        ex = 100 * (scat.C_ext - mie.C_ext) / mie.C_ext
        @printf("%8d %8.1f | %10.4e %+6.2f%% | %10.4e %+6.2f%% | %10.4e %+6.2f%%\n",
                duffy_n, t,
                scat.C_abs, ea, scat.C_sca, es, scat.C_ext, ex)
        flush(stdout)
        push!(results, (; duffy_n, C_abs=scat.C_abs, C_sca=scat.C_sca, C_ext=scat.C_ext))
    end

    # Deltas between duffy_n values on the same mesh
    if length(results) >= 2
        println()
        ref = results[end]  # duffy=10 as "converged" reference
        @printf("  Relative change from duffy=5 to duffy=10 (same mesh, same solve):\n")
        d5 = results[1]
        d_abs = (ref.C_abs - d5.C_abs) / d5.C_abs
        d_sca = (ref.C_sca - d5.C_sca) / d5.C_sca
        d_ext = (ref.C_ext - d5.C_ext) / d5.C_ext
        @printf("    dC_abs = %+7.3f%%   dC_sca = %+7.3f%%   dC_ext = %+7.3f%%\n",
                100 * d_abs, 100 * d_sca, 100 * d_ext)
        flush(stdout)
    end
end

println()
println("  Interpretation:")
println("    |delta| << 1%   -> Duffy 5-pt is converged; investigation #2 eliminated.")
println("    |delta| ~ 1%+   -> 5-pt is under-resolved; raise the default Duffy order.")
println("    |delta| >> 1%   -> Duffy quadrature is the dominant error source.")
flush(stdout)
