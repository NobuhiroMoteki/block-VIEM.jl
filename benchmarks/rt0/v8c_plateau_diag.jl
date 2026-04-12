# V8c: diagnose the ~0.2% plateau seen in v8b.
#
# Fix lc = 0.35 (where v8b gave C_abs err -0.03%, C_sca err +0.28%) and
# vary:
#   - far-field angular quadrature order n_theta ∈ {20, 40, 80}
#   - tri / duffy quadrature orders used by K^B/K^C/K^D
#
# If the plateau is from far-field quadrature, C_sca/C_ext will drop
# when n_theta increases while C_abs (volumetric Joule) stays put.
# If the plateau is from impedance quadrature, higher-order tri/duffy
# rules will drop ALL three errors uniformly.
# If the plateau is geometric (mesh faceting), nothing will change.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v8c_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v8c_$(lc)")
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

const m_p    = 1.5 + 0.01im
const wl_0   = 10.0
const m_m    = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0     = 2π * m_m / wl_0
const k_hat  = Vec3(0, 0, 1)
const E0     = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

lc = 0.35
path = sphere_mesh(lc)
mesh = read_msh(path)
V_mesh = total_volume(mesh)
r_ve = (3V_mesh / (4π))^(1/3)
mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)

basis = build_swg_basis(mesh; include_boundary_faces = true)
N = n_basis(basis)

println("=" ^ 110); flush(stdout)
println("  V8c: plateau diagnosis at lc=$lc, N=$N, r_ve=$(round(r_ve, digits=4))")
println("  Reference Mie: C_abs=$(mie.C_abs), C_sca=$(mie.C_sca), C_ext=$(mie.C_ext)")
println("=" ^ 110); flush(stdout)

# Build Z once with default quadrature and again with high order; reuse
# between n_theta values.
function solve_and_score(basis, mie; outer_rule, duffy_rule, tri_rule, tri_duffy_rule, n_theta)
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
    b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg)
    Z = assemble_impedance_matrix(basis;
                                    k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                    outer_rule=outer_rule,
                                    duffy_rule=duffy_rule,
                                    tri_rule=tri_rule,
                                    tri_duffy_rule=tri_duffy_rule,
                                    symmetrize=true)
    D = Z \ b
    scat = compute_scattering(basis, D;
                                k_hat=k_hat, E0=E0, k0=k0,
                                eps_p=eps_p, eps_bg=eps_bg,
                                csca_method=:farfield, n_theta=n_theta)
    ea = 100 * (scat.C_abs - mie.C_abs) / mie.C_abs
    es = 100 * (scat.C_sca - mie.C_sca) / mie.C_sca
    ex = 100 * (scat.C_ext - mie.C_ext) / mie.C_ext
    return (; scat, ea, es, ex)
end

# Default rules — same as v8b (these gave err~0.2%)
default_Z = (outer_rule = TET_QUAD_5PT,
             duffy_rule = duffy_reference_rule(5),
             tri_rule = tri_collapsed_rule(4),
             tri_duffy_rule = tri_duffy_reference_rule(6))

# High-order rules
hi_Z = (outer_rule = TET_QUAD_125PT,
        duffy_rule = duffy_reference_rule(7),
        tri_rule = tri_collapsed_rule(6),
        tri_duffy_rule = tri_duffy_reference_rule(8))

println()
println("  (a) Default Z quadrature, vary n_theta:")
@printf("    %8s | %+7s %% | %+7s %% | %+7s %%\n", "n_theta", "C_abs", "C_sca", "C_ext")
println("    " * "-"^52)
for nt in (20, 40, 80)
    r = solve_and_score(basis, mie;
                         default_Z...,
                         n_theta=nt)
    @printf("    %8d | %+7.3f   | %+7.3f   | %+7.3f\n", nt, r.ea, r.es, r.ex)
    flush(stdout)
end

println()
println("  (b) High-order Z quadrature, n_theta=40:")
r_hi = solve_and_score(basis, mie; hi_Z..., n_theta=40)
@printf("    %8s | %+7s %% | %+7s %% | %+7s %%\n", "config", "C_abs", "C_sca", "C_ext")
println("    " * "-"^52)
@printf("    %8s | %+7.3f   | %+7.3f   | %+7.3f\n", "HQ-40", r_hi.ea, r_hi.es, r_hi.ex)
flush(stdout)
