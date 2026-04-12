# V8d: post-processing rule sweep at lc=0.35. The plateau in v8c was
# traced to `compute_scattering`'s tet rule for far-field / mass-matrix
# integration. Here we vary only that rule while keeping Z fixed at
# high-order quadrature.

using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

include(joinpath(@__DIR__, "..", "..", "test", "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v8d_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v8d_$(lc)")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.write(path); gmsh.finalize()
    return path
end

const m_p   = 1.5 + 0.01im
const wl_0  = 10.0
const k0    = 2π / wl_0
const k_hat = Vec3(0, 0, 1)
const E0    = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

path = sphere_mesh(0.35)
mesh = read_msh(path)
r_ve = (3 * total_volume(mesh) / (4π))^(1/3)
mie  = mie_cross_sections(; wl_0=wl_0, m_m=1.0, r_p=r_ve, m_p=m_p)

basis = build_swg_basis(mesh; include_boundary_faces = true)
println("lc=0.35, N=", n_basis(basis), ", r_ve=", round(r_ve, digits=4))
flush(stdout)

k_bg = ComplexF64(k0) * sqrt(ComplexF64(1.0))
b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg)

println("Assembling Z with HQ rules...")
flush(stdout)
Z = assemble_impedance_matrix(basis;
                                k0=k0, eps_p=ComplexF64(m_p)^2, eps_bg=1.0,
                                outer_rule=TET_QUAD_125PT,
                                duffy_rule=duffy_reference_rule(7),
                                tri_rule=tri_collapsed_rule(6),
                                tri_duffy_rule=tri_duffy_reference_rule(8),
                                symmetrize=true)
D = Z \ b
println("Solved. Running post-processing sweep...")
flush(stdout)

for (label, rule) in (("5PT",   TET_QUAD_5PT),
                       ("64PT",  TET_QUAD_64PT),
                       ("125PT", TET_QUAD_125PT))
    scat = compute_scattering(basis, D;
                                k_hat=k_hat, E0=E0, k0=k0,
                                eps_p=ComplexF64(m_p)^2, eps_bg=1.0,
                                rule=rule, csca_method=:farfield, n_theta=40)
    ea = 100 * (scat.C_abs - mie.C_abs) / mie.C_abs
    es = 100 * (scat.C_sca - mie.C_sca) / mie.C_sca
    ex = 100 * (scat.C_ext - mie.C_ext) / mie.C_ext
    @printf("  rule=%-5s  C_abs err=%+.4f%%  C_sca err=%+.4f%%  C_ext err=%+.4f%%\n",
            label, ea, es, ex)
    flush(stdout)
end
