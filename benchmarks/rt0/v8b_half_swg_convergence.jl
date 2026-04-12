# V8b: half-SWG (K^A+K^B+K^C+K^D) convergence sweep lc = 0.7, 0.5, 0.35, 0.25.
#
# Runs dense assemble + LU solve at each lc and reports Mie errors. Fits
# a log-log convergence slope to estimate the observed order of accuracy.

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

const TEST_DIR = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST_DIR, "mie_reference.jl"))

function sphere_mesh(lc)
    path = joinpath(tempdir(), "sph_v8b_$(lc).msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("v8b_$(lc)")
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

function run_half_swg(mesh, k0, eps_p, eps_bg, k_hat, E0)
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    N = n_basis(basis)
    k_bg = ComplexF64(k0) * sqrt(ComplexF64(eps_bg))
    b = project_plane_wave(basis; k_hat=k_hat, E0=E0, k_bg=k_bg)
    t_asm = @elapsed Z = assemble_impedance_matrix(basis;
                                                     k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                                     symmetrize=true)
    t_sol = @elapsed D = Z \ b
    scat = compute_scattering(basis, D;
                                k_hat=k_hat, E0=E0, k0=k0,
                                eps_p=eps_p, eps_bg=eps_bg,
                                csca_method=:farfield, n_theta=20)
    return (; basis, N, t_asm, t_sol, scat)
end

const m_p    = 1.5 + 0.01im
const wl_0   = 10.0
const m_m    = 1.0
const eps_bg = m_m^2
const eps_p  = ComplexF64(m_p)^2
const k0     = 2π * m_m / wl_0
const k_hat  = Vec3(0, 0, 1)
const E0     = SVector{3, ComplexF64}(1.0, 0.0, 0.0)

println("=" ^ 116); flush(stdout)
println("  V8b: half-SWG dense convergence sweep (m=1.5+0.01i, wl=10, ka_ve~0.63)")
println("=" ^ 116); flush(stdout)

@printf("%6s %6s %7s %8s %8s | %10s %+7s %% | %10s %+7s %% | %10s %+7s %%\n",
        "lc", "N", "h_bar", "t_asm", "t_sol",
        "C_abs", "err", "C_sca", "err", "C_ext", "err")
println("-"^116); flush(stdout)

results = NamedTuple[]
for lc in (0.7, 0.5, 0.35, 0.25)
    path = sphere_mesh(lc)
    mesh = read_msh(path)
    V_mesh = total_volume(mesh)
    r_ve = (3V_mesh / (4π))^(1/3)
    h_bar = mean_h(mesh)
    mie = mie_cross_sections(; wl_0=wl_0, m_m=m_m, r_p=r_ve, m_p=m_p)
    r = run_half_swg(mesh, k0, eps_p, eps_bg, k_hat, E0)
    ea = (r.scat.C_abs - mie.C_abs) / mie.C_abs
    es = (r.scat.C_sca - mie.C_sca) / mie.C_sca
    ex = (r.scat.C_ext - mie.C_ext) / mie.C_ext
    @printf("%6.3f %6d %7.3f %8.2f %8.2f | %10.4e %+7.3f %% | %10.4e %+7.3f %% | %10.4e %+7.3f %%\n",
            lc, r.N, h_bar, r.t_asm, r.t_sol,
            r.scat.C_abs, 100*ea, r.scat.C_sca, 100*es, r.scat.C_ext, 100*ex)
    flush(stdout)
    push!(results, (; lc, N=r.N, h=h_bar, ea, es, ex))
end

println()
println("  Convergence slopes (log|err| vs log h):")
for (name, getter) in (("C_abs", r -> abs(r.ea)),
                        ("C_sca", r -> abs(r.es)),
                        ("C_ext", r -> abs(r.ex)))
    errs = [getter(r) for r in results]
    hs = [r.h for r in results]
    rates = Float64[]
    for i in 2:length(errs)
        if errs[i-1] > 0 && errs[i] > 0
            push!(rates, log(errs[i]/errs[i-1]) / log(hs[i]/hs[i-1]))
        end
    end
    mean_r = isempty(rates) ? 0.0 : sum(rates)/length(rates)
    rates_str = join([@sprintf("%5.2f", r) for r in rates], ", ")
    @printf("    %-6s: p = [%s]  avg=%.2f\n", name, rates_str, mean_r)
end
flush(stdout)
