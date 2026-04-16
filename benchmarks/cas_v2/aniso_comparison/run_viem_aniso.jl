# Cross-validate block-VIEM.jl anisotropic ε_p against block-DDA_Py.
#
# Usage:
#     julia --project=. benchmarks/cas_v2/aniso_comparison/run_viem_aniso.jl

using BlockVIEM
using BlockVIEM: Vec3
using StaticArrays
using Printf
import Gmsh: gmsh
import JSON3

const DDA_FILE = joinpath(@__DIR__, "dda_aniso_results.json")
isfile(DDA_FILE) || error("Run run_dda_aniso.py first")

dda = JSON3.read(read(DDA_FILE, String))
const WL_0 = Float64(dda.wl_0)
const M_M  = Float64(dda.m_m)
const R_VE = Float64(dda.r_ve)

# Build sphere mesh (same for all cases)
function sphere_mesh(r, lc)
    path = joinpath(tempdir(), "aniso_sphere.msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere")
        sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, r)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [sph], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

println("=" ^ 70)
println("VIEM–DDA anisotropic cross-validation")
println("=" ^ 70)
@printf("  wl_0=%.3f, m_m=%.3f, r_ve=%.3f\n", WL_0, M_M, R_VE)

mesh_path = sphere_mesh(R_VE, 0.035)
mesh  = read_msh(mesh_path)
basis = build_swg_basis(mesh; include_boundary_faces=true)
r_ve_mesh = (3 * total_volume(mesh) / (4π))^(1/3)
@printf("  mesh: %d tets, %d basis, r_ve=%.5f\n", n_tets(mesh), n_basis(basis), r_ve_mesh)

struct Row
    case_name::String
    beta_ori::Float64
    Sth_dda::ComplexF64; Sth_viem::ComplexF64; err_Sth::Float64
    Sph_dda::ComplexF64; Sph_viem::ComplexF64; err_Sph::Float64
end
rows = Row[]

for case_json in dda.cases
    name = String(case_json.name)
    mp_re = Float64.(case_json.m_p_xyz_re)
    mp_im = Float64.(case_json.m_p_xyz_im)
    m_p = [complex(mp_re[i], mp_im[i]) for i in 1:3]

    println("\n── Case $name: m_p = $m_p ──")

    euler_list = [(0.0, Float64(o.beta_ori), 0.0) for o in case_json.orientations]

    results = solve_cas_v2_orientations(basis, euler_list;
                wl_0=WL_0, m_m=M_M, m_p=m_p, method=:dense, verbose=false)

    for (k, o) in enumerate(case_json.orientations)
        Sth_d = ComplexF64(o.S_fw_theta_re + o.S_fw_theta_im * im)
        Sph_d = ComplexF64(o.S_fw_phi_re   + o.S_fw_phi_im   * im)
        Sth_v = results[k].S_fw_theta
        Sph_v = results[k].S_fw_phi

        err_th = abs(Sth_v - Sth_d) / max(abs(Sth_d), 1e-30)
        err_ph = abs(Sph_v - Sph_d) / max(abs(Sph_d), 1e-30)

        @printf("  β=%.4f  |ΔS_θ|=%.2f%%  |ΔS_φ|=%.2f%%\n",
                Float64(o.beta_ori), 100*err_th, 100*err_ph)
        @printf("    DDA  S_θ=%+.5e%+.5ej  S_φ=%+.5e%+.5ej\n",
                real(Sth_d), imag(Sth_d), real(Sph_d), imag(Sph_d))
        @printf("    VIEM S_θ=%+.5e%+.5ej  S_φ=%+.5e%+.5ej\n",
                real(Sth_v), imag(Sth_v), real(Sph_v), imag(Sph_v))

        push!(rows, Row(name, Float64(o.beta_ori),
                        Sth_d, Sth_v, err_th, Sph_d, Sph_v, err_ph))
    end
end

println("\n" * "=" ^ 70)
println("Summary")
println("=" ^ 70)
@printf("%-7s %6s  %8s %8s\n", "Case", "β_ori", "|ΔS_θ|%", "|ΔS_φ|%")
println("-" ^ 40)
for r in rows
    @printf("%-7s %6.3f  %7.2f%%  %7.2f%%\n",
            r.case_name, r.beta_ori, 100*r.err_Sth, 100*r.err_Sph)
end
