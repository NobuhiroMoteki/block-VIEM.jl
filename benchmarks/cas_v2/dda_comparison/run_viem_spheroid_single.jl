# Run block-VIEM.jl on a single oblate spheroid at the same orientation
# used by `run_dda_spheroid_single.py` and dump the PCAS forward-scattering
# amplitudes as JSON.
#
# Usage:
#     julia --project=. benchmarks/cas_v2/dda_comparison/run_viem_spheroid_single.jl

using BlockVIEM
using BlockVIEM: Vec3
using StaticArrays
using LinearAlgebra: norm
using Printf
import Gmsh: gmsh

# ---------------------------------------------------------------------------
# Physical parameters — must match run_dda_spheroid_single.py exactly
# ---------------------------------------------------------------------------
const WL_0     = 0.638              # vacuum wavelength [um]
const M_M      = 1.0                # medium refractive index
const M_P      = 1.5 + 0.0im        # particle refractive index
const R_VE     = 0.20               # volume-equivalent radius [um]
const BC_RATIO = 3.0                # b / c  (> 1 ⇒ oblate)
const LC       = 0.035              # target mesh size [um]   (≈ r_ve/6)
const BETA_LIST = [π/4, π/2]         # tilt angles to compare against DDA

const OUTPUT = joinpath(@__DIR__, "viem_result.json")

# Oblate spheroid semi-axes for ab_ratio = 1, bc_ratio = AR:
#   V = (4π/3)·b²·c = (4π/3)·r_ve³  with  c = b/AR
# ⇒ b = r_ve·AR^(1/3),  c = r_ve·AR^(-2/3)
const B_EQ = R_VE * BC_RATIO ^ ( 1/3)
const C_AX = R_VE * BC_RATIO ^ (-2/3)

const EPS_P  = M_P^2
const EPS_BG = M_M^2
const K0     = 2π * M_M / WL_0

function spheroid_mesh(b::Float64, c::Float64, lc::Float64)
    path = joinpath(tempdir(), "viem_cmp_$(b)_$(c)_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("spheroid")
        sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, sph)], 0.0, 0.0, 0.0, b, b, c)
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

function _json_escape(s::AbstractString)
    return replace(s, "\\" => "\\\\", "\"" => "\\\"")
end

function write_json(path::AbstractString, params::Dict, orientations::Vector)
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"params\": {")
        entries = String[]
        for (k, v) in params
            if v isa AbstractString
                push!(entries, "    \"$(k)\": \"$(_json_escape(v))\"")
            elseif v isa Integer
                push!(entries, "    \"$(k)\": $(v)")
            elseif v isa AbstractFloat
                push!(entries, "    \"$(k)\": $(v)")
            else
                push!(entries, "    \"$(k)\": $(v)")
            end
        end
        println(io, join(entries, ",\n"))
        println(io, "  },")
        println(io, "  \"orientations\": [")
        ostrs = String[]
        for o in orientations
            entries = String[]
            for (k, v) in o
                push!(entries, "      \"$(k)\": $(v)")
            end
            push!(ostrs, "    {\n" * join(entries, ",\n") * "\n    }")
        end
        println(io, join(ostrs, ",\n"))
        println(io, "  ]")
        println(io, "}")
    end
    return path
end

println("=" ^ 70)
println("VIEM spheroid single-orientation comparison with block-DDA_Py")
println("=" ^ 70)
@printf("  r_ve = %.3f μm  bc_ratio = %.1f (oblate)\n", R_VE, BC_RATIO)
@printf("  semi-axes (a, b, c) = (%.4f, %.4f, %.4f) μm\n", B_EQ, B_EQ, C_AX)
@printf("  λ_0 = %.3f μm,  m_p = %s,  m_m = %.3f\n", WL_0, M_P, M_M)
@printf("  mesh lc = %.3f μm,  k0·r_ve = %.4f\n", LC, K0*R_VE)

# ---------------------------------------------------------------------------
# Build mesh and solve
# ---------------------------------------------------------------------------
mesh_path = spheroid_mesh(B_EQ, C_AX, LC)
mesh = read_msh(mesh_path)
basis = build_swg_basis(mesh; include_boundary_faces = true)
V_mesh = total_volume(mesh)
r_ve_mesh = (3 * V_mesh / (4π))^(1/3)
@printf("  mesh: %d tets, %d basis functions, r_ve_mesh = %.5f (err %.2f%%)\n",
        n_tets(mesh), n_basis(basis), r_ve_mesh,
        100*abs(r_ve_mesh - R_VE)/R_VE)

euler_list = [(0.0, β, 0.0) for β in BETA_LIST]
@printf("  Assembling Z and factorising…\n")
t_asm = @elapsed Z = assemble_impedance_matrix(basis; k0 = K0,
                              eps_p = EPS_P, eps_bg = EPS_BG,
                              duffy_rule = duffy_reference_rule(7),
                              symmetrize = true)
@printf("  Z assembly: %.1fs\n", t_asm)
import LinearAlgebra
t_lu = @elapsed F = LinearAlgebra.lu(Z)
@printf("  LU       : %.1fs\n", t_lu)

@printf("  Solving %d orientations + computing CAS + far-field…\n", length(euler_list))
results = []
scats   = []
t_solve = @elapsed for ea in euler_list
    ori = cas_orientation(ea...)
    bvec = project_plane_wave(basis; k_hat = ori.u_inc,
                              E0 = ori.e0_inc, k_bg = ComplexF64(K0))
    D = F \ bvec
    push!(results, compute_cas_observables(basis, D; orientation = ori,
                              k0 = K0, eps_p = EPS_P, eps_bg = EPS_BG))
    push!(scats, compute_scattering(basis, D;
                              k_hat = ori.u_inc, E0 = ori.e0_inc,
                              k0 = K0, eps_p = EPS_P, eps_bg = EPS_BG))
end
@printf("  per-orientation solve+observables: %.1fs (total)\n\n", t_solve)

params = [
    "wl_0"      => WL_0,
    "m_m"       => M_M,
    "m_p_re"    => real(M_P),
    "m_p_im"    => imag(M_P),
    "r_ve"      => R_VE,
    "r_ve_mesh" => r_ve_mesh,
    "bc_ratio"  => BC_RATIO,
    "lc"        => LC,
    "n_tets"    => n_tets(mesh),
    "n_basis"   => n_basis(basis),
]

orientations = Vector{Any}()
for (k, ea) in enumerate(euler_list)
    o = [
        "alpha"          => ea[1],
        "beta"           => ea[2],
        "gamma"          => ea[3],
        "S_fw_theta_re"  => real(results[k].S_fw_theta),
        "S_fw_theta_im"  => imag(results[k].S_fw_theta),
        "S_fw_phi_re"    => real(results[k].S_fw_phi),
        "S_fw_phi_im"    => imag(results[k].S_fw_phi),
        "S_fw_mean_re"   => real(results[k].S_fw_mean),
        "S_fw_mean_im"   => imag(results[k].S_fw_mean),
        "S_bk_re"        => real(results[k].S_bk),
        "S_bk_im"        => imag(results[k].S_bk),
        "C_ext"          => scats[k].C_ext,
        "C_abs"          => scats[k].C_abs,
        "C_sca"          => scats[k].C_sca,
    ]
    push!(orientations, o)
    @printf("  (α=%.4f β=%.4f γ=%.4f)\n", ea...)
    @printf("    S_fw_θ  = %+.6e %+.6ej\n",
            real(results[k].S_fw_theta), imag(results[k].S_fw_theta))
    @printf("    S_fw_φ  = %+.6e %+.6ej\n",
            real(results[k].S_fw_phi), imag(results[k].S_fw_phi))
    @printf("    C_ext   = %.6e\n", scats[k].C_ext)
    @printf("    C_abs   = %.6e\n", scats[k].C_abs)
    @printf("    C_sca   = %.6e\n", scats[k].C_sca)
end

write_json(OUTPUT, Dict(params), orientations)
println("\nWrote $(OUTPUT)")
