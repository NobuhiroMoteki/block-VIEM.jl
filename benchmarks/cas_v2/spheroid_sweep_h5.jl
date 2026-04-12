# CAS-v2 spheroid parameter sweep — writes a small HDF5 file in the
# block-DDA_Py-compatible schema (Phase 5.4).
#
# This is a *minimal* example sized for fast iteration: a 2 × 2 × 3 × 4 × 5
# grid takes a few minutes on a laptop. To produce a real sweep matching
# the recommended block-DDA_Py grid sizes (25 × 19 × 15 × 13 × 21), edit
# the constants in the SETTINGS section and budget hours/days of compute.
#
# Run with:
#     julia --project=. benchmarks/cas_v2/spheroid_sweep_h5.jl

using LinearAlgebra: norm
using Printf
using StaticArrays
using BlockVIEM
using BlockVIEM: Vec3
import Gmsh: gmsh

# ── SETTINGS ─────────────────────────────────────────────────────────────────
const OUT_FILE = joinpath(@__DIR__, "spheroid_sweep_viem_minimal.h5")

# Wavelengths (μm) and medium refractive index
const WAVELENGTHS = (0.638, 0.834)
const M_M         = 1.0

# Sweep grids (kept tiny for the example)
const D_VE_GRID            = collect(range(0.30, 0.50, length = 2))
const RI_REAL_GRID         = collect(range(1.50, 1.60, length = 2))
const LOG_AR_GRID          = collect(range(-0.4, 0.4, length = 3))   # AR=b/c
const COS_THETA_O_HALF     = collect(range(0.0, 1.0, length = 4))    # cos(β)
const PHI_O_GRID           = collect(range(0.0, π,   length = 5))    # = α grid

# Mesh size as a fraction of the smaller spheroid semi-axis
const LC_FACTOR = 0.30
# ──────────────────────────────────────────────────────────────────────────────

const N_Dve  = length(D_VE_GRID)
const N_RI   = length(RI_REAL_GRID)
const N_AR   = length(LOG_AR_GRID)
const N_u    = length(COS_THETA_O_HALF)
const N_ph   = length(PHI_O_GRID)

println("=" ^ 70)
println("CAS-v2 spheroid sweep → HDF5 (Phase 5.4)")
println("=" ^ 70)
@printf("  Grid: %d D_ve × %d RI × %d AR × %d cos_θ × %d φ_o\n",
        N_Dve, N_RI, N_AR, N_u, N_ph)
@printf("  Wavelengths: %s   m_m = %.3f\n", string(WAVELENGTHS), M_M)
@printf("  Output: %s\n", OUT_FILE)

function spheroid_mesh(b::Float64, c::Float64, lc::Float64)
    path = joinpath(tempdir(), "viem_sweep_$(b)_$(c)_$(lc).msh")
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sph")
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

function compute_one_wavelength(wl_0::Float64)
    k0 = 2π * M_M / wl_0

    S_theta = Array{ComplexF64}(undef, N_Dve, N_RI, N_AR, N_u, N_ph)
    S_phi   = similar(S_theta)
    converged = trues(N_Dve, N_RI, N_AR)

    for i in 1:N_Dve, j in 1:N_RI, k in 1:N_AR
        D_ve = D_VE_GRID[i]
        RI   = RI_REAL_GRID[j]
        AR   = 10.0 ^ LOG_AR_GRID[k]
        r_ve = D_ve / 2

        # Oblate (AR > 1) ⇒ b > c. b = r_ve·AR^(1/3), c = r_ve·AR^(-2/3).
        b = r_ve * AR ^ ( 1/3)
        c = r_ve * AR ^ (-2/3)
        lc = LC_FACTOR * min(b, c)

        m_p   = ComplexF64(RI)         # non-absorbing for the example sweep
        eps_p = m_p^2
        eps_bg = M_M^2

        @printf("  [%d,%d,%d] D_ve=%.3f RI=%.2f AR=%.3f → semi-axes (%.4f,%.4f,%.4f), lc=%.4f  ",
                i, j, k, D_ve, RI, AR, b, b, c, lc)
        flush(stdout)

        try
            mesh = read_msh(spheroid_mesh(b, c, lc))
            basis = build_swg_basis(mesh; include_boundary_faces = true)

            # Solve at α=0 for each cos(θ_o); apply analytical α-expansion.
            beta_list = [(0.0, acos(cos_th), 0.0) for cos_th in COS_THETA_O_HALF]
            t = @elapsed results = solve_cas_v2_orientations(
                basis, beta_list;
                k0 = k0, eps_p = eps_p, eps_bg = eps_bg,
                duffy_rule = duffy_reference_rule(7))
            @printf("N=%d  %.1fs\n", n_basis(basis), t)

            for m in 1:N_u
                Sth_a, Sph_a = expand_alpha_from_alpha0(
                    results[m].S_fw_theta, results[m].S_fw_phi, PHI_O_GRID)
                S_theta[i,j,k,m,:] .= Sth_a
                S_phi[i,j,k,m,:]   .= Sph_a
            end
        catch err
            @warn "Solve failed" wl_0 i j k err
            converged[i,j,k] = false
            S_theta[i,j,k,:,:] .= NaN + NaN*im
            S_phi[i,j,k,:,:]   .= NaN + NaN*im
        end
    end

    return SpheroidSweepData(wl_0, M_M, S_theta, S_phi, converged)
end

grids = SpheroidSweepGrids(D_VE_GRID, RI_REAL_GRID, LOG_AR_GRID,
                            COS_THETA_O_HALF, PHI_O_GRID)

data_per_wl = SpheroidSweepData[]
for wl in WAVELENGTHS
    println("\n--- λ = $wl μm ---")
    push!(data_per_wl, compute_one_wavelength(wl))
end

write_spheroid_sweep_h5(OUT_FILE, grids, data_per_wl;
                        block_viem_version = "0.1.1",
                        solver_tol = 1.0e-8,
                        extra_root_attrs = Dict("producer" => "block-VIEM.jl"))

println("\nWrote $(OUT_FILE)")
println("\nVerifying round-trip read…")
grids_r, data_r = read_spheroid_sweep_h5(OUT_FILE)
@printf("  %d wavelengths recovered: %s\n", length(data_r),
        string([d.wl_0 for d in data_r]))
@printf("  Grid sizes: %s\n", string((length(grids_r.D_ve), length(grids_r.RI_real),
                                       length(grids_r.log_AR),
                                       length(grids_r.cos_theta_o_half),
                                       length(grids_r.phi_o))))
println("Done.")
