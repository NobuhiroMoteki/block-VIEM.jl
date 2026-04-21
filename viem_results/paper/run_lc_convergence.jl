# lc-convergence study for the paper (CLAUDE.md §4).
#
# For one (shape, material) combination at a_eq = 0.1 μm and a single
# representative orientation, build the tet mesh with five lc values
# (factor ∈ {1.5, 1.0, 0.7, 0.5, 0.35} × adaptive_lc) and record the
# converged observables (Q_ext, Q_sca, Q_abs, S_fw_θ, S_fw_φ, S_bk) plus
# the per-mesh setup / solve wall time.  Results are written to
# viem_results/paper/convergence_{shape}_{material}.hdf5.
#
# Usage:
#   julia --project=. -t auto viem_results/paper/run_lc_convergence.jl \
#         <shape> <material>
#
#   shape    ∈ {sphere, oblate, gre}
#   material ∈ {n15, n317, Au}

using BlockVIEM
using BlockVIEM: GREParams, gre_mesh, adaptive_lc,
                 build_swg_basis, n_basis, n_tets, mean_edge_length,
                 cas_orientation, solve_cas_v2_orientations,
                 compute_scattering, duffy_reference_rule,
                 aim_grid, build_aim_projection, assemble_mass_matrix
using HDF5
using Printf
using Random
using Dates

include(joinpath(@__DIR__, "_common.jl"))
include(joinpath(dirname(@__DIR__), "..", "test", "mie_reference.jl"))

# ──────────────────────────────────────────────────────────────────────
#  Settings (mirror viem_results/run_viem.jl)
# ──────────────────────────────────────────────────────────────────────
const RNG_SEED        = 12345
const SOLVER_TOL      = 1e-5
const MAXITER         = 100
const N_PW            = 10
const DUFFY_ORDER     = 5
const AIM_PITCH_RATIO = 0.5
const AIM_PADDING     = 4
const A_EQ_CONV       = 0.1                       # CLAUDE.md §4
const LC_FACTORS      = [1.5, 1.0, 0.7, 0.5, 0.35]
const SINGLE_ORIENT   = (0.0, 0.0, 0.0)           # ZYZ, identity rotation

# ──────────────────────────────────────────────────────────────────────
#  Shape / material dispatch
# ──────────────────────────────────────────────────────────────────────
function shape_params(name::AbstractString)
    name == "sphere" && return (bc_ratio=1.0, ab_ratio=1.0, gre_beta=0.0)
    name == "oblate" && return (bc_ratio=3.0, ab_ratio=1.0, gre_beta=0.0)
    name == "gre"    && return (bc_ratio=1.0, ab_ratio=1.0, gre_beta=0.2)
    error("unknown shape '$name' (expected: sphere | oblate | gre)")
end

function material_m_p(name::AbstractString)
    name == "n15"  && return N_LOW
    name == "n317" && return N_HIGH
    name == "Au"   && return N_AU
    error("unknown material '$name' (expected: n15 | n317 | Au)")
end

# ──────────────────────────────────────────────────────────────────────
#  Per-lc-point solve
# ──────────────────────────────────────────────────────────────────────
function solve_one(p::GREParams, lc_value, m_p_xyz; wl_0=WL_PAPER, m_m=M_M_PAPER)
    rng = Random.MersenneTwister(RNG_SEED)

    t_setup = @elapsed begin
        mesh, r_ve = gre_mesh(p, rng; lc=lc_value)
        basis = build_swg_basis(mesh; include_boundary_faces=true)
        h_bar = mean_edge_length(mesh)
        pitch = AIM_PITCH_RATIO * h_bar
        grid       = aim_grid(basis.mesh; pitch=pitch, padding=AIM_PADDING)
        projection = build_aim_projection(basis, grid; poly_order=2, stencil=3)
        mass       = assemble_mass_matrix(basis)
    end

    euler_list = [SINGLE_ORIENT]
    t_solve = @elapsed begin
        cas_results, D_block = solve_cas_v2_orientations(
            basis, euler_list;
            wl_0=wl_0, m_m=m_m, m_p=m_p_xyz,
            method=:aim_gmres, tol=SOLVER_TOL, maxiter=MAXITER,
            verbose=false, return_D=true,
            duffy_rule=duffy_reference_rule(DUFFY_ORDER),
            symmetrize=true,
            pitch=pitch, padding=AIM_PADDING,
            projection=projection, mass=mass)

        ori = cas_orientation(SINGLE_ORIENT...)
        k0 = ComplexF64(2π * m_m / wl_0)
        eps_p  = ComplexF64.(m_p_xyz) .^ 2
        eps_bg = ComplexF64(m_m)^2
        sr = compute_scattering(basis, @view(D_block[:, 1]);
                                k_hat=ori.u_inc, E0=ori.e0_inc,
                                k0=k0, eps_p=eps_p, eps_bg=eps_bg,
                                csca_method=:farfield)
    end

    return (
        n_tet      = n_tets(mesh),
        n_dof      = n_basis(basis),
        r_ve       = r_ve,
        h_bar      = h_bar,
        t_setup    = t_setup,
        t_solve    = t_solve,
        C_abs      = sr.C_abs,
        C_ext      = sr.C_ext,
        C_sca      = sr.C_ext - sr.C_abs,
        S_fw_theta = cas_results[1].S_fw_theta,
        S_fw_phi   = cas_results[1].S_fw_phi,
        S_bk       = cas_results[1].S_bk,
    )
end

# ──────────────────────────────────────────────────────────────────────
#  Mie reference for the volume-equivalent sphere
# ──────────────────────────────────────────────────────────────────────
function mie_ref(m_p_xyz)
    m_p_avg = ComplexF64(sum(m_p_xyz) / length(m_p_xyz))
    cs  = mie_cross_sections(; wl_0=WL_PAPER, m_m=M_M_PAPER,
                             r_p=A_EQ_CONV, m_p=m_p_avg)
    cas = mie_cas_observables(; wl_0=WL_PAPER, m_m=M_M_PAPER,
                              r_p=A_EQ_CONV, m_p=m_p_avg)
    return (
        C_abs   = cs.C_abs,
        C_ext   = cs.C_ext,
        C_sca   = cs.C_ext - cs.C_abs,
        S_fw    = cas.S_fw_mean,
        S_bk    = cas.S_bk,
    )
end

# ──────────────────────────────────────────────────────────────────────
#  HDF5 writer
# ──────────────────────────────────────────────────────────────────────
function write_convergence_h5(path, shape, material, m_p_xyz, p, results, mie)
    n = length(LC_FACTORS)
    geom_area = π * A_EQ_CONV^2

    h5open(path, "w") do f
        g = create_group(f, "target")
        attrs(g)["shape"]      = shape
        attrs(g)["material"]   = material
        attrs(g)["wl_0_um"]    = WL_PAPER
        attrs(g)["m_m"]        = M_M_PAPER
        attrs(g)["a_eq_um"]    = A_EQ_CONV
        attrs(g)["bc_ratio"]   = p.bc_ratio
        attrs(g)["ab_ratio"]   = p.ab_ratio
        attrs(g)["gre_beta"]   = p.beta
        attrs(g)["n_lc_points"]= n
        attrs(g)["units"]      = "C:[um^2], S:[um], lc:[um], t:[s]"
        attrs(g)["orientation"]= "ZYZ Euler (0,0,0): incidence +z, e0_inc +x"

        write_dataset(g, "m_p_xyz", ComplexF64.(m_p_xyz))

        c = create_group(g, "lc_convergence")
        write_dataset(c, "lc_factor", Float64.(LC_FACTORS))
        write_dataset(c, "lc",        Float64[r.h_bar for r in results])
        write_dataset(c, "n_tet",     Int.([r.n_tet for r in results]))
        write_dataset(c, "n_dof",     Int.([r.n_dof for r in results]))
        write_dataset(c, "r_ve",      Float64[r.r_ve   for r in results])
        write_dataset(c, "t_setup",   Float64[r.t_setup for r in results])
        write_dataset(c, "t_solve",   Float64[r.t_solve for r in results])
        write_dataset(c, "C_abs",     Float64[r.C_abs   for r in results])
        write_dataset(c, "C_ext",     Float64[r.C_ext   for r in results])
        write_dataset(c, "C_sca",     Float64[r.C_sca   for r in results])
        write_dataset(c, "Q_abs",     Float64[r.C_abs / geom_area for r in results])
        write_dataset(c, "Q_ext",     Float64[r.C_ext / geom_area for r in results])
        write_dataset(c, "Q_sca",     Float64[r.C_sca / geom_area for r in results])
        write_dataset(c, "S_fw_theta", ComplexF64[r.S_fw_theta for r in results])
        write_dataset(c, "S_fw_phi",   ComplexF64[r.S_fw_phi   for r in results])
        write_dataset(c, "S_bk",       ComplexF64[r.S_bk       for r in results])

        r = create_group(g, "reference")
        attrs(r)["definition"] = "Mie of volume-equivalent sphere (radius a_eq); " *
                                 "exact for shape=sphere, approximation otherwise"
        write_dataset(r, "C_abs_mie",  mie.C_abs)
        write_dataset(r, "C_ext_mie",  mie.C_ext)
        write_dataset(r, "C_sca_mie",  mie.C_sca)
        write_dataset(r, "Q_abs_mie",  mie.C_abs / geom_area)
        write_dataset(r, "Q_ext_mie",  mie.C_ext / geom_area)
        write_dataset(r, "Q_sca_mie",  mie.C_sca / geom_area)
        write_dataset(r, "S_fw_mie",   mie.S_fw)
        write_dataset(r, "S_bk_mie",   mie.S_bk)
    end
end

# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
function main()
    length(ARGS) == 2 || error("Usage: julia --project=. " *
        "viem_results/paper/run_lc_convergence.jl <shape> <material>\n" *
        "  shape    ∈ {sphere, oblate, gre}\n" *
        "  material ∈ {n15, n317, Au}")

    shape    = ARGS[1]
    material = ARGS[2]

    sp = shape_params(shape)
    m_p = material_m_p(material)
    m_p_xyz = ComplexF64[m_p, m_p, m_p]

    p = GREParams(A_EQ_CONV, sp.bc_ratio, sp.ab_ratio, sp.gre_beta)

    # Reference adaptive_lc to scale by factor
    lc_base = adaptive_lc(p;
                          wl_0=WL_PAPER,
                          m_p_max=maximum(abs.(m_p_xyz)),
                          N_pw=N_PW)
    println("[$(Dates.format(now(), "HH:MM:SS"))] " *
            "lc convergence: shape=$shape  material=$material")
    println("  m_p     = $m_p_xyz")
    println("  a_eq    = $A_EQ_CONV μm")
    println("  bc=$(sp.bc_ratio) ab=$(sp.ab_ratio) β_gre=$(sp.gre_beta)")
    @printf("  lc_base = %.5f μm  (adaptive)\n", lc_base)
    println("  factors = $LC_FACTORS")
    println("─"^96)
    @printf("%6s %10s %8s %8s %10s %10s %12s %12s\n",
            "factor", "lc[μm]", "N_tet", "N_DOF",
            "t_setup[s]", "t_solve[s]", "C_ext[μm²]", "C_abs[μm²]")
    println("─"^96)

    results = NamedTuple[]
    for factor in LC_FACTORS
        lc_val = factor * lc_base
        r = solve_one(p, lc_val, m_p_xyz)
        push!(results, r)
        @printf("%6.2f %10.5f %8d %8d %10.1f %10.1f %12.4e %12.4e\n",
                factor, lc_val, r.n_tet, r.n_dof,
                r.t_setup, r.t_solve, r.C_ext, r.C_abs)
        flush(stdout)
    end
    println("─"^96)

    mie = mie_ref(m_p_xyz)
    out = joinpath(@__DIR__, "convergence_$(shape)_$(material).hdf5")
    write_convergence_h5(out, shape, material, m_p_xyz, p, results, mie)
    println("[$(Dates.format(now(), "HH:MM:SS"))] Wrote $out")
    @printf("  Mie reference: C_ext=%.4e  C_abs=%.4e  μm²\n", mie.C_ext, mie.C_abs)
end

main()
