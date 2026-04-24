# Shared HDF5 schema writer for the paper-production sweeps.
#
# This is a thin wrapper around the schema in viem_results/create_h5.jl,
# parameterised by (shape, material, size list) so the per-(shape,material)
# scripts under viem_results/paper/ stay short and declarative.
#
# Schema is bit-for-bit compatible with create_h5.jl and run_viem.jl.

using HDF5

# ──────────────────────────────────────────────────────────────────────
#  Material constants  (all at λ₀ = 0.638 μm; CLAUDE.md §2)
# ──────────────────────────────────────────────────────────────────────
const N_LOW   = 1.5  + 0.01im                # low-index dielectric (n15)
const N_20    = 2.0  + 0.0im                 # mid-index non-absorbing (n20, paper "high" since v0.7.5)
const N_HIGH  = 3.17 + 0.16im                # legacy high-index (n317, kept as reference; v0.7.4 and earlier paper material)
const N_AU    = 0.17525 + 3.4830im           # Au, Johnson & Christy 1972 @ 0.638 μm

const WL_PAPER     = 0.638                   # vacuum wavelength [μm]
const M_M_PAPER    = 1.0                     # vacuum background

# Volume-equivalent radii [μm]
const A_EQ_FULL    = [0.05, 0.1, 0.2, 0.4]   # n15 / n20 (a_eq=0.5 dropped
                                              # to keep wall-time per slot manageable;
                                              # originally set for the heavier n317 cases)
const A_EQ_AU      = [0.05, 0.1, 0.2]        # Au only (≤ 0.2 μm); a_eq=0.5 μm
                                              # dropped — at |m_p|≈3.49 the
                                              # wavelength-driven lc ≈ 0.018 μm
                                              # yields prohibitive N_DOF
                                              # (>24 h per slot).  See CLAUDE.md §2.

# Multi-orientation grid (CLAUDE.md §6: max ≈ 100 simultaneous RHS)
# Spheroid mode (sphere, oblate): block solver gets L = N_β = 5.
# GRE mode: block solver gets L = N_α × N_β × N_γ = 100.
const N_ALPHA_DEFAULT = 4
const N_BETA_DEFAULT  = 5
const N_GAMMA_DEFAULT = 5

# Residual-history length for /target/cost/residual_history (CLAUDE.md §7.5 /
# block-DDA_Py parity).  Must match the MAXITER used by run_viem.jl and
# friends; slots that converge earlier are NaN-padded.
const MAXITER_HISTORY = 100

# ──────────────────────────────────────────────────────────────────────
#  Schema writer
# ──────────────────────────────────────────────────────────────────────
"""
    create_paper_h5(filename;
                    m_p, a_eq_list, bc_ratio, ab_ratio, gre_beta,
                    shape_kind = "gre",
                    N_alpha=N_ALPHA_DEFAULT,
                    N_beta=N_BETA_DEFAULT,
                    N_gamma=N_GAMMA_DEFAULT,
                    light_source="(paper)",
                    polarization="left-handed circular: E0_θ=1/√2, E0_φ=i/√2",
                    overwrite=false)

Write a block-DDA_Py-compatible HDF5 sweep file for the paper run defined by
`(m_p, a_eq_list)` and the shape parameters.  The wavelength is fixed at
0.638 μm (vacuum background n_m = 1.0).

`shape_kind ∈ {"gre", "doublet"}`:
* `"gre"` — `bc_ratio, ab_ratio, gre_beta` describe the GRE family.
* `"doublet"` — touching equal-sphere doublet, monomer radius
  `R = a_eq / 2^(1/3)`, gap `= 0.1 R` along the doublet axis (CLAUDE.md §2).
  `bc_ratio, ab_ratio, gre_beta` are written for schema parity but ignored
  by the runner.

`m_p` is a scalar `ComplexF64` (isotropic) or a 3-tuple/3-vector for
anisotropic permittivity along principal axes.
"""
function create_paper_h5(filename::AbstractString;
                         m_p,
                         a_eq_list::AbstractVector,
                         bc_ratio::Real,
                         ab_ratio::Real,
                         gre_beta::Real,
                         shape_kind::AbstractString = "gre",
                         N_alpha::Int = N_ALPHA_DEFAULT,
                         N_beta::Int  = N_BETA_DEFAULT,
                         N_gamma::Int = N_GAMMA_DEFAULT,
                         light_source::AbstractString = "(paper run)",
                         polarization::AbstractString =
                             "left-handed circular: E0_theta=1/sqrt(2), E0_phi=1j/sqrt(2)",
                         overwrite::Bool = false)
    shape_kind in ("gre", "doublet") ||
        error("shape_kind must be \"gre\" or \"doublet\", got \"$shape_kind\"")

    if isfile(filename) && !overwrite
        error("$filename already exists.  Pass overwrite=true to replace it.")
    end

    # Shape-up the inputs into the create_h5.jl conventions:
    #   wl_m_m_pairs : (1, 2) — single λ point
    #   m_p_xyz_list : (1, 3) — single (anisotropic-vector) entry
    wl_m_m_pairs = reshape(Float64[WL_PAPER, M_M_PAPER], 1, 2)

    if m_p isa Number
        m_p_vec = ComplexF64[m_p, m_p, m_p]
    else
        length(m_p) == 3 || error("m_p must be scalar or length-3 vector")
        m_p_vec = ComplexF64.(collect(m_p))
    end
    m_p_xyz_list = reshape(m_p_vec, 1, 3)

    r_v_base_list = Float64.(collect(a_eq_list))
    bc_ratio_list = Float64[bc_ratio]
    ab_ratio_list = Float64[ab_ratio]
    gre_beta_list = Float64[gre_beta]

    num_orientations = N_alpha * N_beta * N_gamma
    # Spheroid α-expansion applies to any axially symmetric particle:
    # GRE family with ab_ratio = 1 ∧ gre_beta = 0, OR a doublet placed
    # along the particle z-axis (cylindrical symmetry, see CLAUDE.md §2).
    spheroid_mode    = shape_kind == "doublet" ||
                       ((ab_ratio == 1.0) && (gre_beta == 0.0))

    N_pairs = size(wl_m_m_pairs, 1)
    N_m_p   = size(m_p_xyz_list, 1)
    N_rv    = length(r_v_base_list)
    N_bc    = length(bc_ratio_list)
    N_ab    = length(ab_ratio_list)
    N_bt    = length(gre_beta_list)

    shape_geo  = (N_rv, N_bc, N_ab, N_bt)
    shape_cond = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt)
    shape_ori  = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, num_orientations)
    shape_ang  = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, num_orientations, 3)

    h5open(filename, "w") do f
        grp = create_group(f, "target")
        attrs(grp)["light_source"]       = light_source
        attrs(grp)["polarization_state"] = polarization
        attrs(grp)["shape_kind"]         = shape_kind
        attrs(grp)["N_alpha_ori"]        = N_alpha
        attrs(grp)["N_beta_ori"]         = N_beta
        attrs(grp)["N_gamma_ori"]        = N_gamma
        attrs(grp)["num_orientations"]   = num_orientations
        attrs(grp)["spheroid_mode"]      = spheroid_mode ? 1 : 0

        write_dataset(grp, "wl_m_m_pairs",  wl_m_m_pairs)
        write_dataset(grp, "m_p_xyz_list",  m_p_xyz_list)
        write_dataset(grp, "r_v_base_list", r_v_base_list)
        write_dataset(grp, "bc_ratio_list", bc_ratio_list)
        write_dataset(grp, "ab_ratio_list", ab_ratio_list)
        write_dataset(grp, "gre_beta_list", gre_beta_list)

        sd = create_group(grp, "simulated_data")
        attrs(sd)["scattering_code"] = "block-VIEM.jl (volume integral equation method)"
        attrs(sd)["orientation"]     =
            "Deterministic ZYZ Euler angle grid (alpha, beta, gamma). " *
            "alpha: equally spaced in [0,2pi), N_alpha divisions. " *
            "beta: cos(beta) equally spaced in (-1,1), N_beta equal-area divisions. " *
            "gamma: equally spaced in [0,2pi), N_gamma divisions. " *
            "Ordering: alpha slowest, gamma fastest. " *
            "Spheroid mode (ab_ratio=1, gre_beta=0): VIEM solves only N_beta " *
            "orientations; full grid filled analytically."
        attrs(sd)["units"]        = "r_ve:[um], euler_angles:[rad], C:[um^2], S:[um]"
        attrs(sd)["S_definition"] = "S(0)_theta = S11(0)+1j*S12(0), " *
                                    "S(0)_phi = S22(0)-1j*S21(0), " *
                                    "S(180) = (S11+S22+1j*S12-1j*S21)(180)/sqrt(2), " *
                                    "per Mishchenko 2000"

        function ds(parent, name, shape, dtype, definition)
            d = create_dataset(parent, name, dtype, shape)
            attrs(d)["definition"] = definition
            return d
        end

        ds(sd, "r_ve",            shape_geo,  Float64,
           "volume-equivalent radius [um] from discretised particle volume")
        ds(sd, "Euler_angles",    shape_ang,  Float64,
           "ZYZ Euler angles (alpha,beta,gamma) rotating particle frame from lab frame [rad]")
        ds(sd, "C_abs",           shape_ori,  Float64,
           "absorption cross section per orientation [um^2]")
        ds(sd, "C_ext",           shape_ori,  Float64,
           "extinction cross section per orientation [um^2]")
        ds(sd, "S_fw_PCAS_theta", shape_ori,  ComplexF64,
           "forward-scattering amplitude S(0)_theta per orientation [um]")
        ds(sd, "S_fw_PCAS_phi",   shape_ori,  ComplexF64,
           "forward-scattering amplitude S(0)_phi per orientation [um]")
        ds(sd, "S_bk_OCBS",       shape_ori,  ComplexF64,
           "backward-scattering amplitude S(180) per orientation [um]")
        ds(sd, "C_abs_mie",       shape_cond, Float64,
           "Mie C_abs of volume-equivalent sphere [um^2]")
        ds(sd, "C_ext_mie",       shape_cond, Float64,
           "Mie C_ext of volume-equivalent sphere [um^2]")
        ds(sd, "S_fw_PCAS_mie",   shape_cond, ComplexF64,
           "Mie S(0) of volume-equivalent sphere [um]")
        ds(sd, "S_bk_OCBS_mie",   shape_cond, ComplexF64,
           "Mie S(180) of volume-equivalent sphere [um]")

        # ── /target/cost/ ───────────────────────────────────────────────
        # Per-slot cost + solver diagnostics. Schema is bit-for-bit
        # symmetric with block-DDA_Py's `/target/cost/`, modulo three
        # VIEM-specific renames for physically corresponding fields:
        #   DDA  n_cuboid    ↔  VIEM  n_tet
        #   DDA  n_occ       ↔  VIEM  n_dof
        #   DDA  lattice_lf  ↔  VIEM  mean_edge_length
        # The other eight fields (t_*_s, peak_rss_bytes, iters,
        # converged, solver_err) match the DDA names exactly so
        # end-to-end cost comparisons reduce to a direct HDF5 lookup.
        cost = create_group(grp, "cost")
        attrs(cost)["description"] = "per-slot cost and solver diagnostics"
        attrs(cost)["units"]       =
            "t_*:[s], peak_rss_bytes:[B], mean_edge_length:[um], solver_err:[1]"
        ds(cost, "t_build_s",        shape_cond, Float64,
           "wall time: tet mesh + SWG basis build [s]")
        ds(cost, "t_setup_s",        shape_cond, Float64,
           "wall time: AIM grid + projection + mass-matrix assembly [s]")
        ds(cost, "t_solve_s",        shape_cond, Float64,
           "wall time: block-Krylov iterative solve (no mesh/setup) [s]")
        ds(cost, "t_total_s",        shape_cond, Float64,
           "end-to-end wall time (build + setup + solve + observables) [s]")
        ds(cost, "peak_rss_bytes",   shape_cond, Int64,
           "peak resident set size observed during this slot [bytes]")
        ds(cost, "n_tet",            shape_cond, Int64,
           "number of tetrahedra in the VIEM mesh " *
           "(corresponds to block-DDA_Py's n_cuboid)")
        ds(cost, "n_dof",            shape_cond, Int64,
           "number of SWG basis functions (unknowns) " *
           "(corresponds to block-DDA_Py's n_occ)")
        ds(cost, "mean_edge_length", shape_cond, Float64,
           "tet-mesh mean edge length h̄ [um] " *
           "(corresponds to block-DDA_Py's lattice_lf)")
        ds(cost, "iters",            shape_cond, Int64,
           "block-Krylov outer iteration count")
        ds(cost, "converged",        shape_cond, Int8,
           "1 if block-Krylov reached tol, else 0")
        ds(cost, "solver_err",       shape_cond, Float64,
           "final relative residual ‖B − A·X‖_F / ‖B‖_F")

        # Per-iteration residual history (v0.7.5, symmetric with DDA).
        # Shape: shape_cond × MAXITER_HISTORY.  NaN-padded beyond iter_fin.
        # residual_history[..., k] = relative residual after iteration k.
        shape_hist = (shape_cond..., MAXITER_HISTORY)
        d_hist = create_dataset(cost, "residual_history", Float64, shape_hist)
        attrs(d_hist)["definition"] =
            "per-iteration relative residual ‖B − A·X‖_F / ‖B‖_F, " *
            "NaN-padded to length $MAXITER_HISTORY.  entry [..., k] is " *
            "the residual after iteration k (1-indexed); NaN if k > iter_fin.  " *
            "Enables convergence-profile figures symmetric with block-DDA_Py."
        # Initialise with NaN so unfilled slots are distinguishable from 0.
        write(d_hist, fill(NaN, shape_hist))
    end

    println("Created $filename")
    println("  shape_kind: $shape_kind  " *
            "(bc=$bc_ratio  ab=$ab_ratio  β_gre=$gre_beta, spheroid_mode=$spheroid_mode)")
    println("  m_p  : $m_p_vec")
    println("  a_eq : $r_v_base_list μm")
    println("  N_α=$N_alpha  N_β=$N_beta  N_γ=$N_gamma  " *
            "→ $num_orientations nominal orientations")
    println("  block-Krylov L (per shape slot) = " *
            (spheroid_mode ? "$N_beta (spheroid mode)" : "$num_orientations (GRE mode)"))
    return filename
end
