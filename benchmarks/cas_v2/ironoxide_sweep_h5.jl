# Iron-oxide spheroid sweep for PCAS_Bayes_for_liquid_2wls (hematite, goethite).
#
# Differs from spheroid_sweep_h5.jl in four ways, each forced by the physics:
#
#   1. AIM (`:aim_gmres`). These are high-contrast particles (hematite Re(m) ~ 3.0
#      against a water host of 1.33) and the meshes run to ~10^5 DOF at the top of
#      the size range, where a dense operator needs O(N^2) memory. Measured
#      2026-09-01: 88k tetrahedra at D_ve = 0.70 um costs 18.6 GB with AIM.
#
#   2. Mesh rule `lc = min(lambda_0/(|m_p| N_pw), LC_GEOM * min(b,c))`. The
#      wavelength criterion alone leaves a 0.05 um particle with a few dozen
#      tetrahedra -- the shape is not represented and |A| disagreed by a factor of
#      two between two mesh factors. The geometric term is what the older sweep
#      used; neither alone is sufficient across this size range.
#
#   3. A COMPLEX m_p whose imaginary part is fixed per wavelength. The consumer's
#      grid has a real-index axis only, so Im(m) rides along as the per-wavelength
#      `m_imag` attribute. Hematite's Im(m) is 0.0813 at 637 nm and 0.0318 at
#      773 nm -- a factor of 2.6 -- so a single root value would be wrong for one
#      of the two bands.
#
#   4. Both oblate AND prolate: the log-AR grid straddles zero. Flow alignment
#      differs between the two and the hand comparisons could not settle which
#      applies, so both stay candidates.
#
# The refractive indices are PROVISIONAL literature constants
# (references/IronOxide/provisional_indices.json in the PCAS tree). The real-index
# axis exists because the consumer's spline needs >= 3 points per axis, and it is
# the minimum that also leaves the future index-validation study possible without
# recomputing the table.
#
# Run with:
#     julia --project=. benchmarks/cas_v2/ironoxide_sweep_h5.jl --species hematite

using Printf, Dates
using Serialization: serialize, deserialize
using BlockVIEM
using BlockVIEM: build_swg_basis, n_basis, n_tets, read_msh, mean_edge_length,
                 solve_cas_v2_orientations, aim_grid, build_aim_projection,
                 assemble_mass_matrix, duffy_reference_rule,
                 expand_alpha_from_alpha0,
                 SpheroidSweepGrids, SpheroidSweepData, write_spheroid_sweep_h5
import Gmsh: gmsh

# ── CLI ──────────────────────────────────────────────────────────────────────
function _arg(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i + 1]
end
const SPECIES = lowercase(_arg("--species", "hematite"))
const N_DVE   = parse(Int, _arg("--n-dve", "8"))
const N_AR    = parse(Int, _arg("--n-ar",  "7"))
const N_RI    = parse(Int, _arg("--n-ri",  "3"))
const DRY_RUN = "--dry-run" in ARGS
# Size range: overridable so the checkpoint/resume path can be exercised cheaply
# on small particles instead of a run that reaches the expensive top of the grid.
const DVE_MIN = parse(Float64, _arg("--dve-min", "0.05"))
const DVE_MAX = parse(Float64, _arg("--dve-max", "0.70"))
# Shape-axis half-width in log10(AR), and the mesh coarseness multiplier. Both were
# measured on 2026-09-03 rather than chosen: the largest amplitude stratum needs
# |log10 AR| ~ 0.98 (extrapolating the table's own slope of 1.01), and coarsening lc
# by 1.6 costs 4.5x less while |B/A| moves only inside the convergence wobble --
# 2.02% at 1.0 against 2.89% at 1.6, with 2.5 breaking down at -14%.
const LOG_AR_MAX = parse(Float64, _arg("--log-ar-max", "0.6"))
const LC_FACTOR  = parse(Float64, _arg("--lc-factor", "1.0"))
# Solve ONE index and replicate it across a degenerate real-index axis.
#
# The consumer's schema puts an axis on Re(m_p) and bakes Im(m_p) in as an
# attribute, and its spline needs three points on every axis -- so a fixed-index
# species pays 3x for an axis it never queries off-centre. The user's call
# (2026-09-03): if a sensitivity study is worth running it should be on the
# IMAGINARY part, which carries the larger uncertainty -- goethite's k is digitised
# off a figure with a spread near 30% and is carried to one significant figure,
# while the real parts are read cleanly. That study cannot be an axis here, so it
# becomes a SECOND TABLE at a different Im, and the real axis stops being paid for.
#
# The file says so in its own attributes, and the consumer refuses to be queried
# off the degenerate value -- otherwise a table that LOOKS like it has an index axis
# would return the same amplitudes for every index, silently.
const DEGENERATE_RI = "--degenerate-ri" in ARGS
# FORM BIREFRINGENCE (PCAS handoff 2026-09-08, user's choice): the particle is a porous
# aggregate of needles aligned with the spheroid's symmetry axis z, modelled as a
# uniaxial Maxwell-Garnett medium of crystal cylinders in the host. The index axis then
# carries the FILL FRACTION f (three points), and each point maps to a per-band tensor
#   eps_par  = f eps_c + (1-f) eps_m,   eps_perp = eps_m [(1+f)eps_c + (1-f)eps_m] / [(1-f)eps_c + (1+f)eps_m]
# with m_p = [n_perp, n_perp, n_par] locked to the mesh frame. Axial symmetry is kept,
# so the analytic azimuth expansion and the consumer's table format are unchanged. The
# axis VALUE written to the file is Re of the eps-average index at the first band, so the
# consumer's "RI" coordinate keeps its meaning of an effective real index; the fills and
# per-band tensors are recorded as attributes. Measured (form_birefringence_probe.jl):
# a prolate-3 aggregate at fill 0.5 depolarizes 0.25 at beta = 90, twice the isotropic
# control, which is what the small goethite particles show and the isotropic tables cannot.
const FORM_BIREF = "--form-biref" in ARGS
const FILLS = FORM_BIREF ? parse.(Float64, split(_arg("--fills", "0.4,0.5,0.6"), ",")) : Float64[]

# ── species constants (provisional; see the JSON in the PCAS tree) ───────────
# (wl_um, m_m, Re(m_p) at that wl, Im(m_p) at that wl)
const SPECIES_TABLE = Dict(
    "hematite" => [(0.637, 1.3315, 3.0060, 0.0813),
                   (0.773, 1.3300, 2.7895, 0.0318)],
    # Goethite is biaxial and its principal indices are unpublished, so this is an
    # effective (sphere-assumed) index and birefringence cannot be represented.
    # The 773 nm value is SUBSTITUTED from ~0.70 um, the end of the published range.
    "goethite" => [(0.637, 1.3315, 2.2200, 0.0800),
                   (0.773, 1.3300, 2.1800, 0.1100)],
    # "Soft" goethite: the EFFECTIVE index the Koju standard's mean-amplitude cloud asks
    # for (PCAS handoff 2026-09-05/07). At 2.22+0.08i the model's arg A runs ahead of the
    # data by up to 22 degrees and no spheroid shape brings it back; a Mie sphere scan
    # reproduces the phase-versus-|A| trend at 1.60+0.04i (637 nm) and 1.70+0.02i
    # (773 nm) to about a degree, and the silicate table (k = 0) pinned at its 1.69
    # ceiling and still fell short -- the cloud wants BOTH the low real part and the
    # small absorption. Far below the crystal's principal indices (2.26-2.52): porous
    # needle aggregates, presumably. Real axis kept non-degenerate so Phase 1 can
    # estimate Re(m) on it; Im(m) fixed per band from the sphere scan.
    "goethite_soft" => [(0.637, 1.3315, 1.6000, 0.0400),
                        (0.773, 1.3300, 1.7000, 0.0200)],
    # Same finding for hematite (PCAS handoff 2026-09-08): the Koju standard's phase-
    # versus-|A| trend wants n ~ 2.2 with k 0.05-0.08 (rms 3 / 1 degree) against 20 / 14
    # for the Querry epsilon-average 3.006+0.08i. The observed cloud is also Mie-side cut,
    # so the sphere scan is a guide and the posterior-predictive check decides.
    "hematite_soft" => [(0.637, 1.3315, 2.2000, 0.0800),
                        (0.773, 1.3300, 2.2000, 0.0500)],
    # --form-biref only: (wl, m_m, Re m_c, Im m_c) of the CRYSTAL needles. The real part is a
    # representative goethite value at red wavelengths (principal indices 2.26-2.52 at Na D,
    # strong dispersion); the imaginary part follows the sphere scan's k ratio between bands.
    "goethite_needle" => [(0.637, 1.3315, 2.3000, 0.1000),
                          (0.773, 1.3300, 2.3000, 0.0600)],
)
haskey(SPECIES_TABLE, SPECIES) ||
    error("unknown species $(SPECIES) (have: $(join(sort(collect(keys(SPECIES_TABLE))), ", ")))")
const COND = SPECIES_TABLE[SPECIES]
const OUT_FILE = _arg("--output",
    joinpath(@__DIR__, "spheroid_sweep_viem_$(SPECIES)_liquid.h5"))

# ── grids ────────────────────────────────────────────────────────────────────
# Log-spaced sizes: the consumer requires the LOG axis to be equidistant.
const D_VE_GRID = 10 .^ collect(range(log10(DVE_MIN), log10(DVE_MAX), length = N_DVE))
const LOG_AR_GRID = collect(range(-LOG_AR_MAX, LOG_AR_MAX, length = N_AR))
const COS_THETA_O_HALF = collect(range(0.0, 1.0, length = 13))
const PHI_O_GRID = collect(range(0.0, pi, length = 21))         # analytic: free

function mg_uniaxial(m_c, m_m, f)
    ec, em = m_c^2, m_m^2
    e_par  = f * ec + (1 - f) * em
    e_perp = em * ((1 + f) * ec + (1 - f) * em) / ((1 - f) * ec + (1 + f) * em)
    return sqrt(e_par), sqrt(e_perp), sqrt((e_par + 2e_perp) / 3)
end
# Real-index axis: 3 points spanning BOTH wavelengths' values with margin. The
# axis is shared by the wavelength groups, so it has to bracket both.
_re = [c[3] for c in COND]
# Overridable: a narrower axis interpolates better when Re(m) will be ESTIMATED on it
# (three cubic-spline points over 0.3 rather than 0.5 of index).
const RI_HALFSPAN = parse(Float64, _arg("--ri-halfspan", "0.20"))
# N_RI = 1 is for cost probes only -- the consumer's spline needs >= 3 per axis and
# refuses such a table. range() cannot take differing endpoints with length 1.
const RI_REAL_GRID = FORM_BIREF ?
    [real(mg_uniaxial(complex(COND[1][3], COND[1][4]), COND[1][2], f)[3]) for f in FILLS] :
    N_RI == 1 ?
    [(minimum(_re) + maximum(_re)) / 2] :
    collect(range(minimum(_re) - RI_HALFSPAN, maximum(_re) + RI_HALFSPAN, length = N_RI))
FORM_BIREF && length(FILLS) != N_RI && error("--fills has $(length(FILLS)) values but --n-ri is $N_RI")
FORM_BIREF && DEGENERATE_RI && error("--form-biref and --degenerate-ri are alternatives")

# ── solver settings (measured 2026-09-01, see docs/handoff) ──────────────────
const N_PW        = 10       # mesh cells per wavelength inside the particle
const LC_GEOM     = 0.30     # ... but never coarser than this x the smallest semi-axis
const PITCH_RATIO = 0.5
const PADDING     = 4
const TOL         = 1e-5
const MAXITER     = 600
const DUFFY_ORDER = 5

function spheroid_mesh(b, c, lc)
    path = joinpath(tempdir(), "viem_fe_$(round(b,digits=6))_$(round(c,digits=6))_$(round(lc,digits=6)).msh")
    isfile(path) && return path
    # Write to a scratch name and rename: a run killed mid-write would otherwise
    # leave a truncated .msh that the next run happily reuses. The scratch name
    # must KEEP the .msh extension -- gmsh picks its output format from it, and a
    # ".msh.part" suffix fails with "Unknown output file format".
    tmp = joinpath(dirname(path), "part$(getpid())_" * basename(path))
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sph")
        s = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, s)], 0.0, 0.0, 0.0, b, b, c)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [s], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)
        gmsh.write(tmp)
    finally
        gmsh.finalize()
    end
    mv(tmp, path; force = true)
    path
end

# ── checkpointing ────────────────────────────────────────────────────────────
# This sweep runs for the better part of a day. Without a checkpoint a failure in
# the last hour throws away every hour before it -- which is exactly how 13.5 h
# went missing on 2026-08-13. State is saved after each geometry row and reloaded
# on restart, so a rerun with the same settings resumes instead of starting over.
const CKPT_FILE = _arg("--checkpoint", OUT_FILE * ".ckpt.jls")
const NO_RESUME = "--no-resume" in ARGS

# Settings that invalidate a checkpoint if they change. Grids are compared by
# value, not by count: a rerun with the same N but a different range must not
# silently inherit amplitudes computed on the old grid.
_ckpt_key() = (SPECIES, DEGENERATE_RI, D_VE_GRID, RI_REAL_GRID, LOG_AR_GRID, COS_THETA_O_HALF,
               PHI_O_GRID, N_PW, LC_GEOM, LC_FACTOR, TOL, DUFFY_ORDER, COND)

function _load_ckpt()
    (NO_RESUME || !isfile(CKPT_FILE)) && return nothing
    try
        st = open(deserialize, CKPT_FILE)
        if st.key != _ckpt_key()
            @warn "checkpoint settings differ from this run; ignoring it" file=CKPT_FILE
            return nothing
        end
        @printf("resuming from %s (%d of %d geometry rows already done)\n",
                CKPT_FILE, count(st.done), length(st.done))
        return st
    catch err
        @warn "could not read checkpoint; starting fresh" file=CKPT_FILE err
        return nothing
    end
end

function _save_ckpt(st)
    tmp = CKPT_FILE * ".part"
    open(f -> serialize(f, st), tmp, "w")
    mv(tmp, CKPT_FILE; force = true)   # atomic: never leave a half-written checkpoint
end

mutable struct SweepState
    key::Any
    S_theta::Vector{Array{ComplexF64,5}}   # one per wavelength
    S_phi::Vector{Array{ComplexF64,5}}
    converged::Vector{Array{Bool,3}}
    done::Array{Bool,3}                    # (wavelength, D_ve, AR) geometry rows
end

function _fresh_state()
    sz = (N_DVE, N_RI, N_AR, length(COS_THETA_O_HALF), length(PHI_O_GRID))
    nw = length(COND)
    SweepState(_ckpt_key(),
               [fill(NaN + NaN*im, sz) for _ in 1:nw],
               [fill(NaN + NaN*im, sz) for _ in 1:nw],
               [trues(N_DVE, N_RI, N_AR) for _ in 1:nw],
               falses(nw, N_DVE, N_AR))
end

function sweep_one_wavelength!(st::SweepState, w::Int, wl_0, m_m, re_axis, im_fixed,
                               re_fixed)
    # With a degenerate axis one index is solved and copied across. It must be the
    # species' own value FOR THIS BAND, not the axis midpoint: the axis is shared by
    # the two wavelengths and their real parts differ (goethite 2.22 against 2.18,
    # hematite 3.006 against 2.7895), so the midpoint matches neither.
    solve_idx = DEGENERATE_RI ? [(length(re_axis) + 1) ÷ 2] : collect(1:N_RI)
    re_used = DEGENERATE_RI ? fill(re_fixed, length(re_axis)) : re_axis
    # The mesh is shared across the real-index axis: lc uses the LARGEST |m_p| on
    # the axis, which is the finest requirement, so the shared mesh is at least as
    # fine as any single index needs. That amortises meshing, the AIM grid, the
    # projection and the mass matrix over N_RI solves -- none of them depend on m_p.
    m_worst = DEGENERATE_RI ? abs(complex(re_fixed, im_fixed)) :
              FORM_BIREF ? maximum(abs(mg_uniaxial(complex(re_fixed, im_fixed), m_m, f)[1]) for f in FILLS) :
              maximum(abs(complex(re, im_fixed)) for re in re_axis)
    # per-index particle tensor [x, y, z] in the mesh frame (z = symmetry axis)
    tensor_of(j) = FORM_BIREF ?
        (let (np_, nq_, _) = mg_uniaxial(complex(re_fixed, im_fixed), m_m, FILLS[j]); [nq_, nq_, np_] end) :
        (let mp = complex(re_used[j], im_fixed); [mp, mp, mp] end)

    for i in 1:N_DVE, k in 1:N_AR
        D_ve = D_VE_GRID[i]; AR = 10.0 ^ LOG_AR_GRID[k]
        r = D_ve / 2
        b = r * AR ^ ( 1/3)          # AR > 1 oblate (b > c), AR < 1 prolate
        c = r * AR ^ (-2/3)
        lc = min(LC_FACTOR * wl_0 / (m_worst * N_PW), LC_GEOM * min(b, c))

        if st.done[w, i, k]
            @printf("  [D=%.4f AR=%.3f] (checkpoint)\n", D_ve, AR); continue
        end
        @printf("  [D=%.4f AR=%.3f] lc=%.5f ", D_ve, AR, lc); flush(stdout)
        if DRY_RUN
            println("(dry run)"); continue
        end
        try
            mesh = read_msh(spheroid_mesh(b, c, lc))
            basis = build_swg_basis(mesh; include_boundary_faces = true)
            pitch = PITCH_RATIO * mean_edge_length(mesh)
            grid  = aim_grid(basis.mesh; pitch = pitch, padding = PADDING)
            proj  = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
            mass  = assemble_mass_matrix(basis)
            @printf("N_tet=%d N=%d ", n_tets(mesh), n_basis(basis)); flush(stdout)

            # One block per index: the polar grid is the multi-RHS dimension
            # (block-Krylov halves the iteration count against single-RHS solves),
            # and the azimuth is expanded analytically from alpha = 0.
            eul = [(0.0, acos(u), 0.0) for u in COS_THETA_O_HALF]
            for j in solve_idx
                m_vec = tensor_of(j)
                t = @elapsed res, _, info = solve_cas_v2_orientations(
                    basis, eul;
                    wl_0 = wl_0, m_m = m_m, m_p = m_vec,
                    method = :aim_gmres, tol = TOL, maxiter = MAXITER,
                    return_D = true, return_solve_info = true,
                    duffy_rule = duffy_reference_rule(DUFFY_ORDER), symmetrize = true,
                    pitch = pitch, padding = PADDING, projection = proj, mass = mass)
                st.converged[w][i, j, k] = info.converged
                for m in 1:length(COS_THETA_O_HALF)
                    Sth, Sph = expand_alpha_from_alpha0(
                        res[m].S_fw_theta, res[m].S_fw_phi, PHI_O_GRID)
                    st.S_theta[w][i, j, k, m, :] .= Sth
                    st.S_phi[w][i, j, k, m, :]   .= Sph
                end
                @printf("| %s %.0fs/%dit%s ",
                        FORM_BIREF ? @sprintf("f=%.2f n⊥=%.3f n∥=%.3f", FILLS[j], real(m_vec[1]), real(m_vec[3])) :
                                     @sprintf("n=%.3f", re_used[j]),
                        t, info.iterations, info.converged ? "" : " NOTCONV")
                flush(stdout)
            end
            if DEGENERATE_RI
                j0 = solve_idx[1]
                for j in 1:N_RI
                    j == j0 && continue
                    st.S_theta[w][i, j, k, :, :] .= st.S_theta[w][i, j0, k, :, :]
                    st.S_phi[w][i, j, k, :, :]   .= st.S_phi[w][i, j0, k, :, :]
                    st.converged[w][i, j, k] = st.converged[w][i, j0, k]
                end
                @printf("| (%d indices replicated) ", N_RI - 1)
            end
            println()
        catch err
            println("FAILED: ", sprint(showerror, err)[1:min(90, end)])
            st.converged[w][i, :, k] .= false
            st.S_theta[w][i, :, k, :, :] .= NaN + NaN*im
            st.S_phi[w][i, :, k, :, :]   .= NaN + NaN*im
        end
        # Mark done even on failure: the cell is recorded as non-converged and a
        # rerun must not silently retry it into a different answer. Use
        # --no-resume to recompute everything.
        st.done[w, i, k] = true
        _save_ckpt(st)
    end
    SpheroidSweepData(wl_0, m_m, im_fixed, st.S_theta[w], st.S_phi[w], st.converged[w])
end

@printf("iron-oxide sweep  species=%s  %s\n", SPECIES, Dates.format(now(), "yyyy-mm-dd HH:MM"))
@printf("  D_ve  %d points  %.3f .. %.3f um (log-spaced)\n", N_DVE, first(D_VE_GRID), last(D_VE_GRID))
@printf("  AR    %d points  %.3f .. %.3f (prolate through oblate)\n",
        N_AR, 10.0^first(LOG_AR_GRID), 10.0^last(LOG_AR_GRID))
@printf("  mesh  lc factor %.2f  (1.0 = ten cells per wavelength inside the particle)\n", LC_FACTOR)
@printf("  Re(m) %d points  %.3f .. %.3f%s\n", N_RI, first(RI_REAL_GRID), last(RI_REAL_GRID),
        DEGENERATE_RI ? "  DEGENERATE: solved per band at " *
                        join([string(c[3]) for c in COND], "/") * ", copied across" :
        FORM_BIREF ? "  FORM BIREFRINGENCE: axis = eps-average Re(n) at band 1 for fills " *
                     join(string.(FILLS), "/") * " (uniaxial MG, needles along z)" : "")
if FORM_BIREF
    for (wl, m_m, re_c, im_c) in COND, f in FILLS
        np_, nq_, ni_ = mg_uniaxial(complex(re_c, im_c), m_m, f)
        @printf("    wl %.3f fill %.2f: n∥ %.3f%+.3fi  n⊥ %.3f%+.3fi  eps-avg %.3f%+.3fi\n", wl, f,
                real(np_), imag(np_), real(nq_), imag(nq_), real(ni_), imag(ni_))
    end
end
@printf("  theta %d (one block)   phi %d (analytic)\n",
        length(COS_THETA_O_HALF), length(PHI_O_GRID))
@printf("  cells: %d geometries x %d indices x %d wavelengths\n",
        N_DVE * N_AR, N_RI, length(COND))
@printf("  checkpoint: %s\n", DRY_RUN ? "(dry run)" : CKPT_FILE)
@printf("  output: %s\n\n", OUT_FILE)

state = something(_load_ckpt(), _fresh_state())
data = SpheroidSweepData[]
for (w, (wl, m_m, re_c, im_c)) in enumerate(COND)
    @printf("--- lambda = %.3f um   m_m = %.4f   Im(m_p) = %.4f (fixed) ---\n", wl, m_m, im_c)
    push!(data, sweep_one_wavelength!(state, w, wl, m_m, RI_REAL_GRID, im_c, re_c))
end

if !DRY_RUN
    grids = SpheroidSweepGrids(D_VE_GRID, RI_REAL_GRID, LOG_AR_GRID,
                               COS_THETA_O_HALF, PHI_O_GRID)
    write_spheroid_sweep_h5(OUT_FILE, grids, data;
        block_viem_version = "0.1.1", solver_tol = TOL,
        extra_root_attrs = merge(
            Dict("producer" => "block-VIEM.jl",
                 "species" => SPECIES,
                 "index_status" => FORM_BIREF ? "uniaxial Maxwell-Garnett effective medium of aligned needles; RI axis = eps-average Re(n) at band 1" :
                                                "provisional literature constants"),
            FORM_BIREF ? Dict(
                "form_birefringence" => 1,
                "fill_fractions" => join(string.(FILLS), ","),
                "crystal_m_per_wl" => join([@sprintf("%.3f:%.4f+%.4fi", c[1], c[3], c[4]) for c in COND], ";"),
                "n_par_n_perp_per_wl_per_fill" => join([@sprintf("%.3f:f%.2f:%.4f+%.4fi/%.4f+%.4fi", c[1], f,
                    real(mg_uniaxial(complex(c[3], c[4]), c[2], f)[1]), imag(mg_uniaxial(complex(c[3], c[4]), c[2], f)[1]),
                    real(mg_uniaxial(complex(c[3], c[4]), c[2], f)[2]), imag(mg_uniaxial(complex(c[3], c[4]), c[2], f)[2]))
                    for c in COND for f in FILLS], ";")) : Dict(),
            DEGENERATE_RI ?
                Dict("ri_axis_degenerate" => 1,
                     "ri_axis_value_per_wl" =>
                        join(["$(c[1])=$(c[3])" for c in COND], ","),
                     "ri_axis_note" =>
                        "Re(m_p) was solved at ri_axis_value_per_wl (one value PER BAND, " *
                        "since the two bands' real parts differ) and copied across the " *
                        "axis. Querying any other index returns the SAME amplitudes. " *
                        "A refractive-index sensitivity study belongs on Im(m_p) and is " *
                        "a separate table, not a point on this axis.") :
                Dict{String,Any}()))
    n_bad = sum(!d.converged[i] for d in data for i in eachindex(d.converged))
    @printf("\nWrote %s   non-converged cells: %d\n", OUT_FILE, n_bad)
    n_bad == 0 || @warn "non-converged cells present; the consumer's surrogate REFUSES a LUT with NaNs"
end
