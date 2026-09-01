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
)
haskey(SPECIES_TABLE, SPECIES) ||
    error("unknown species $(SPECIES) (have: $(join(sort(collect(keys(SPECIES_TABLE))), ", ")))")
const COND = SPECIES_TABLE[SPECIES]
const OUT_FILE = _arg("--output",
    joinpath(@__DIR__, "spheroid_sweep_viem_$(SPECIES)_liquid.h5"))

# ── grids ────────────────────────────────────────────────────────────────────
# Log-spaced sizes: the consumer requires the LOG axis to be equidistant.
const D_VE_GRID = 10 .^ collect(range(log10(DVE_MIN), log10(DVE_MAX), length = N_DVE))
const LOG_AR_GRID = collect(range(-0.6, 0.6, length = N_AR))   # AR 0.25 .. 4.0
const COS_THETA_O_HALF = collect(range(0.0, 1.0, length = 13))
const PHI_O_GRID = collect(range(0.0, pi, length = 21))         # analytic: free

# Real-index axis: 3 points spanning BOTH wavelengths' values with margin. The
# axis is shared by the wavelength groups, so it has to bracket both.
_re = [c[3] for c in COND]
const RI_HALFSPAN = 0.20
# N_RI = 1 is for cost probes only -- the consumer's spline needs >= 3 per axis and
# refuses such a table. range() cannot take differing endpoints with length 1.
const RI_REAL_GRID = N_RI == 1 ?
    [(minimum(_re) + maximum(_re)) / 2] :
    collect(range(minimum(_re) - RI_HALFSPAN, maximum(_re) + RI_HALFSPAN, length = N_RI))

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
_ckpt_key() = (SPECIES, D_VE_GRID, RI_REAL_GRID, LOG_AR_GRID, COS_THETA_O_HALF,
               PHI_O_GRID, N_PW, LC_GEOM, TOL, DUFFY_ORDER, COND)

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

function sweep_one_wavelength!(st::SweepState, w::Int, wl_0, m_m, re_axis, im_fixed)
    # The mesh is shared across the real-index axis: lc uses the LARGEST |m_p| on
    # the axis, which is the finest requirement, so the shared mesh is at least as
    # fine as any single index needs. That amortises meshing, the AIM grid, the
    # projection and the mass matrix over N_RI solves -- none of them depend on m_p.
    m_worst = maximum(abs(complex(re, im_fixed)) for re in re_axis)

    for i in 1:N_DVE, k in 1:N_AR
        D_ve = D_VE_GRID[i]; AR = 10.0 ^ LOG_AR_GRID[k]
        r = D_ve / 2
        b = r * AR ^ ( 1/3)          # AR > 1 oblate (b > c), AR < 1 prolate
        c = r * AR ^ (-2/3)
        lc = min(wl_0 / (m_worst * N_PW), LC_GEOM * min(b, c))

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
            for j in 1:N_RI
                m_p = complex(re_axis[j], im_fixed)
                t = @elapsed res, _, info = solve_cas_v2_orientations(
                    basis, eul;
                    wl_0 = wl_0, m_m = m_m, m_p = [m_p, m_p, m_p],
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
                @printf("| n=%.3f %.0fs/%dit%s ", re_axis[j], t, info.iterations,
                        info.converged ? "" : " NOTCONV")
                flush(stdout)
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
@printf("  Re(m) %d points  %.3f .. %.3f\n", N_RI, first(RI_REAL_GRID), last(RI_REAL_GRID))
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
    push!(data, sweep_one_wavelength!(state, w, wl, m_m, RI_REAL_GRID, im_c))
end

if !DRY_RUN
    grids = SpheroidSweepGrids(D_VE_GRID, RI_REAL_GRID, LOG_AR_GRID,
                               COS_THETA_O_HALF, PHI_O_GRID)
    write_spheroid_sweep_h5(OUT_FILE, grids, data;
        block_viem_version = "0.1.1", solver_tol = TOL,
        extra_root_attrs = Dict("producer" => "block-VIEM.jl",
                                "species" => SPECIES,
                                "index_status" => "provisional literature constants"))
    n_bad = sum(!d.converged[i] for d in data for i in eachindex(d.converged))
    @printf("\nWrote %s   non-converged cells: %d\n", OUT_FILE, n_bad)
    n_bad == 0 || @warn "non-converged cells present; the consumer's surrogate REFUSES a LUT with NaNs"
end
