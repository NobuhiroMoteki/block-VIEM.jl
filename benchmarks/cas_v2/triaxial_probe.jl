# Does a TRIAXIAL ellipsoid thin the low-depolarization tail that an axisymmetric spheroid
# cannot avoid? (PCAS handoff 2026-09-14, user's choice (a) over a rotated crystal axis.)
#
# An axisymmetric particle has B = 0 exactly when its axis is along the beam, whatever the
# aspect ratio, and near that axis |B| grows as sin^2(theta): the fraction of isotropic
# orientations with |B/A| < eps is LINEAR in eps. The hematite standard shows a floor
# (p10 0.04-0.05 in the bright strata) the spheroid population cannot reproduce (p10 0.003-0.006).
# A triaxial ellipsoid still has isolated zeros of B (a 2-D map to the complex plane), but
# near an isolated zero |B| grows LINEARLY with the angle, so the fraction below eps is
# QUADRATIC in eps -- a much lighter tail. This probe measures it directly: isotropic random
# orientations (Euler alpha, cos beta, gamma uniform) at fixed size and index, for a
# spheroid (b/a = 1) and triaxial ellipsoids (b/a = 1.2, 1.5), prolate and oblate, and reports
# percentiles of |B/A|. Semi-axes: c/a = AR (the spheroid's aspect ratio), b/a = R2, volume fixed
# by D_ve.
#
#   julia --project=. -t 8 benchmarks/cas_v2/triaxial_probe.jl
using Printf, Random, Statistics
using BlockVIEM
using BlockVIEM: build_swg_basis, n_tets, read_msh, mean_edge_length,
                 solve_cas_v2_orientations, aim_grid, build_aim_projection,
                 assemble_mass_matrix, duffy_reference_rule
import Gmsh: gmsh

const WL  = 0.637
const M_M = 1.3315
const M_P = parse(ComplexF64, get(ENV, "M_P", "2.52+0.08im"))      # hematite effective index (fixed point W 0.2)
const D_VE_LIST = [parse(Float64, s) for s in split(get(ENV, "D_VE", "0.20,0.35"), ",")]
const SHAPES = [("prolate_2", 2.0), ("oblate_2", 0.5)]              # AR = c/a
const R2S    = [1.0, 1.2, 1.5]                                       # b/a
const N_OR   = parse(Int, get(ENV, "N_OR", "120"))
# Orientations are solved in CHUNKS. The first launch (2026-09-14 00:40) passed all 150 as one
# block right-hand side: the block-Krylov workspace scales with N_basis x n_rhs, the process grew
# to 142 GB, and the kernel's OOM killer took it -- and with it the editor's whole process scope
# (queue loop, 16 MSTM sweeps, a GPU fit). 10 per solve keeps the workspace at the size the
# production sweeps use.
const CHUNK  = parse(Int, get(ENV, "CHUNK", "10"))
const N_PW = 10; const LC_GEOM = 0.30; const LC_FACTOR = 1.6
const PITCH_RATIO = 0.5; const PADDING = 4; const TOL = 1e-5; const MAXITER = 600

function ellipsoid_mesh(a, b, c, lc)
    path = joinpath(tempdir(), "viem_tri_$(round(a,digits=6))_$(round(b,digits=6))_$(round(c,digits=6))_$(round(lc,digits=6)).msh")
    isfile(path) && return path
    tmp = joinpath(dirname(path), "part$(getpid())_" * basename(path))
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ell")
        s = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, s)], 0.0, 0.0, 0.0, a, b, c)
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

rng = MersenneTwister(7)
const EUL = [(2π * rand(rng), acos(2 * rand(rng) - 1), 2π * rand(rng)) for _ in 1:N_OR]

@printf("triaxial probe  wl %.3f um  m_m %.4f  m_p %.2f%+.2fi  %d isotropic orientations\n", WL, M_M, real(M_P), imag(M_P), N_OR)
@printf("%-10s %-5s %-4s  %s   min   [N_tet]\n", "shape", "D_ve", "b/a", join([@sprintf("%6s", "p" * string(q)) for q in (2, 5, 10, 25, 50, 90)], " "))
m_worst = abs(M_P)
for (name, AR) in SHAPES, D in D_VE_LIST, r2 in R2S
    a = (D / 2) / cbrt(AR * r2); b = r2 * a; c = AR * a
    lc = min(LC_FACTOR * WL / (m_worst * N_PW), LC_GEOM * min(a, b, c))
    mesh  = read_msh(ellipsoid_mesh(a, b, c, lc))
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    pitch = PITCH_RATIO * mean_edge_length(mesh)
    grid  = aim_grid(basis.mesh; pitch = pitch, padding = PADDING)
    proj  = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    mass  = assemble_mass_matrix(basis)
    ratios = Float64[]
    for i0 in 1:CHUNK:length(EUL)
        res = solve_cas_v2_orientations(basis, EUL[i0:min(i0 + CHUNK - 1, end)]; wl_0 = WL, m_m = M_M, m_p = M_P,
                  method = :aim_gmres, tol = TOL, maxiter = MAXITER,
                  duffy_rule = duffy_reference_rule(5), symmetrize = true,
                  pitch = pitch, padding = PADDING, projection = proj, mass = mass)
        append!(ratios, [abs((rr.S_fw_theta - rr.S_fw_phi) / 2) / abs((rr.S_fw_theta + rr.S_fw_phi) / 2) for rr in res])
        GC.gc()
    end
    qs = [quantile(ratios, q / 100) for q in (2, 5, 10, 25, 50, 90)]
    @printf("%-10s %-5.2f %-4.1f  %s   %.4f  [%d]\n", name, D, r2, join([@sprintf("%.4f", x) for x in qs], " "), minimum(ratios), n_tets(mesh))
    flush(stdout)
end
