# Feasibility probe (PCAS handoff 2026-09-08, user's choice (1)): can FORM birefringence
# of aligned goethite needles inside a porous aggregate produce the depolarization the
# Koju standard shows, where an isotropic spheroid at the effective index cannot?
#
# The particle is a spheroid whose permittivity is a uniaxial EFFECTIVE MEDIUM locked to
# the mesh frame: needles aligned with the symmetry axis z. Maxwell-Garnett for aligned
# cylinders in water with crystal m_c = 2.30+0.10i and fill fraction f:
#   eps_par  = f eps_c + (1-f) eps_m                       (along the needles, z)
#   eps_perp = eps_m [(1+f) eps_c + (1-f) eps_m] / [(1-f) eps_c + (1+f) eps_m]
# f = 0.5 gives n_par 1.88+0.06i, n_perp 1.72+0.03i, eps-average 1.77+0.04i -- the
# mean matching the 1.79 the isotropic fit chose. The isotropic control uses the
# eps-average on the same mesh, so the difference is the anisotropy alone.
#
# Orientation: the incident wave is rotated (particle fixed), beta = angle between the
# beam and the symmetry/optic axis: beta = 0 is axis along the beam, where an isotropic
# spheroid depolarizes nothing. The number that matters is |B/A| there.
#
#   julia --project=. -t 4 benchmarks/cas_v2/form_birefringence_probe.jl

using Printf
using BlockVIEM
using BlockVIEM: build_swg_basis, n_basis, n_tets, read_msh, mean_edge_length,
                 solve_cas_v2_orientations, aim_grid, build_aim_projection,
                 assemble_mass_matrix, duffy_reference_rule
import Gmsh: gmsh

const WL   = 0.637
const M_M  = 1.3315
const M_C  = 2.30 + 0.10im          # goethite crystal, representative at 637 nm
const FILL = parse(Float64, get(ENV, "FILL", "0.5"))
const D_VE_LIST = [0.15, 0.20, 0.30]
const SHAPES = [("sphere", 1.0), ("prolate_3", 1/3), ("oblate_3", 3.0)]   # AR = b/c
const BETAS  = [0.0, 30.0, 60.0, 90.0]
const N_PW = 10; const LC_GEOM = 0.30; const LC_FACTOR = 1.6
const PITCH_RATIO = 0.5; const PADDING = 4; const TOL = 1e-5; const MAXITER = 600

function mg_uniaxial(m_c, m_m, f)
    ec, em = m_c^2, m_m^2
    e_par  = f * ec + (1 - f) * em
    e_perp = em * ((1 + f) * ec + (1 - f) * em) / ((1 - f) * ec + (1 + f) * em)
    return sqrt(e_par), sqrt(e_perp), sqrt((e_par + 2e_perp) / 3)
end

function spheroid_mesh(b, c, lc)
    path = joinpath(tempdir(), "viem_fb_$(round(b,digits=6))_$(round(c,digits=6))_$(round(lc,digits=6)).msh")
    isfile(path) && return path
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
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

n_par, n_perp, n_iso = mg_uniaxial(M_C, M_M, FILL)
@printf("form birefringence probe  wl %.3f um  m_m %.4f  crystal %s  fill %.2f\n", WL, M_M, M_C, FILL)
@printf("  n_par %.3f%+.3fi  n_perp %.3f%+.3fi  dn %.3f   eps-average %.3f%+.3fi\n\n",
        real(n_par), imag(n_par), real(n_perp), imag(n_perp), real(n_par) - real(n_perp),
        real(n_iso), imag(n_iso))
m_worst = maximum(abs, (n_par, n_perp, n_iso))
eul = [(0.0, deg2rad(b), 0.0) for b in BETAS]

@printf("%-10s %-6s  %-18s  %s\n", "shape", "D_ve", "medium", join([@sprintf("|B/A|@%2.0f", b) for b in BETAS], "  ") * "   sinb-avg   |A|@0")
for (name, AR) in SHAPES, D in D_VE_LIST
    r = D / 2; b = r * AR^(1/3); c = r * AR^(-2/3)
    lc = min(LC_FACTOR * WL / (m_worst * N_PW), LC_GEOM * min(b, c))
    mesh  = read_msh(spheroid_mesh(b, c, lc))
    basis = build_swg_basis(mesh; include_boundary_faces = true)
    pitch = PITCH_RATIO * mean_edge_length(mesh)
    grid  = aim_grid(basis.mesh; pitch = pitch, padding = PADDING)
    proj  = build_aim_projection(basis, grid; poly_order = 2, stencil = 3)
    mass  = assemble_mass_matrix(basis)
    for (label, mvec) in (("uniaxial (z=needle)", [n_perp, n_perp, n_par]),
                          ("isotropic control",   [n_iso, n_iso, n_iso]))
        res = solve_cas_v2_orientations(basis, eul; wl_0 = WL, m_m = M_M, m_p = mvec,
                  method = :aim_gmres, tol = TOL, maxiter = MAXITER,
                  duffy_rule = duffy_reference_rule(5), symmetrize = true,
                  pitch = pitch, padding = PADDING, projection = proj, mass = mass)
        ratios = Float64[]; A0 = 0.0
        for (i, rr) in enumerate(res)
            A = (rr.S_fw_theta + rr.S_fw_phi) / 2; B = (rr.S_fw_theta - rr.S_fw_phi) / 2
            push!(ratios, abs(B) / abs(A)); i == 1 && (A0 = abs(A))
        end
        # sin(beta)-weighted average over the four betas (trapezoid on a coarse grid)
        w = sind.(BETAS); avg = sum(w .* ratios) / sum(w)
        @printf("%-10s %-6.2f  %-18s  %s   %.3f      %.4f   [N_tet %d]\n", name, D, label,
                join([@sprintf("%.3f", x) for x in ratios], "      "), avg, A0, n_tets(mesh))
        flush(stdout)
    end
end
