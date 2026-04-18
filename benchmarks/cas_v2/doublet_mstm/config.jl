# Shared benchmark configuration for the 2-sphere doublet comparison
# between block-VIEM.jl and MSTMforCAS.jl.
#
# Both solver drivers `run_viem.jl` and `run_mstm.jl` include this file
# and consume the constants below, so the two codes solve the exact same
# physical problem.
#
# Geometry
# --------
# Two equal-radius spheres separated by a small GAP along the particle
# z-axis.  The GAP is deliberately positive so the spheres are
# topologically disjoint — this lets MSTM treat each monomer as an
# isolated scatterer (the premise of the exact multipole expansion),
# while block-VIEM meshes each monomer as a separate tetrahedral volume.
#
# Orientations
# ------------
# The "orientation angle" BETA is the angle between the incidence
# direction and the doublet axis.  In block-VIEM we realise BETA by
# rotating the incidence via `cas_orientation(0, β, 0)` (particle frame
# fixed along lab-z).  In MSTM, with incidence fixed along lab-z, we
# rotate the two monomer centres about the lab-y axis by β so their
# separation vector makes angle β with lab-z.
#
# Materials
# ---------
# Two cases are benchmarked:
#   - "PS"  (polystyrene-like high-RI dielectric): m_p = 1.60 + 0.01i
#   - "Au"  (gold at λ₀ = 638 nm, Johnson & Christy 1972):
#           m_p = 0.17525 + 3.4830i

const R_MONOMER = 0.030      # monomer radius [μm]
const GAP       = 0.003      # centre-to-surface gap along axis [μm]
const D_CENTRE  = 2 * R_MONOMER + GAP   # centre-to-centre separation [μm]
const WL_0      = 0.638      # wavelength in vacuum [μm]
const M_MEDIUM  = 1.0        # background medium refractive index (real)

const BETAS = (0.0, π/4, π/2)    # angle between incidence and doublet axis [rad]

const MATERIALS = (
    (name = "PS", m_p = 1.60 + 0.01im),
    (name = "Au", m_p = 0.17525 + 3.4830im),
)

# block-VIEM AIM solver options (used by run_viem.jl)
const VIEM_LC_BASE   = R_MONOMER / 5  # base mesh size [μm] (overridden per material)
const VIEM_TOL       = 1e-7
const VIEM_MAXITER   = 400
const VIEM_PADDING   = 4

# MSTM solver options (used by run_mstm.jl)
const MSTM_USE_FFT   = false
const MSTM_TOL       = 1e-8
# VSWF truncation order.  The auto-selected Wiscombe-based value is N=3
# for our small-x monomers, which under-truncates the doublet forward
# amplitude for plasmonic Au by ~5% on |S_fw| (|m·x|≈1 amplifies higher
# multipoles through the inter-sphere translation coupling).  With
# MSTMforCAS ≥ 0.4.3 (Miller ψ_n + Miller j_n translation + mie_vecs
# sizing fixes) the reference is numerically stable up to N=15 at
# x≈0.3.  Au converges to ≤ 1e-4 by N=10 and to reference (~1e-7) at
# N=15.  We force N=15 for both materials — cheap for a 2-sphere
# aggregate (<0.1 s per orientation) and gives a fully converged
# exact reference for the VIEM benchmark.
const MSTM_TRUNC_ORDER = 15

# Output file paths (relative to this directory)
_OUT_DIR = @__DIR__
const VIEM_JSON = joinpath(_OUT_DIR, "results_viem.json")
const MSTM_JSON = joinpath(_OUT_DIR, "results_mstm.json")
const REPORT_MD = joinpath(_OUT_DIR, "comparison.md")

# Adaptive mesh size — finer when the particle has strong |m| (e.g. Au)
function lc_for_material(m_p::ComplexF64)
    return min(VIEM_LC_BASE, WL_0 / (abs(m_p) * 12))
end

# Consistent record for serialization
function record_key(material_name::String, beta::Float64)
    return string(material_name, "/beta_", round(beta; digits=6))
end
