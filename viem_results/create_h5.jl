# Create an empty HDF5 file with the block-DDA_Py-compatible schema for
# storing VIEM CAS-v2 simulation results across a multi-dimensional
# parameter sweep.
#
# This is the Julia equivalent of block-DDA_Py/dda_results/create_h5py.ipynb.
#
# The HDF5 layout matches block-DDA_Py exactly:
#
#   /target
#     ├─ attrs: N_alpha_ori, N_beta_ori, N_gamma_ori, num_orientations, ...
#     ├─ wl_m_m_pairs         (N_pairs, 2)  Float64
#     ├─ m_p_xyz_list         (N_m_p, 3)    ComplexF64
#     ├─ r_v_base_list        (N_rv,)       Float64
#     ├─ bc_ratio_list        (N_bc,)       Float64
#     ├─ ab_ratio_list        (N_ab,)       Float64
#     ├─ gre_beta_list        (N_bt,)       Float64
#     └─ /simulated_data
#         ├─ r_ve              (N_rv, N_bc, N_ab, N_bt)                        Float64
#         ├─ Euler_angles      (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, L, 3) Float64
#         ├─ C_abs             (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, L)    Float64
#         ├─ C_ext             (...)                                           Float64
#         ├─ S_fw_PCAS_theta   (...)                                           ComplexF64
#         ├─ S_fw_PCAS_phi     (...)                                           ComplexF64
#         ├─ S_bk_OCBS         (...)                                           ComplexF64
#         ├─ C_abs_mie         (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt)       Float64
#         ├─ C_ext_mie         (...)                                           Float64
#         ├─ S_fw_PCAS_mie     (...)                                           ComplexF64
#         └─ S_bk_OCBS_mie     (...)                                           ComplexF64
#
# Usage:
#     julia --project=. viem_results/create_h5.jl

using HDF5

# ═══════════════════════════════════════════════════════════════════════════════
#  EDIT THIS SECTION to define the sweep parameters
# ═══════════════════════════════════════════════════════════════════════════════

# (wl_0 [μm], m_m) pairs
wl_m_m_pairs = [
    0.633  1.330;   # 633 nm, water
    0.8337 1.329    # 834 nm, water
]

# Particle refractive index: each row is [m_p_x, m_p_y, m_p_z]
m_p_xyz_list = ComplexF64[
    1.5+0.0im  1.5+0.0im  1.5+0.0im;
    1.55+0.0im 1.55+0.0im 1.55+0.0im
]

# Shape parameters
r_v_base_list = [0.1, 0.2, 0.3, 0.4]   # [μm]
bc_ratio_list = [3.0]
ab_ratio_list = [1.0]
gre_beta_list = [0.0]

# Orientation grid (deterministic ZYZ Euler angle grid)
N_alpha_ori = 12
N_beta_ori  = 10
N_gamma_ori = 12

# Output file
filename = joinpath(@__DIR__, "pcas_ocbs_simulated_data.hdf5")

# ═══════════════════════════════════════════════════════════════════════════════
#  (no edits needed below)
# ═══════════════════════════════════════════════════════════════════════════════

num_orientations = N_alpha_ori * N_beta_ori * N_gamma_ori

N_pairs = size(wl_m_m_pairs, 1)
N_m_p   = size(m_p_xyz_list, 1)
N_rv    = length(r_v_base_list)
N_bc    = length(bc_ratio_list)
N_ab    = length(ab_ratio_list)
N_bt    = length(gre_beta_list)

# Spheroid-mode detection: ab_ratio == 1 and gre_beta == 0
# When active, the solver only solves N_beta_ori orientations at α=0
# and fills the full grid analytically.
spheroid_mode = all(ab == 1.0 for ab in ab_ratio_list) &&
                all(bt == 0.0 for bt in gre_beta_list)

shape_geo  = (N_rv, N_bc, N_ab, N_bt)
shape_cond = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt)
shape_ori  = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, num_orientations)
shape_ang  = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, num_orientations, 3)

h5open(filename, "w") do f
    grp = create_group(f, "target")
    attrs(grp)["light_source"]       = "Thorlabs SLD830S-A10 @ I=140mA, T=25C"
    attrs(grp)["polarization_state"] = "left-handed circular: E0_theta=1/sqrt(2), E0_phi=1j/sqrt(2)"
    attrs(grp)["N_alpha_ori"]        = N_alpha_ori
    attrs(grp)["N_beta_ori"]         = N_beta_ori
    attrs(grp)["N_gamma_ori"]        = N_gamma_ori
    attrs(grp)["num_orientations"]   = num_orientations
    attrs(grp)["spheroid_mode"]      = spheroid_mode ? 1 : 0

    write_dataset(grp, "wl_m_m_pairs",  Float64.(wl_m_m_pairs))
    write_dataset(grp, "m_p_xyz_list",  ComplexF64.(m_p_xyz_list))
    write_dataset(grp, "r_v_base_list", Float64.(r_v_base_list))
    write_dataset(grp, "bc_ratio_list", Float64.(bc_ratio_list))
    write_dataset(grp, "ab_ratio_list", Float64.(ab_ratio_list))
    write_dataset(grp, "gre_beta_list", Float64.(gre_beta_list))

    sd = create_group(grp, "simulated_data")
    attrs(sd)["scattering_code"] = "block-VIEM.jl (volume integral equation method)"
    attrs(sd)["orientation"]     = (
        "Deterministic ZYZ Euler angle grid (alpha, beta, gamma). " *
        "alpha: equally spaced in [0,2pi), N_alpha divisions. " *
        "beta: cos(beta) equally spaced in (-1,1), N_beta equal-area divisions. " *
        "gamma: equally spaced in [0,2pi), N_gamma divisions. " *
        "Ordering: alpha slowest, gamma fastest. " *
        "Spheroid mode (ab_ratio=1, gre_beta=0): VIEM solves only N_beta " *
        "orientations; full grid filled analytically.")
    attrs(sd)["units"]        = "r_ve:[um], euler_angles:[rad], C:[um^2], S:[um]"
    attrs(sd)["S_definition"] = ("S(0)_theta = S11(0)+1j*S12(0), " *
                                 "S(0)_phi = S22(0)-1j*S21(0), " *
                                 "S(180) = (S11+S22+1j*S12-1j*S21)(180)/sqrt(2), " *
                                 "per Mishchenko 2000")

    # Helper: create a dataset with a definition attribute
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
end

println("Created $filename")
println("  Orientation grid: N_alpha=$N_alpha_ori, N_beta=$N_beta_ori, N_gamma=$N_gamma_ori")
println("  Total orientations: $num_orientations")
println("  Spheroid mode: $spheroid_mode")
println("  wl_m_m_pairs : $(size(wl_m_m_pairs, 1)) pairs")
println("  m_p_xyz_list : $(size(m_p_xyz_list, 1)) entries")
println("  Shape dims   : N_pairs=$N_pairs, N_m_p=$N_m_p, N_rv=$N_rv, N_bc=$N_bc, N_ab=$N_ab, N_bt=$N_bt")
println("  shape_cond   : $shape_cond")
println("  shape_ori    : $shape_ori")
