"""Run block-DDA_Py on a single oblate spheroid at a small set of orientations
and dump the PCAS forward-scattering amplitudes as JSON.

The output is consumed by `run_viem_spheroid_single.jl` and
`compare_viem_dda.py` to verify that VIEM and DDA give the same S_fw_theta,
S_fw_phi for an axisymmetric particle.

Usage (from inside the block-DDA_Py repo so that bl_dda can be imported):

    cd ~/Python_in_WSL/block-DDA_Py
    .venv/bin/python /path/to/run_dda_spheroid_single.py
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

# Make sure we can import block-DDA_Py regardless of cwd.
_BLOCK_DDA_ROOT = os.path.expanduser("~/Python_in_WSL/block-DDA_Py")
if _BLOCK_DDA_ROOT not in sys.path:
    sys.path.insert(0, _BLOCK_DDA_ROOT)

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles


# ---------------------------------------------------------------------------
# Physical parameters — edit to match the VIEM driver
# ---------------------------------------------------------------------------
WL_0      = 0.638           # vacuum wavelength [um]
M_M       = 1.0             # medium refractive index
M_P       = 1.5 + 0.0j      # particle refractive index (non-absorbing)
R_VE      = 0.20            # volume-equivalent radius [um]     (D_ve = 0.40)
BC_RATIO  = 3.0             # b / c — > 1 is oblate
AB_RATIO  = 1.0             # a / b — axisymmetric spheroid
DPL       = 17              # dipoles per wavelength inside particle

# Orientation grid (match VIEM):  alpha=0, gamma=0, single beta.
BETA_LIST = [np.pi/4]       # single tilt
EULER_NP  = np.column_stack([
    np.zeros(len(BETA_LIST)), np.array(BETA_LIST), np.zeros(len(BETA_LIST))
])                           # shape (L, 3)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "dda_result.json")


def main() -> None:
    rng = np.random.default_rng(12345)

    # Build the (non-random, beta=0) spheroid lattice
    gre = gaussian_ellipsoid_shape_model(
        r_v_base=R_VE, bc_ratio=BC_RATIO, ab_ratio=AB_RATIO, beta=0.0,
        wl_0=WL_0, m_p_xyz=np.array([M_P, M_P, M_P]), dpl=DPL,
    )
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist  = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)

    target = Target(
        shape_name=gre.name, lattice_n=lattice_n, lattice_lf=gre.lattice_lf,
        lattice_grid_points=grid, lattice_grid_points_is_in_target=is_in,
        m_p_xyz=np.array([M_P, M_P, M_P]), r_v_base=R_VE,
    )
    print(f"[DDA] N dipoles: {target.num_element_occupy}")
    print(f"[DDA] d_adj: {target.lattice_lf:.6f} um")
    print(f"[DDA] r_ve (rescaled): {target.ve_radius:.6f} um")

    inc = IncidentField(WL_0, M_M, EULER_NP)
    dd  = DiscreteDipoles(target, inc)
    dd.set_interaction_matrix()
    dd.solve_matrix_equation()

    if not dd.converge:
        raise RuntimeError("DDA solver did not converge")

    dd.compute_C_abs()
    dd.compute_C_ext()
    dd.compute_PCAS_observable_S_fw()

    # shape (L,)
    result = {
        "params": {
            "wl_0": WL_0,
            "m_m": M_M,
            "m_p_re": float(np.real(M_P)),
            "m_p_im": float(np.imag(M_P)),
            "r_ve_target": R_VE,
            "bc_ratio": BC_RATIO,
            "ab_ratio": AB_RATIO,
            "dpl": DPL,
            "n_dipoles": int(target.num_element_occupy),
            "r_ve_rescaled": float(target.ve_radius),
            "lattice_lf": float(target.lattice_lf),
        },
        "orientations": [
            {
                "alpha": float(EULER_NP[l, 0]),
                "beta":  float(EULER_NP[l, 1]),
                "gamma": float(EULER_NP[l, 2]),
                "S_fw_theta_re": float(np.real(dd.S_fw_PCAS_theta[l])),
                "S_fw_theta_im": float(np.imag(dd.S_fw_PCAS_theta[l])),
                "S_fw_phi_re":   float(np.real(dd.S_fw_PCAS_phi[l])),
                "S_fw_phi_im":   float(np.imag(dd.S_fw_PCAS_phi[l])),
                "C_ext": float(dd.C_ext[l]),
                "C_abs": float(dd.C_abs[l]),
            }
            for l in range(EULER_NP.shape[0])
        ],
    }

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DDA] wrote {OUTPUT}")

    for l, ori in enumerate(result["orientations"]):
        print(f"  l={l}  (α={ori['alpha']:.4f}, β={ori['beta']:.4f}, γ={ori['gamma']:.4f})")
        print(f"    S_fw_θ = {ori['S_fw_theta_re']:+.6e} {ori['S_fw_theta_im']:+.6e}j")
        print(f"    S_fw_φ = {ori['S_fw_phi_re']:+.6e} {ori['S_fw_phi_im']:+.6e}j")
        print(f"    C_ext  = {ori['C_ext']:.6e}")
        print(f"    C_abs  = {ori['C_abs']:.6e}")


if __name__ == "__main__":
    main()
