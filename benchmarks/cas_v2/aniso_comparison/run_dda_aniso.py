"""Run block-DDA_Py on a sphere with anisotropic refractive index
for cross-validation against block-VIEM.jl.

Usage:
    cd ~/Python/block-DDA_Py
    .venv/bin/python <path>/run_dda_aniso.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

_BLOCK_DDA_ROOT = os.path.expanduser("~/Python/block-DDA_Py")
if _BLOCK_DDA_ROOT not in sys.path:
    sys.path.insert(0, _BLOCK_DDA_ROOT)

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

WL_0 = 0.638
M_M  = 1.0
DPL  = 17
R_VE = 0.20

# Anisotropic cases
CASES = [
    {"name": "iso",   "m_p_xyz": [1.5+0j,   1.5+0j,   1.5+0j  ]},
    {"name": "mild",  "m_p_xyz": [1.55+0j,  1.5+0j,   1.45+0j ]},
    {"name": "strong","m_p_xyz": [1.6+0j,   1.5+0j,   1.4+0j  ]},
]

BETA_ORI = [np.pi/4, np.pi/2]
EULER_NP = np.column_stack([
    np.zeros(len(BETA_ORI)), np.array(BETA_ORI), np.zeros(len(BETA_ORI))
])

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "dda_aniso_results.json")


def run_case(case):
    rng = np.random.default_rng(42)
    m_p_xyz = np.array(case["m_p_xyz"])

    gre = gaussian_ellipsoid_shape_model(
        r_v_base=R_VE, bc_ratio=1.0, ab_ratio=1.0, beta=0.0,
        wl_0=WL_0, m_p_xyz=m_p_xyz, dpl=DPL)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)

    target = Target(gre.name, lattice_n, gre.lattice_lf, grid, is_in,
                    m_p_xyz, R_VE)
    print(f"  Case {case['name']}: N={target.num_element_occupy}, "
          f"r_ve={target.ve_radius:.5f}, m_p={m_p_xyz}")

    inc = IncidentField(WL_0, M_M, EULER_NP)
    dd = DiscreteDipoles(target, inc)
    dd.set_interaction_matrix()
    dd.solve_matrix_equation()
    assert dd.converge, f"DDA did not converge for case {case['name']}"

    dd.compute_C_abs()
    dd.compute_C_ext()
    dd.compute_PCAS_observable_S_fw()

    oris = []
    for l in range(EULER_NP.shape[0]):
        oris.append({
            "beta_ori": float(EULER_NP[l, 1]),
            "S_fw_theta_re": float(np.real(dd.S_fw_PCAS_theta[l])),
            "S_fw_theta_im": float(np.imag(dd.S_fw_PCAS_theta[l])),
            "S_fw_phi_re":   float(np.real(dd.S_fw_PCAS_phi[l])),
            "S_fw_phi_im":   float(np.imag(dd.S_fw_PCAS_phi[l])),
            "C_ext": float(dd.C_ext[l]),
            "C_abs": float(dd.C_abs[l]),
        })
        print(f"    β={EULER_NP[l,1]:.4f}  "
              f"S_θ={dd.S_fw_PCAS_theta[l]:+.5e}  "
              f"S_φ={dd.S_fw_PCAS_phi[l]:+.5e}  "
              f"C_ext={dd.C_ext[l]:.4e}")

    return {
        "name": case["name"],
        "m_p_xyz_re": [float(np.real(m)) for m in m_p_xyz],
        "m_p_xyz_im": [float(np.imag(m)) for m in m_p_xyz],
        "n_dipoles": int(target.num_element_occupy),
        "r_ve_dda": float(target.ve_radius),
        "orientations": oris,
    }


def main():
    print("=" * 70)
    print("block-DDA_Py anisotropic cross-validation")
    print("=" * 70)
    all_results = {"wl_0": WL_0, "m_m": M_M, "r_ve": R_VE, "cases": []}
    for case in CASES:
        all_results["cases"].append(run_case(case))
    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
