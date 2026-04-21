"""Run block-DDA_Py on several GRE shapes (beta > 0) and save results +
the coarse-grid deformation field s_coarse so that block-VIEM.jl can
reconstruct the identical shape for cross-validation.

Usage:
    cd ~/Python/block-DDA_Py
    .venv/bin/python <path>/run_dda_gre.py
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

_BLOCK_DDA_ROOT = os.path.expanduser("~/Python/block-DDA_Py")
if _BLOCK_DDA_ROOT not in sys.path:
    sys.path.insert(0, _BLOCK_DDA_ROOT)

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

# ---------------------------------------------------------------------------
# Physical parameters (shared with VIEM script)
# ---------------------------------------------------------------------------
WL_0 = 0.638          # vacuum wavelength [um]
M_M  = 1.0            # medium refractive index
M_P  = 1.5 + 0.01j    # particle RI (weakly absorbing)
DPL  = 17              # dipoles per wavelength

# Orientation grid: (alpha=0, beta, gamma=0)
BETA_ORI = [np.pi / 4, np.pi / 2]
EULER_NP = np.column_stack([
    np.zeros(len(BETA_ORI)), np.array(BETA_ORI), np.zeros(len(BETA_ORI))
])

# GRE test cases
CASES = [
    {"name": "A", "r_v_base": 0.20, "bc_ratio": 2.0, "ab_ratio": 1.0, "beta": 0.10, "seed": 100},
    {"name": "B", "r_v_base": 0.20, "bc_ratio": 3.0, "ab_ratio": 1.0, "beta": 0.15, "seed": 200},
    {"name": "C", "r_v_base": 0.20, "bc_ratio": 1.5, "ab_ratio": 1.5, "beta": 0.20, "seed": 300},
]

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "dda_gre_results.json")


def run_case(case: dict) -> dict:
    rng = np.random.default_rng(case["seed"])
    m_p_xyz = np.array([M_P, M_P, M_P])

    gre = gaussian_ellipsoid_shape_model(
        r_v_base=case["r_v_base"],
        bc_ratio=case["bc_ratio"],
        ab_ratio=case["ab_ratio"],
        beta=case["beta"],
        wl_0=WL_0,
        m_p_xyz=m_p_xyz,
        dpl=DPL,
    )

    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist
    )

    target = Target(
        shape_name=gre.name,
        lattice_n=lattice_n,
        lattice_lf=gre.lattice_lf,
        lattice_grid_points=grid,
        lattice_grid_points_is_in_target=is_in,
        m_p_xyz=m_p_xyz,
        r_v_base=case["r_v_base"],
    )

    print(f"  [DDA] Case {case['name']}: N={target.num_element_occupy}, "
          f"d_adj={target.lattice_lf:.5f}, r_ve={target.ve_radius:.5f}")

    inc = IncidentField(WL_0, M_M, EULER_NP)
    dd = DiscreteDipoles(target, inc)
    dd.set_interaction_matrix()
    dd.solve_matrix_equation()

    if not dd.converge:
        raise RuntimeError(f"DDA did not converge for case {case['name']}")

    dd.compute_C_abs()
    dd.compute_C_ext()
    dd.compute_PCAS_observable_S_fw()

    # --- recover the s_coarse field for Julia consumption ---
    # Re-run the GRE surface generation with the SAME seed to capture s_coarse.
    rng2 = np.random.default_rng(case["seed"])
    N_theta, N_phi = 25, 100
    a_ax = gre.ab_ratio * gre.bc_ratio
    c_ax = np.cbrt(case["r_v_base"] ** 3 / (gre.ab_ratio * gre.bc_ratio ** 2))
    b_ax = c_ax * gre.bc_ratio
    a_ax = b_ax * gre.ab_ratio
    h0 = c_ax ** 2 / a_ax
    lc_corr = 0.3 * c_ax

    theta_c = np.linspace(0, np.pi, N_theta)
    phi_c = np.linspace(0, 2 * np.pi, N_phi)
    theta_mesh, phi_mesh = np.meshgrid(theta_c, phi_c, indexing="ij")

    x0 = a_ax * np.sin(theta_mesh) * np.cos(phi_mesh)
    y0 = b_ax * np.sin(theta_mesh) * np.sin(phi_mesh)
    z0 = c_ax * np.cos(theta_mesh)

    N = N_theta * N_phi
    px = x0.ravel(); py = y0.ravel(); pz = z0.ravel()
    d2 = ((px[:, None] - px[None, :]) ** 2
          + (py[:, None] - py[None, :]) ** 2
          + (pz[:, None] - pz[None, :]) ** 2)
    cov = case["beta"] ** 2 * np.exp(-0.5 * d2 / lc_corr ** 2)

    s_vec_mean = np.zeros(N)
    s_samples = rng2.multivariate_normal(s_vec_mean, cov,
                                         size=1, method="eigh").squeeze()
    s_coarse = s_samples.reshape(N_theta, N_phi)

    # Build result dict
    orientations = []
    for l in range(EULER_NP.shape[0]):
        orientations.append({
            "alpha": float(EULER_NP[l, 0]),
            "beta_ori": float(EULER_NP[l, 1]),
            "gamma": float(EULER_NP[l, 2]),
            "S_fw_theta_re": float(np.real(dd.S_fw_PCAS_theta[l])),
            "S_fw_theta_im": float(np.imag(dd.S_fw_PCAS_theta[l])),
            "S_fw_phi_re":   float(np.real(dd.S_fw_PCAS_phi[l])),
            "S_fw_phi_im":   float(np.imag(dd.S_fw_PCAS_phi[l])),
            "C_ext": float(dd.C_ext[l]),
            "C_abs": float(dd.C_abs[l]),
        })
        print(f"    β_ori={EULER_NP[l,1]:.4f}  "
              f"S_θ={dd.S_fw_PCAS_theta[l]:+.5e}  "
              f"S_φ={dd.S_fw_PCAS_phi[l]:+.5e}")

    return {
        "name": case["name"],
        "r_v_base": case["r_v_base"],
        "bc_ratio": case["bc_ratio"],
        "ab_ratio": case["ab_ratio"],
        "beta": case["beta"],
        "seed": case["seed"],
        "n_dipoles": int(target.num_element_occupy),
        "r_ve_dda": float(target.ve_radius),
        "theta_grid": theta_c.tolist(),
        "phi_grid": phi_c.tolist(),
        "s_coarse": s_coarse.tolist(),
        "orientations": orientations,
    }


def main() -> None:
    print("=" * 70)
    print("block-DDA_Py GRE cross-validation benchmark")
    print("=" * 70)
    print(f"  wl_0={WL_0}, m_m={M_M}, m_p={M_P}, dpl={DPL}")

    all_results = {"wl_0": WL_0, "m_m": M_M,
                   "m_p_re": float(np.real(M_P)), "m_p_im": float(np.imag(M_P)),
                   "cases": []}

    for case in CASES:
        result = run_case(case)
        all_results["cases"].append(result)

    with open(OUTPUT, "w") as f:
        json.dump(all_results, f)   # compact — file is ~1 MB
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
