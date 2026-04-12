"""
Verification 3: same-h DDA vs VIEM-SWG comparison on a sphere.

This script computes Mie cross sections for a sphere using block-DDA_Py
at several lattice spacings and compares against the analytical Mie value.
The output is meant to be cross-referenced with our Julia VIEM-SWG mesh
convergence study (benchmarks/rt0/b1_mesh_convergence.jl) at MATCHING h.

Method:
- Build a sphere of radius 1.0 using gaussian_ellipsoid_shape_model with
  bc_ratio=1, ab_ratio=1, beta=0 (this gives a deterministic sphere with
  no Gaussian deformation).
- The DDA lattice spacing d is set by `dpl = wl_0 / (|m_p| * d)`. So we
  pick `dpl` to match a target d.
- Run for two contrast cases (low and high) at several d values.
- Print C_abs, C_sca, C_ext and relative errors vs Mie at the
  volume-equivalent radius of the actual lattice fill.

Output is human-readable and JSON for downstream comparison with
the VIEM Julia results.

Run with the block-DDA_Py virtualenv:
   ~/Python_in_WSL/block-DDA_Py/.venv/bin/python \\
       benchmarks/rt0/v3_dda_vs_viem.py
"""
import sys, os, json, time
sys.path.insert(0, '/home/moteki/Python_in_WSL/block-DDA_Py')

import numpy as np

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s


def make_sphere_target(r_v_base, m_p, wl_0, dpl):
    """Build a deterministic sphere DDA target.
    bc_ratio=1, ab_ratio=1, beta=0  →  spherical (no GRE deformation).
    """
    m_p_xyz = np.array([m_p, m_p, m_p], dtype=np.complex128)
    rng = np.random.default_rng(0)
    gre = gaussian_ellipsoid_shape_model(
        r_v_base=r_v_base, bc_ratio=1.0, ab_ratio=1.0, beta=0.0,
        wl_0=wl_0, m_p_xyz=m_p_xyz, dpl=dpl)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist  = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    target = Target(gre.name, lattice_n, gre.lattice_lf, grid, is_in,
                    m_p_xyz, r_v_base)
    return target


def run_one(label, m_p, wl_0, dpl):
    print(f"\n--- {label}  m={m_p}  wl_0={wl_0}  dpl={dpl} ---", flush=True)
    target = make_sphere_target(r_v_base=1.0, m_p=m_p, wl_0=wl_0, dpl=dpl)
    N = target.num_element_occupy
    d_lattice = target.lattice_lf
    V_target = target.total_vol
    r_ve = target.ve_radius
    print(f"  N_dipole = {N}", flush=True)
    print(f"  d_lattice = {d_lattice:.4f}  (after rescale)", flush=True)
    print(f"  V_filled = {V_target:.4f}", flush=True)
    print(f"  r_ve = {r_ve:.4f}", flush=True)

    # Single orientation: alpha=beta=gamma=0 (lab frame)
    euler_angles = np.zeros((1, 3), dtype=np.float64)
    inc = IncidentField(wl_0, m_m=1.0, euler_angles=euler_angles)
    dd  = DiscreteDipoles(target, inc)
    dd.itermax = 200
    dd.tol = 1e-6
    dd.set_interaction_matrix()
    t0 = time.time()
    dd.solve_matrix_equation()
    t_solve = time.time() - t0
    print(f"  solve time = {t_solve:.1f}s, converged={dd.converge}", flush=True)
    if not dd.converge:
        return None

    C_abs = float(dd.compute_C_abs()[0])
    C_ext = float(dd.compute_C_ext()[0])
    C_sca = C_ext - C_abs

    # Mie reference at the same r_ve
    Q_sca, Q_abs, Q_ext, S_fw_mie, S_bk_mie = mie_compute_q_and_s(
        wl_0=wl_0, m_m=1.0, r_p=r_ve, m_p=m_p, nang=3)
    G = np.pi * r_ve ** 2  # geometric cross section
    C_abs_mie = float(Q_abs) * G
    C_sca_mie = float(Q_sca) * G
    C_ext_mie = float(Q_ext) * G

    err_abs = 100 * abs(C_abs - C_abs_mie) / abs(C_abs_mie)
    err_sca = 100 * abs(C_sca - C_sca_mie) / abs(C_sca_mie)
    err_ext = 100 * abs(C_ext - C_ext_mie) / abs(C_ext_mie)

    print(f"  C_abs: DDA={C_abs:.4e}  Mie={C_abs_mie:.4e}  err={err_abs:.2f}%", flush=True)
    print(f"  C_sca: DDA={C_sca:.4e}  Mie={C_sca_mie:.4e}  err={err_sca:.2f}%", flush=True)
    print(f"  C_ext: DDA={C_ext:.4e}  Mie={C_ext_mie:.4e}  err={err_ext:.2f}%", flush=True)

    return {
        'label': label,
        'm_p_real': float(np.real(m_p)),
        'm_p_imag': float(np.imag(m_p)),
        'wl_0': wl_0,
        'dpl': dpl,
        'N_dipole': N,
        'd_lattice': d_lattice,
        'V_filled': float(V_target),
        'r_ve': float(r_ve),
        't_solve': t_solve,
        'C_abs': C_abs, 'C_abs_mie': C_abs_mie, 'err_abs_pct': err_abs,
        'C_sca': C_sca, 'C_sca_mie': C_sca_mie, 'err_sca_pct': err_sca,
        'C_ext': C_ext, 'C_ext_mie': C_ext_mie, 'err_ext_pct': err_ext,
    }


# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------
results = []

# Scenario 1: low contrast m=1.5+0.01i, wl_0=10  (k0=0.628, ka≈0.6)
# Target lattice spacings d ≈ {0.7, 0.5, 0.35, 0.25, 0.15, 0.10}
# dpl = wl_0 / (|m_p| * d)
m_p_low = 1.5 + 0.01j
wl_0_low = 10.0
m_abs_low = abs(m_p_low)
for d_target in [0.7, 0.5, 0.35, 0.25, 0.15, 0.10]:
    dpl = wl_0_low / (m_abs_low * d_target)
    res = run_one(
        label=f"low_lc_d={d_target}",
        m_p=m_p_low, wl_0=wl_0_low, dpl=dpl)
    if res is not None:
        results.append(res)

# Scenario 2: high contrast eps=10+1i, ka=0.316
m_p_hi = np.sqrt(10.0 + 1.0j)
wl_0_hi = 2 * np.pi / 0.316
m_abs_hi = abs(m_p_hi)
for d_target in [0.7, 0.5, 0.35, 0.25, 0.15, 0.10]:
    dpl = wl_0_hi / (m_abs_hi * d_target)
    res = run_one(
        label=f"hi_ct_d={d_target}",
        m_p=m_p_hi, wl_0=wl_0_hi, dpl=dpl)
    if res is not None:
        results.append(res)

# ----------------------------------------------------------------------------
# Save results to JSON for cross-reference with Julia VIEM data
# ----------------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(__file__), 'v3_dda_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n=== Results written to {out_path} ===", flush=True)

print("\n=== SUMMARY (DDA) ===", flush=True)
print(f"{'label':25s} {'N':>6} {'d':>8} {'C_abs%':>10} {'C_sca%':>10} {'C_ext%':>10}", flush=True)
print("-" * 75, flush=True)
for r in results:
    print(f"{r['label']:25s} {r['N_dipole']:6d} {r['d_lattice']:8.4f} "
          f"{r['err_abs_pct']:9.2f}% {r['err_sca_pct']:9.2f}% {r['err_ext_pct']:9.2f}%", flush=True)
