#!/usr/bin/env python3
"""Compare DDA and VIEM (via Mie reference) at various discretization levels.

Runs block-DDA_Py on a sphere with the same physical parameters as the VIEM
Mie validation, at several dpl values to match the VIEM mesh N.

Usage: cd ~/Julia/block-VIEM.jl && python3 test/compare_dda_viem.py
"""
import sys
sys.path.insert(0, '/home/moteki/Python_in_WSL/block-DDA_Py')
sys.path.insert(0, '/home/moteki/Python_in_WSL/MieScat_Py')

import numpy as np
from miescat import miescat

# --- Physical parameters (same as VIEM Mie validation) ---
wl_0 = 10.0       # vacuum wavelength [arbitrary length unit]
m_m = 1.0          # medium RI
r_v = 1.0          # sphere radius
m_p_real = 1.5
m_p_imag = 0.01
m_p = m_p_real + 1j * m_p_imag

k = 2 * np.pi * m_m / wl_0
x = k * r_v
print(f"Sphere: r={r_v}, wl_0={wl_0}, m_p={m_p}, x={x:.4f}")
print(f"k = {k:.6f}")

# --- Mie reference ---
d_p = 2 * r_v
Qsca, Qext, Qabs, *_ = miescat(wl_0, m_m, d_p, m_p_real, m_p_imag, nang=3)
G = np.pi * r_v**2
C_ext_mie = Qext * G
C_abs_mie = Qabs * G
C_sca_mie = Qsca * G
print(f"Mie: C_ext={C_ext_mie:.6f}  C_abs={C_abs_mie:.6f}  C_sca={C_sca_mie:.6f}")
print()

# --- DDA at several dpl values ---
# Single orientation: alpha=0, beta=0, gamma=0 (lab = particle frame)
euler_angles = np.array([[0.0, 0.0, 0.0]])

from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

for dpl in [5, 8, 10, 13, 17, 20, 25]:
    d = wl_0 / (abs(m_p) * dpl)  # lattice spacing

    # Generate cubic lattice inside a sphere of radius r_v
    half_n = int(np.ceil(r_v / d)) + 1
    ix = np.arange(-half_n, half_n + 1)
    grid = np.array(np.meshgrid(ix, ix, ix, indexing='ij')).reshape(3, -1).T
    lattice_grid_points = grid * d
    dist = np.linalg.norm(lattice_grid_points, axis=1)
    is_in = dist <= r_v

    lattice_n = np.array([2*half_n+1, 2*half_n+1, 2*half_n+1], dtype=int)

    m_p_xyz = np.array([m_p, m_p, m_p])
    target = Target("sphere", lattice_n, d, lattice_grid_points, is_in, m_p_xyz, r_v)
    incfield = IncidentField(wl_0, m_m, euler_angles)
    dd = DiscreteDipoles(target, incfield)

    N_occ = dd.num_element_occupy
    actual_dpl = dd.dpl

    dd.set_interaction_matrix()
    dd.tol = 1e-6
    dd.itermax = 200
    dd.solve_matrix_equation()

    if not dd.converge:
        print(f"  dpl={dpl:2d}  N={N_occ:5d}  DID NOT CONVERGE")
        continue

    C_abs_dda = dd.compute_C_abs()[0]
    C_ext_dda = dd.compute_C_ext()[0]
    C_sca_dda = C_ext_dda - C_abs_dda

    err_abs = (C_abs_dda - C_abs_mie) / C_abs_mie * 100
    err_ext = (C_ext_dda - C_ext_mie) / C_ext_mie * 100

    print(f"  dpl={dpl:2d}  N={N_occ:5d}  dpl_eff={actual_dpl:.1f}  "
          f"C_abs={C_abs_dda:.6f} ({err_abs:+.1f}%)  "
          f"C_ext={C_ext_dda:.6f} ({err_ext:+.1f}%)")
