"""Compare DDA and VIEM single-spheroid PCAS forward amplitudes.

Reads `dda_result.json` (from run_dda_spheroid_single.py) and
`viem_result.json` (from run_viem_spheroid_single.jl) and prints a side-by-side
table of S_fw_theta, S_fw_phi, and C_ext with relative errors.
"""
from __future__ import annotations

import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))


def load(name: str) -> dict:
    with open(os.path.join(DIR, name)) as f:
        return json.load(f)


def rel_err_complex(a_re: float, a_im: float, b_re: float, b_im: float) -> float:
    num = ((a_re - b_re) ** 2 + (a_im - b_im) ** 2) ** 0.5
    den = (b_re ** 2 + b_im ** 2) ** 0.5
    return num / den if den > 0 else float("inf")


def fmt_complex(re: float, im: float) -> str:
    sign = "+" if im >= 0 else "-"
    return f"{re:+.6e} {sign} {abs(im):.6e}j"


def main() -> int:
    dda = load("dda_result.json")
    viem = load("viem_result.json")

    print("=" * 78)
    print("DDA vs VIEM: single oblate spheroid, CAS-v2 forward amplitude comparison")
    print("=" * 78)

    p_dda, p_viem = dda["params"], viem["params"]
    print(f"  wl_0     = {p_dda['wl_0']} μm      (DDA/VIEM)")
    print(f"  m_p      = {p_dda['m_p_re']:+} {p_dda['m_p_im']:+}j")
    print(f"  r_ve     = DDA {p_dda.get('r_ve_rescaled', p_dda.get('r_ve_target')):.6f}"
          f"   VIEM mesh {p_viem['r_ve_mesh']:.6f}")
    print(f"  bc_ratio = {p_dda['bc_ratio']}   (oblate)")
    print(f"  DDA  dipoles:  {p_dda['n_dipoles']}  (dpl={p_dda['dpl']})")
    print(f"  VIEM tets/basis: {p_viem['n_tets']} / {p_viem['n_basis']}  (lc={p_viem['lc']})")
    print()

    n = len(dda["orientations"])
    assert n == len(viem["orientations"])

    worst_rel_theta = 0.0
    worst_rel_phi = 0.0
    for k in range(n):
        d = dda["orientations"][k]
        v = viem["orientations"][k]
        print(f"  Orientation {k}:  α={d['alpha']:.4f}  β={d['beta']:.4f}  γ={d['gamma']:.4f}")

        print("    S_fw_θ:")
        print(f"      DDA  = {fmt_complex(d['S_fw_theta_re'], d['S_fw_theta_im'])}")
        print(f"      VIEM = {fmt_complex(v['S_fw_theta_re'], v['S_fw_theta_im'])}")
        rel = rel_err_complex(v['S_fw_theta_re'], v['S_fw_theta_im'],
                              d['S_fw_theta_re'], d['S_fw_theta_im'])
        worst_rel_theta = max(worst_rel_theta, rel)
        print(f"      |ΔS|/|S|_DDA = {rel:.3%}")

        print("    S_fw_φ:")
        print(f"      DDA  = {fmt_complex(d['S_fw_phi_re'], d['S_fw_phi_im'])}")
        print(f"      VIEM = {fmt_complex(v['S_fw_phi_re'], v['S_fw_phi_im'])}")
        rel = rel_err_complex(v['S_fw_phi_re'], v['S_fw_phi_im'],
                              d['S_fw_phi_re'], d['S_fw_phi_im'])
        worst_rel_phi = max(worst_rel_phi, rel)
        print(f"      |ΔS|/|S|_DDA = {rel:.3%}")

        print(f"    C_ext:  DDA={d['C_ext']:.6e}  VIEM={v['C_ext']:.6e}")
        if d['C_ext'] > 0:
            ce = abs(v['C_ext'] - d['C_ext']) / d['C_ext']
            print(f"      rel err      = {ce:.3%}")
        print()

    print(f"  Worst-case relative error:  S_fw_θ {worst_rel_theta:.3%},  S_fw_φ {worst_rel_phi:.3%}")
    if max(worst_rel_theta, worst_rel_phi) < 0.05:
        print("  ✓ PASS (< 5 %)")
        return 0
    print("  ~ MARGINAL or FAIL (≥ 5 %) — both codes are discretisation-limited")
    return 1


if __name__ == "__main__":
    sys.exit(main())
