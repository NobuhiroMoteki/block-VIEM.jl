#!/usr/bin/env python3
"""Compute the Au-doublet CAS-v2 error table at beta=pi/2 for all
refinement levels in `phase_a_memory.json` against the MSTM N=15
reference.

Writes the Markdown table and the log-log fitted exponents to stdout.
They are the source of docs/benchmark_results.md sections 8.1 and 8.2.
A fitted exponent is a convergence rate only where the error decreases
monotonically with the mesh size. Components for which it does not are
marked in the output.
"""
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).parent

def load_records():
    """Records of phase_a_memory.json, sorted from the coarsest level."""
    current = json.loads((BASE / "phase_a_memory.json").read_text())
    return sorted(current["records"], key=lambda r: r["lc_ratio"])

def load_mstm_au_beta_pi2():
    d = json.loads((BASE / "results_mstm.json").read_text())
    for rec in d["results"]:
        if rec.get("material") == "Au" and abs(rec["beta"] - math.pi / 2) < 1e-6:
            return rec
    raise RuntimeError("MSTM Au beta=pi/2 reference not found")

def rel_err(viem, mstm):
    return abs(viem - mstm) / abs(mstm)

def complex_rel_err(v_re, v_im, m_re, m_im):
    dre = v_re - m_re
    dim = v_im - m_im
    mag = math.hypot(m_re, m_im)
    return math.hypot(dre, dim) / mag

def fit_slope(lc_ratios, errors):
    """Fit log(err) ~ p * log(1/k) via least squares. Returns p."""
    xs = [math.log(1.0 / k) for k in lc_ratios]
    ys = [math.log(e) for e in errors]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den

def main():
    records = load_records()
    mstm = load_mstm_au_beta_pi2()

    print(f"# Au doublet convergence at beta=pi/2 (Phase A measurements)")
    print(f"# MSTM N=15 reference: |S_fw_mean|={math.hypot(mstm['S_fw_mean_re'], mstm['S_fw_mean_im']):.4e}")
    print()

    rows = []
    for rec in records:
        k = rec["lc_ratio"]
        beta_obs = next((o for o in rec["observables"]
                          if abs(o["beta"] - math.pi / 2) < 1e-6), None)
        if beta_obs is None:
            continue
        v_mean_re = beta_obs["S_fw_mean_re"]; v_mean_im = beta_obs["S_fw_mean_im"]
        v_th_re = beta_obs["S_fw_theta_re"]; v_th_im = beta_obs["S_fw_theta_im"]
        v_ph_re = beta_obs["S_fw_phi_re"];   v_ph_im = beta_obs["S_fw_phi_im"]

        err_mag  = rel_err(math.hypot(v_mean_re, v_mean_im),
                           math.hypot(mstm["S_fw_mean_re"], mstm["S_fw_mean_im"]))
        err_cplx = complex_rel_err(v_mean_re, v_mean_im,
                                    mstm["S_fw_mean_re"], mstm["S_fw_mean_im"])
        err_th_re = rel_err(v_th_re, mstm["S_fw_theta_re"])
        err_th_im = rel_err(v_th_im, mstm["S_fw_theta_im"])
        err_ph_re = rel_err(v_ph_re, mstm["S_fw_phi_re"])
        err_ph_im = rel_err(v_ph_im, mstm["S_fw_phi_im"])
        rows.append((k, err_mag, err_th_re, err_th_im, err_ph_re, err_ph_im))

    # Markdown table
    print("| lc / R | |S_fw_mean| | Re S_fw_theta | Im S_fw_theta | Re S_fw_phi | Im S_fw_phi |")
    print("|--------|-------------|---------------|---------------|--------------|--------------|")
    for k, *errs in rows:
        s = " | ".join(f"{e*100:.2f} %" for e in errs)
        print(f"| 1/{k}    | {s} |")

    # Fit slopes if >= 2 points
    if len(rows) >= 2:
        ks = [r[0] for r in rows]
        print()
        print(f"# Slopes p in err ~ (1/k)^p  (log-log least-squares, {len(rows)} points):")
        for i, name in enumerate(("|S_fw_mean|", "Re S_fw_th", "Im S_fw_th",
                                    "Re S_fw_ph", "Im S_fw_ph"), start=1):
            errs = [r[i] for r in rows]
            if all(e > 0 for e in errs):
                p = fit_slope(ks, errs)
                # ks runs from the coarsest level: a converging error decreases.
                monotone = all(a > b for a, b in zip(errs, errs[1:]))
                flag = "" if monotone else "   (error not monotone: not a rate)"
                print(f"  {name:16s} p = {p:+.2f}{flag}")

if __name__ == "__main__":
    main()
