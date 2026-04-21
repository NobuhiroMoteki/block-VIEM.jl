# Inspect a block-DDA_Py-compatible HDF5 file produced by the VIEM solver.
#
# This is the Julia equivalent of block-DDA_Py/dda_results/check_h5py.ipynb.
# It reads the file structure, shows sweep parameters, checks completion
# status, and prints per-orientation and Mie-reference results for a
# user-selected condition.
#
# Usage:
#     julia --project=. viem_results/check_h5.jl [filename]

using HDF5
using Printf
using Statistics: mean, std

# ═══════════════════════════════════════════════════════════════════════════════
#  File to inspect (override via command-line argument)
# ═══════════════════════════════════════════════════════════════════════════════
filename = length(ARGS) >= 1 ? ARGS[1] :
           joinpath(@__DIR__, "pcas_ocbs_simulated_data.hdf5")

isfile(filename) || error("File not found: $filename")

# ═══════════════════════════════════════════════════════════════════════════════
#  Condition to inspect (change these indices)
# ═══════════════════════════════════════════════════════════════════════════════
const I_PAIR = 1   # index into wl_m_m_pairs  (1-based)
const I_MP   = 1   # index into m_p_xyz_list
const I_RV   = 1   # index into r_v_base_list
const I_BC   = 1   # index into bc_ratio_list
const I_AB   = 1   # index into ab_ratio_list
const I_BT   = 1   # index into gre_beta_list
const I_ORI  = 1   # orientation index to inspect in detail

# ═══════════════════════════════════════════════════════════════════════════════
#  Read and display
# ═══════════════════════════════════════════════════════════════════════════════
h5open(filename, "r") do f
    t  = f["target"]
    sd = t["simulated_data"]

    # ── file structure overview ───────────────────────────────────────────
    println("=== target attributes ===")
    for k in keys(attrs(t))
        println("  $k: $(read_attribute(t, k))")
    end

    println("\n=== target datasets ===")
    for name in keys(t)
        name == "simulated_data" && continue
        d = t[name]
        println("  $name: shape=$(size(d))  dtype=$(eltype(d))")
    end

    println("\n=== simulated_data datasets ===")
    for name in keys(sd)
        d = sd[name]
        println("  $name: shape=$(size(d))  dtype=$(eltype(d))")
    end

    # ── sweep parameters ─────────────────────────────────────────────────
    wl_m_m    = read(t["wl_m_m_pairs"])     # (N_pairs, 2) or (2, N_pairs)
    m_p_xyz   = read(t["m_p_xyz_list"])
    rv_list   = read(t["r_v_base_list"])
    bc_list   = read(t["bc_ratio_list"])
    ab_list   = read(t["ab_ratio_list"])
    bt_list   = read(t["gre_beta_list"])

    # HDF5.jl reads column-major; transpose if needed
    if ndims(wl_m_m) == 2 && size(wl_m_m, 1) == 2 && size(wl_m_m, 2) > 2
        wl_m_m = permutedims(wl_m_m)
    end
    if ndims(m_p_xyz) == 2 && size(m_p_xyz, 1) == 3 && size(m_p_xyz, 2) > 3
        m_p_xyz = permutedims(m_p_xyz)
    end

    N_pairs = size(wl_m_m, 1)
    N_mp    = size(m_p_xyz, 1)

    N_alpha = haskey(attrs(t), "N_alpha_ori") ? Int(attrs(t)["N_alpha_ori"]) : 0
    N_beta  = haskey(attrs(t), "N_beta_ori")  ? Int(attrs(t)["N_beta_ori"])  : 0
    N_gamma = haskey(attrs(t), "N_gamma_ori") ? Int(attrs(t)["N_gamma_ori"]) : 0
    num_ori = Int(attrs(t)["num_orientations"])

    println("\n=== sweep parameters ===")
    @printf("  wl_m_m_pairs  (%d pairs):\n", N_pairs)
    for i in 1:N_pairs
        @printf("    [%d]  wl_0=%.4f μm   m_m=%.4f\n", i, wl_m_m[i,1], wl_m_m[i,2])
    end
    @printf("  m_p_xyz_list  (%d entries):\n", N_mp)
    for i in 1:N_mp
        @printf("    [%d]  m_p_xyz = %s\n", i, m_p_xyz[i,:])
    end
    println("  r_v_base_list : $rv_list")
    println("  bc_ratio_list : $bc_list")
    println("  ab_ratio_list : $ab_list")
    println("  gre_beta_list : $bt_list")

    if N_alpha > 0
        @printf("  Orientation grid: N_alpha=%d, N_beta=%d, N_gamma=%d\n",
                N_alpha, N_beta, N_gamma)
        @printf("  Total orientations: %d  (= %d × %d × %d)\n",
                num_ori, N_alpha, N_beta, N_gamma)
    else
        @printf("  num_orientations : %d  (legacy random sampling)\n", num_ori)
    end

    # ── spheroid mode ────────────────────────────────────────────────────
    sph_mode = haskey(attrs(t), "spheroid_mode") ?
               (attrs(t)["spheroid_mode"] == 1) : false
    println("  Spheroid mode: $sph_mode")

    # ── completion status ────────────────────────────────────────────────
    mie_s = read(sd["S_fw_PCAS_mie"])   # (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt) complex
    done  = imag.(mie_s) .!= 0.0

    N_total = length(done)
    N_done  = count(done)
    @printf("\nCompleted: %d / %d conditions\n\n", N_done, N_total)

    for i_pair in 1:N_pairs
        wl_0 = wl_m_m[i_pair, 1]
        m_m  = wl_m_m[i_pair, 2]
        for i_mp in 1:N_mp
            mp = m_p_xyz[i_mp, :]
            frac  = count(done[i_pair, i_mp, :, :, :, :])
            total = length(done[i_pair, i_mp, :, :, :, :])
            mark = frac == total ? "✓" : (frac > 0 ? "…" : "✗")
            @printf("  %s pair[%d] wl=%.4f m_m=%.4f  m_p[%d] = %s  →  %d/%d\n",
                    mark, i_pair, wl_0, m_m, i_mp, mp, frac, total)
        end
    end

    # ── selected condition ───────────────────────────────────────────────
    idx6 = (I_PAIR, I_MP, I_RV, I_BC, I_AB, I_BT)

    wl_0     = wl_m_m[I_PAIR, 1]
    m_m      = wl_m_m[I_PAIR, 2]
    mp_sel   = m_p_xyz[I_MP, :]
    rv_sel   = rv_list[I_RV]
    bc_sel   = bc_list[I_BC]
    ab_sel   = ab_list[I_AB]
    bt_sel   = bt_list[I_BT]

    r_ve_val = read(sd["r_ve"])[I_RV, I_BC, I_AB, I_BT]

    @printf("\n\nSelected condition  idx6=%s\n", idx6)
    @printf("  wl_0=%.4f μm   m_m=%.4f\n", wl_0, m_m)
    @printf("  m_p_xyz = %s\n", mp_sel)
    @printf("  r_v_base=%.3f  bc=%.1f  ab=%.1f  beta=%.2f\n",
            rv_sel, bc_sel, ab_sel, bt_sel)
    @printf("  r_ve = %.4f μm\n", r_ve_val)

    # ── per-orientation results ──────────────────────────────────────────
    euler   = read(sd["Euler_angles"])[idx6..., :, :]     # (N_ori, 3)
    C_abs   = read(sd["C_abs"])[idx6..., :]               # (N_ori,)
    C_ext   = read(sd["C_ext"])[idx6..., :]
    S_fw_th = read(sd["S_fw_PCAS_theta"])[idx6..., :]     # (N_ori,) complex
    S_fw_ph = read(sd["S_fw_PCAS_phi"])[idx6..., :]
    S_bk    = read(sd["S_bk_OCBS"])[idx6..., :]

    println("\n=== VIEM per-orientation results ===")
    @printf("  %-24s  %14s  %14s\n", "quantity", "mean", "std")
    println("  " * "-"^54)

    function _stat_real(label, arr)
        valid = filter(!isnan, real.(arr))
        if isempty(valid)
            @printf("  %-24s  %14s\n", label, "(no data)")
        else
            @printf("  %-24s  %14.4e  %14.4e\n", label, mean(valid), std(valid))
        end
    end

    function _stat_ri(label, arr)
        valid = filter(x -> !isnan(real(x)), arr)
        if isempty(valid)
            for p in ("Re", "Im")
                @printf("  %-24s  %14s\n", "$p($label)", "(no data)")
            end
        else
            for (p, vals) in [("Re", real.(valid)), ("Im", imag.(valid))]
                @printf("  %-24s  %14.4e  %14.4e\n", "$p($label)", mean(vals), std(vals))
            end
        end
    end

    _stat_real("C_abs [um^2]",    C_abs)
    _stat_real("C_ext [um^2]",    C_ext)
    _stat_ri(  "S_fw_theta [um]", S_fw_th)
    _stat_ri(  "S_fw_phi [um]",   S_fw_ph)
    _stat_ri(  "S_bk_OCBS [um]",  S_bk)

    # Euler angle grid structure
    alpha_u = sort(unique(rad2deg.(euler[:, 1])))
    beta_u  = sort(unique(rad2deg.(euler[:, 2])))
    gamma_u = sort(unique(rad2deg.(euler[:, 3])))
    println("\nEuler angle grid:")
    @printf("  alpha: %d values  %s%s\n", length(alpha_u),
            alpha_u[1:min(6,end)], length(alpha_u) > 6 ? "..." : "")
    @printf("  beta:  %d values  %s%s\n", length(beta_u),
            beta_u[1:min(6,end)], length(beta_u) > 6 ? "..." : "")
    @printf("  gamma: %d values  %s%s\n", length(gamma_u),
            gamma_u[1:min(6,end)], length(gamma_u) > 6 ? "..." : "")

    # ── individual orientation ───────────────────────────────────────────
    if I_ORI <= length(C_abs)
        alpha_d, beta_d, gamma_d = rad2deg.(euler[I_ORI, :])
        @printf("\n=== orientation i_ori=%d ===\n", I_ORI)
        @printf("  Euler angles (α, β, γ) = (%.1f°, %.1f°, %.1f°)\n\n",
                alpha_d, beta_d, gamma_d)
        @printf("  C_abs      = %.4e  um^2\n", C_abs[I_ORI])
        @printf("  C_ext      = %.4e  um^2\n", C_ext[I_ORI])
        @printf("  S_fw_theta = %.4g %+.4gim  um\n",
                real(S_fw_th[I_ORI]), imag(S_fw_th[I_ORI]))
        @printf("  S_fw_phi   = %.4g %+.4gim  um\n",
                real(S_fw_ph[I_ORI]), imag(S_fw_ph[I_ORI]))
        @printf("  S_bk_OCBS  = %.4g %+.4gim  um\n",
                real(S_bk[I_ORI]), imag(S_bk[I_ORI]))
    end

    # ── Mie reference ────────────────────────────────────────────────────
    C_abs_mie = read(sd["C_abs_mie"])[idx6...]
    C_ext_mie = read(sd["C_ext_mie"])[idx6...]
    S_fw_mean_mie = read(sd["S_fw_PCAS_mie"])[idx6...]
    S_bk_mie  = read(sd["S_bk_OCBS_mie"])[idx6...]

    println("\n=== Mie reference (volume-equivalent sphere) ===")
    @printf("  C_abs_mie       = %.4e um^2\n", C_abs_mie)
    @printf("  C_ext_mie       = %.4e um^2\n", C_ext_mie)
    @printf("  S_fw_mean_mie   = %.4g %+.4gim\n",
            real(S_fw_mean_mie), imag(S_fw_mean_mie))
    @printf("  S_bk_mie        = %.4g %+.4gim\n",
            real(S_bk_mie), imag(S_bk_mie))

    # ── VIEM vs Mie comparison ───────────────────────────────────────────
    println("\n=== VIEM (orientation mean) vs Mie ===")
    @printf("  %-24s  %14s  %14s  %16s\n",
            "quantity", "VIEM mean", "Mie", "|VIEM-Mie|/|Mie|")
    println("  " * "-"^72)

    function _cmp_real(label, arr, mie_val)
        valid = filter(!isnan, real.(arr))
        if isempty(valid) || mie_val == 0.0
            @printf("  %-24s  %14s\n", label, "(no data)")
            return
        end
        v = mean(valid)
        rel = abs(v - mie_val) / abs(mie_val)
        @printf("  %-24s  %14.4e  %14.4e  %16.4f\n", label, v, mie_val, rel)
    end

    function _cmp_ri(label, arr, mie_val)
        valid = filter(x -> !isnan(real(x)), arr)
        if isempty(valid)
            for p in ("Re", "Im")
                @printf("  %-24s  %14s\n", "$p($label)", "(no data)")
            end
            return
        end
        v = mean(valid)
        for (p, vp, mp) in [("Re", real(v), real(mie_val)),
                             ("Im", imag(v), imag(mie_val))]
            lbl = "$p($label)"
            if mp == 0.0
                @printf("  %-24s  %14.4e  %14.4e  %16s\n", lbl, vp, mp, "(Mie=0)")
            else
                rel = abs(vp - mp) / abs(mp)
                @printf("  %-24s  %14.4e  %14.4e  %16.4f\n", lbl, vp, mp, rel)
            end
        end
    end

    _cmp_real("C_abs [um^2]",    C_abs,   C_abs_mie)
    _cmp_real("C_ext [um^2]",    C_ext,   C_ext_mie)
    _cmp_ri(  "S_fw_theta [um]", S_fw_th, S_fw_mean_mie)
    _cmp_ri(  "S_fw_phi [um]",   S_fw_ph, S_fw_mean_mie)
    _cmp_ri(  "S_bk [um]",       S_bk,    S_bk_mie)
end

println("\nDone.")
