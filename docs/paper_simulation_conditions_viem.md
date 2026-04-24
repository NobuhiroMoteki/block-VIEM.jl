# Paper Simulation Conditions — block-VIEM.jl

Authoritative reference for the VIEM-side conditions of the paper-production
sweeps.  Mirrors `block-DDA_Py/docs/paper_simulation_conditions_dda.md` for
direct shape × material × N_DOF comparison between the two solvers.

- **Repository**: `~/Julia/block-VIEM.jl`
- **Code version**: v0.7.6 (commit on `main`, tag pending)
- **Authoring date**: 2026-04-24
- **Sister project**: [block-DDA_Py](file:///home/moteki/Python/block-DDA_Py) (paper-symmetric)

---

## 1. Software environment

| Item | Value |
|---|---|
| Language | Julia 1.12.5 |
| Toolchain pin | `juliaup override` set to `1.12.5` for this project |
| Package manifest | `Manifest.toml` tracked, `Pkg.update()` / `Pkg.resolve()`禁止 |
| Threading | `julia --project=. -t auto` (matches `Sys.CPU_THREADS`) |
| BLAS threads | default (12 on this machine; do **not** force `BLAS.set_num_threads(1)`) |
| Test status | 14,422 / 14,422 passing |

---

## 2. Physical constants

| Quantity | Value | Notes |
|---|---|---|
| Vacuum wavelength λ₀ | **0.638 μm** | hardcoded in [viem_results/paper/_common.jl::WL_PAPER](../viem_results/paper/_common.jl) |
| Background medium m_m | **1.0** (vacuum) | `M_M_PAPER` |
| Time convention | `exp(+jωt)` (BH83 / block-DDA_Py compatible) | outgoing wave is `exp(-jk₀R)/(4πR)` |

---

## 3. Particle materials (paper "high" 差替済 v0.7.5)

| ID | n_p @ λ=0.638 μm | a_eq list (μm) | Status |
|---|---|---|---|
| `n15` (低屈折率) | 1.5 + 0.01i | 0.05, 0.10, 0.20, 0.40 | active |
| `n20` (paper "high", v0.7.5+) | **2.0 + 0.0i** | 0.05, 0.10, 0.20, 0.40 | active |
| `n317` (legacy high, ≤ v0.7.4) | 3.17 + 0.16i | (reference 用、新規計算には使わない) | retained for legacy comparison |
| `Au` (Johnson & Christy 1972) | **0.17525 + 3.4830i** | 0.05, 0.10, 0.20 | active |

`a_eq` は体積等価球の半径。Au は `|m_p|≈3.49` で λ/(\|m_p\|·N_pw) 制約による
細かいメッシュが必要なため `a_eq ≤ 0.20 μm` に限定（[CLAUDE.md §2](../CLAUDE.md) 参照）。

**v0.7.5 の n317 → n20 差替理由**: DDA 側 `gre × n317 × r_v=0.4 × L=100` で
peak RSS 215 GB / wall 195 min / GMRES iter=100 stagnation を観測。
n20 (非吸収) で N_DOF が ~4× 縮小し両ソルバとも安定収束見込み。
旧 `*_n317.hdf5` 結果は reference として保持（block-DDA_Py 側 CLAUDE.md §6 と対称）。

---

## 4. Particle shapes

| ID | 形状パラメータ | mesh entry point |
|---|---|---|
| `sphere` | GRE: `bc_ratio=1, ab_ratio=1, gre_beta=0` | [src/gre_mesh.jl](../src/gre_mesh.jl) |
| `oblate` | GRE: `bc_ratio=3, ab_ratio=1, gre_beta=0` | 同上 |
| `gre` | GRE: `bc_ratio=1, ab_ratio=1, gre_beta=0.2` | 同上 |
| `doublet` | 2 球クラスター（後述）| [src/aggregate_mesh.jl](../src/aggregate_mesh.jl) |

### 4.1 形状パラメータの体積等価半径換算

- **球 (sphere)**: 半径 `r = a_eq`
- **扁平回転楕円体 (oblate)**: 3 半軸 (a, b, c) で `a = b = 3c`, `c = a_eq / 9^(1/3)`
- **GRE (Gaussian Random Ellipsoid)**: ベース楕円体 (a=b=c=a_eq) に β_gre=0.2 のガウス変形
  - 形状確定の RNG: `Random.MersenneTwister(12345)`（再現性のため固定）
- **2 球クラスター (doublet)**: 等半径 2 球
  - monomer 半径: `R = a_eq / 2^(1/3) ≈ 0.7937 a_eq`
  - 軸方向 surface-to-surface gap: `g = 0.1 R`
  - **doublet 軸 = 粒子 z 軸**（spheroid mode の α-展開公式が適用可能になる前提条件）
  - a_eq = 0.05, 0.10, 0.20, 0.40 μm → R ≈ 0.0397, 0.0794, 0.1587, 0.3175 μm

---

## 5. Mesh discretization

### 5.1 Adaptive characteristic length lc

[src/gre_mesh.jl::adaptive_lc](../src/gre_mesh.jl#L84) は次の最小値を採択:

```text
lc = min( λ₀ / (|m_p|_max · N_pw),    # 波長制約 (N_pw = 10)
          c_min / N_per_radius,        # 幾何制約 (N_per_radius = 3)
          0.3 · c_min / 3 )            # GRE β > 0 時の相関長制約
```

- `|m_p|` は ComplexF64 のモジュラス最大（Au は `|0.18+3.48i|≈3.49`）
- 本番計算は既定 `N_pw = 10`, `N_per_radius = 3`
- `aggregate_mesh.jl::adaptive_lc_aggregate` は doublet 用、c_min = 各 monomer 半径

### 5.2 lc 収束性スタディ用 factor

[viem_results/paper/run_lc_convergence.jl::LC_FACTORS](../viem_results/paper/run_lc_convergence.jl):

```text
factor = 1.5, 1.0, 0.7, 0.5, 0.35
lc = factor × adaptive_lc
```

各 (shape, material) で a_eq=0.1 μm を代表サイズとし 5 点測定。
本番 sweep は factor=1.0 （adaptive_lc そのまま）を採用。

---

## 6. Orientation grid

### 6.1 多配向 ZYZ Euler グリッド

[viem_results/paper/_common.jl](../viem_results/paper/_common.jl):

| パラメータ | 値 |
|---|---|
| `N_alpha` (α: 0 → 2π) | 4 |
| `N_beta` (β: cos 等分) | 5 |
| `N_gamma` (γ: 0 → 2π) | 5 |
| 公称配向数 | 4 × 5 × 5 = **100** |

α と γ は等間隔開区間 [0, 2π)、β は cos β を等区間で割って `acos(cos β)` で取得（球面一様）。

### 6.2 Spheroid mode (軸対称粒子の解析展開)

`shape_kind == "doublet"` または `(ab_ratio==1 && gre_beta==0)` のとき自動有効。
Block-Krylov は `(α=0, γ=0)` の **N_β=5 配向のみ**実 solve、α-expansion で 100 配向に解析展開:

```text
A_fw   = (S_s + S_p) / 2
B_fw   = (S_s - S_p) / 2
S_fw_θ = A_fw + B_fw · exp(2iα)
S_fw_φ = A_fw - B_fw · exp(2iα)
S_bk   = S_bk_0 · exp(2iα)
```

| shape | spheroid_mode | block solver L |
|---|---|---|
| sphere | ✓ | 5 |
| oblate | ✓ | 5 |
| doublet | ✓ (粒子 z 軸対称) | 5 |
| GRE (β=0.2) | ✗ | 100 |

---

## 7. Solver settings

### 7.1 本番デフォルト

[viem_results/run_viem.jl](../viem_results/run_viem.jl):

| 定数 | 値 | 意味 |
|---|---|---|
| `SOLVER_METHOD` | `:aim_gmres` | block-GMRES (v0.7.1+ デフォルト、incremental Givens QR は v0.7.4+) |
| `SOLVER_TOL` | `1.0e-5` | 相対残差 ‖B − A·X‖_F / ‖B‖_F ≤ TOL で converged |
| `MAXITER` | **200** | v0.7.6: 100 → 200 (n20 では通常 ~30-50 iter で収束、Au stagnation や large L に headroom) |
| `N_PW` | 10 | dpl-equivalent (適応 lc の波長制約係数) |
| `DUFFY_ORDER` | 5 | Duffy 求積の次数 |
| `AIM_PITCH_RATIO` | 0.5 | AIM grid pitch = ratio × mean_edge_length |
| `AIM_PADDING` | 4 | AIM grid padding cells |
| `RNG_SEED` | 12345 | GRE 形状 RNG (block-DDA_Py と同一 seed) |
| `REUSE_PROJECTION_PER_SHAPE` | true | shape slot 毎に worst-case mesh + projection + mass を 1 回構築・再利用 |

### 7.2 BiCGSTAB との比較について

v0.7.6 で paper のスコープから **block-BiCGSTAB を除外**。論文は shape × material × N_DOF
スケーリングに焦点を絞る。BiCGSTAB をリトライしたい場合は
[run_rhs_scaling.jl::METHODS](../viem_results/paper/run_rhs_scaling.jl) に
`(:bicgstab, :aim_bicgstab)` を追記すれば schema 変更なしで復活可能。

### 7.3 BLAS

`BLAS.get_num_threads()` がデフォルト (12)。`-t auto` で起動した Julia の `Threads.@threads`
ループと併用しても問題なし（block-Krylov 内の `qr()`, 行列積は LAPACK 経由で並列化）。
v0.7.4 以前は serial bottleneck があったが O(kL²) Givens QR への書換で解消。

---

## 8. Output observables

[viem_results/paper/_common.jl::create_paper_h5](../viem_results/paper/_common.jl) の HDF5 schema。
全データセット `/target/simulated_data/` 直下、shape は配向数 100 を末尾に持つ。

### 8.1 物理観測量

| dataset | dtype | shape | 単位 | 定義 |
|---|---|---|---|---|
| `C_ext` | Float64 | (1,1,N_rv,1,1,1,100) | μm² | 消散断面積 (per orientation) |
| `C_abs` | Float64 | 同上 | μm² | 吸収断面積 |
| `S_fw_PCAS_theta` | ComplexF64 | 同上 | μm | 前方散乱振幅 θ 成分 = `S11(0) + i·S12(0)` |
| `S_fw_PCAS_phi` | ComplexF64 | 同上 | μm | 前方散乱振幅 φ 成分 = `S22(0) − i·S21(0)` |
| `S_bk_OCBS` | ComplexF64 | 同上 | μm | 後方散乱振幅 = `(S11+S22+i·S12−i·S21)(180°)/√2` |
| `Euler_angles` | Float64 | (...,100,3) | rad | (α, β, γ) per orientation |
| `r_ve` | Float64 | (N_rv,1,1,1) | μm | 離散化メッシュ体積から計算した実効体積等価半径 |

### 8.2 体積等価球 Mie 参照値

| dataset | dtype | shape |
|---|---|---|
| `C_ext_mie`, `C_abs_mie` | Float64 | (1,1,N_rv,1,1,1) |
| `S_fw_PCAS_mie` | ComplexF64 | 同上 |
| `S_bk_OCBS_mie` | ComplexF64 | 同上 |

doublet 形状でも書き込まれるが、Rayleigh 領域以外では参考値（厳密解は MSTM、§11 参照）。

### 8.3 規格化光学断面積（解析時に算出）

論文 Figure では `Q_X = C_X / (π · a_eq²)` を使用。`a_eq` は HDF5 の `r_v_base_list` を使う
(理想値、メッシュ離散化前)、または `r_ve` を使う (実効値、誤差解析時)。

### 8.4 偏光・入射状態

`/target` group attrs:
- `light_source`: 形状ファイル毎にハードコード（例: "(paper) λ=0.638 μm, n=2.0+0.0i, sphere"）
- `polarization_state`: "left-handed circular: E0_theta=1/sqrt(2), E0_phi=1j/sqrt(2)"
- `S_definition`: BH83 → MI02 → CAS-v2 変換式（Mishchenko 2000 準拠）

block-DDA_Py の create_h5py.ipynb と byte 互換。

---

## 9. /target/cost/ — per-slot cost & solver diagnostics (v0.7.3+)

| dataset | dtype | shape | 意味 |
|---|---|---|---|
| `t_build_s` | Float64 | shape_cond | mesh build + SWG basis 時間 [s] |
| `t_setup_s` | Float64 | shape_cond | AIM grid + projection + mass 時間 [s] |
| `t_solve_s` | Float64 | shape_cond | block-Krylov solve 時間 (観測量計算込み) [s] |
| `t_total_s` | Float64 | shape_cond | end-to-end (build + setup + solve + observables + Mie + HDF5) [s] |
| `peak_rss_bytes` | Int64 | shape_cond | per-slot peak RSS [bytes] (`/proc/self/status:VmRSS` 0.2s sampler) |
| `n_tet` | Int64 | shape_cond | テトラ数 (DDA `n_cuboid` に対応) |
| `n_dof` | Int64 | shape_cond | SWG basis 自由度数 (DDA `n_occ` に対応) |
| `mean_edge_length` | Float64 | shape_cond | h̄ [μm] (DDA `lattice_lf` に対応) |
| `iters` | Int64 | shape_cond | block-Krylov 反復数 |
| `converged` | Int8 | shape_cond | 1 / 0 |
| `solver_err` | Float64 | shape_cond | 最終相対残差 |
| `residual_history` | Float64 | shape_cond × MAXITER_HISTORY (=200, v0.7.6+) | per-iter 残差、`NaN` パディング (v0.7.5+) |

`shape_cond = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt)`。
DDA 側 `/target/cost/` と 8 fields (時間 + peak RSS + iters + converged + solver_err + residual_history) は **bit-for-bit 一致**。残り 3 つは VIEM 名称、DDA との 1:1 物理的対応。

### 9.1 RSS sampler

[viem_results/rss_monitor.jl](../viem_results/rss_monitor.jl):
- daemon `Threads.@spawn` で `/proc/self/status:VmRSS` を 0.2s 間隔ポーリング
- `RSSMonitor.reset!(mon)` で per-slot ベースラインリセット
- `Sys.maxrss()` (process-lifetime monotone) は per-slot 計測に不適

---

## 10. Logging

`run_viem.jl` の `_log(msg)` は HH:MM:SS タイムスタンプ付きで stdout に出力:

```
[09:35:01]   mesh+projection+mass cached (1.2s)
[09:35:21]     [spheroid, L=5]: converged ✓  iters=23
[09:35:22]   solve: 18.5s  iters=23  converged=true
[09:35:22]   C_ext(mean)=2.1e-01  Mie C_ext=2.0e-01  [3 done, 0 skipped / 4 total]  t_total=20.1s  peak_RSS=2.3 GB
```

stdout は orchestrator が `/tmp/auto_kickoff.log` に redirect する想定。flush は per-slot
完了時のみ実行されるため、長時間 slot の中間進捗は見えない（HDF5 lock や `ps` で外部観測）。

---

## 11. Runners (entry points)

### 11.1 1 スイープファイル = 1 (shape × material) の生成

[viem_results/paper/](../viem_results/paper/) 配下に 12 個（v0.7.6 では 11 active + 1 legacy n317 reference）:

```
sphere_n15.jl   sphere_n20.jl   sphere_Au.jl
oblate_n15.jl   oblate_n20.jl   oblate_Au.jl
gre_n15.jl      gre_n20.jl      gre_Au.jl
doublet_n15.jl  doublet_n20.jl  doublet_Au.jl
```

各ファイルは [_common.jl::create_paper_h5](../viem_results/paper/_common.jl) を呼び、
`viem_results/paper/{shape}_{material}.hdf5` を空テンプレート（schema のみ）で生成。

### 11.2 本番計算

```bash
julia --project=. -t auto viem_results/run_viem.jl viem_results/paper/sphere_n20.hdf5
```

Resume: `S_fw_PCAS_mie` の imag が非ゼロのスロットはスキップ（block-DDA_Py と同方針）。

### 11.3 事前見積

```bash
julia --project=. -t auto viem_results/estimate_cost.jl viem_results/paper/sphere_n20.hdf5
```

各 shape slot で worst-case (wl_min, |m_p|_max) のメッシュを実構築 → N_DOF 取得 →
empirical scaling で RSS / setup / solve 時間を見積。**24h 超 or RSS > 90% MemAvailable**
で `⚠️` フラグ。

### 11.4 lc 収束スタディ (a_eq=0.1 μm 固定、5 factor)

```bash
julia --project=. -t auto viem_results/paper/run_lc_convergence.jl <shape> <material>
# shape    ∈ {sphere, oblate, gre}
# material ∈ {n15, n20, Au}  (n317 legacy も accept)
```

出力: `viem_results/paper/convergence_{shape}_{material}.hdf5`、
`/target/lc_convergence/` 直下に lc_factor / lc / N_tet / N_DOF / Q_* / S_* /
**residual_history (v0.7.5+)** などを記録。

### 11.5 RHS-scaling 診断 (block-Krylov の L 依存性)

```bash
julia --project=. -t auto viem_results/paper/run_rhs_scaling.jl viem_results/paper/sphere_n20.hdf5
```

`L = 1, 2, 4, 8, 16, 32, 64, 128` の 8 点を `:aim_gmres` のみで測定 (v0.7.6, BiCGSTAB は scope 外)。
出力先: 入力 HDF5 の `/target/rhs_scaling/gmres/` サブグループ。
配向は `Random.MersenneTwister(12345)` の uniform-sphere quasi-random、L=1 ⊂ ⋯ ⊂ L=128 で nested。

### 11.6 MSTM 厳密解 (doublet のみ)

```bash
julia --project=/home/moteki/Julia/MSTMforCAS.jl \
    viem_results/paper/run_mstm_reference.jl viem_results/paper/doublet_n20.hdf5
# → viem_results/paper/mstm_doublet_n20.hdf5
```

`truncation_order = 15` 固定（[README.md §475-482](../README.md) 参照、Au 含む両材料で完全収束）。
スキーマは `/target/{a_eq_um, R_monomer_um, gap_um, beta_rad, observables/{Q_*, S_fw_*, S_bk}, diagnostics/}` の
(a_eq, β) 2D グリッド（α/γ は VIEM の解析展開と物理的に一致するため不要）。

---

## 12. Resume / Escalation policy

[CLAUDE.md §5](../CLAUDE.md):

1. **Resume**: 既存 HDF5 の `S_fw_PCAS_mie` が non-zero なら該当スロットスキップ。`run_viem.jl` 自動。
2. **Escalation 閾値**: estimate_cost で 1 slot あたり推定 **wall time > 24 h** または
   推定 **peak RSS > 90% MemAvailable** で `⚠️` フラグ → 起動前に必ずユーザー確認。
3. **Solver failure**: try / catch で NaN 埋めして次スロットへ続行（block-DDA_Py 準拠）。
   Ctrl+C のみ途中停止扱い。

---

## 13. File naming conventions

```
viem_results/paper/
├── _common.jl                               # schema helper, material constants
├── {shape}_{material}.jl                    # 11 active sweep generators (v0.7.6)
├── {shape}_{material}.hdf5                  # production results
├── pilot_sphere_n20.jl                      # smoke-test wrapper (a_eq=0.1 only)
├── convergence_{shape}_{material}.hdf5      # lc convergence study output
├── mstm_doublet_{material}.hdf5             # MSTM exact reference (doublet only)
├── run_lc_convergence.jl                    # lc convergence runner
├── run_rhs_scaling.jl                       # RHS-scaling diagnostic runner
└── run_mstm_reference.jl                    # MSTM driver (runs in MSTMforCAS env)
```

---

## 14. Cross-reference: block-DDA_Py side

DDA 側との対称項目（要同期）:

| 項目 | VIEM | DDA |
|---|---|---|
| 材料定数 | `_common.jl::N_LOW/N_20/N_HIGH/N_AU` | `_common.py::N_LOW/N_20/N_HIGH/N_AU` |
| 形状パラメータ (`R = a_eq/2^(1/3)`, gap=0.1R) | `run_viem.jl::_doublet_along_z` | `shape_model/two_sphere_cluster.py` |
| HDF5 schema (`/target/cost/` etc.) | `_common.jl::create_paper_h5` | `_common.py::create_paper_h5` |
| RNG seed (12345) | `_common.jl`, `run_rhs_scaling.jl` | 同上 |
| MAXITER (=200, v0.7.6) | `run_*.jl::MAXITER` | `scripts/run_*.py::MAXITER` |
| residual_history (length 200, NaN-pad) | `BlockSolveResult.residual_history` | `bl_*_mvp_fft` returns + per-slot HDF5 |
| RHS-scaling solver scope (GMRES のみ, v0.7.6) | `run_rhs_scaling.jl::METHODS` | `scripts/run_rhs_scaling.py` (要 sync, ユーザー側で対応) |

---

## 15. Version history (paper-relevant)

| Version | Commit | 主な変更 |
|---|---|---|
| v0.7.0 | `0a71cac` | paper-production scaffolding (12 generators, estimator, diagnostics) |
| v0.7.1 | `3583795` | デフォルトソルバ `:aim_gmres`、RHS-scaling L extended 4→6 点、a_eq cap 0.5→0.4 μm |
| v0.7.2 | `21b99c1` | block-Krylov defaults `tol=1e-5, maxiter=100` |
| v0.7.3 | `a11f8a5` | `/target/cost/` group (DDA 対称、11 dataset) |
| v0.7.4 | `1843456` | `block_gmres` per-iter cost `O(k²L³) → O(kL²)` (incremental Givens) |
| v0.7.5a | `d2902fe` | "high" material n317 → n20 (2.0+0.0i) 差替 |
| v0.7.5b | `6f6987a` | `BlockSolveResult.residual_history`, HDF5 全所に per-iter trace 追加 |
| **v0.7.6** | `d2bceaf` | **MAXITER 100 → 200**、**RHS-scaling GMRES のみ** (BiCGSTAB scope 外) |

---

## 16. References

- [`CLAUDE.md`](../CLAUDE.md) (本リポジトリ): 論文計算フェーズの運用ガイド
- [`README.md`](../README.md): 公開向けプロジェクト解説、benchmark 記録
- [`docs/theory_note.tex`](theory_note.tex): VIEM-AIM 理論ノート (formal derivations)
- [`docs/io_spec.md`](io_spec.md): block-DDA_Py 互換 I/O 仕様
- block-DDA_Py: `~/Python/block-DDA_Py` (姉妹プロジェクト)
- MSTMforCAS.jl: `~/Julia/MSTMforCAS.jl` (doublet 厳密解)
