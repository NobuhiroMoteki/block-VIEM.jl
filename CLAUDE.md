# 物理光学・計算電磁気学ソルバー (VIEM-AIM in Julia) — 論文計算フェーズ運用ガイド

## 1. プロジェクト現状 (post-implementation)

本リポジトリはソルバの実装・検証を完了している。**新規モジュールを設計する段階ではない**；論文掲載用の本番計算を `src/` の既存実装で実行するフェーズ。

- 検証状況 (2026-04-21): 14,422 / 14,422 テスト全パス on Julia 1.12.5。Mie 球面 (低/高屈折率・Au)、CAS-v2 2 球 doublet (vs MSTM)、AIM vs 直接法、block-DDA_Py 出力互換の各検証で回帰なし。
- 実装モジュール (src/):
  - メッシュ・基底: [gmsh_io.jl](src/gmsh_io.jl), [mesh.jl](src/mesh.jl), [swg.jl](src/swg.jl), [gre_mesh.jl](src/gre_mesh.jl), [aggregate_mesh.jl](src/aggregate_mesh.jl), [spheroid_sweep_io.jl](src/spheroid_sweep_io.jl)
  - 特異積分: [quadrature.jl](src/quadrature.jl), [duffy.jl](src/duffy.jl), [triangle_quad.jl](src/triangle_quad.jl), [surface_integrals.jl](src/surface_integrals.jl)
  - Green 関数・インピーダンス: [green.jl](src/green.jl), [impedance.jl](src/impedance.jl)
  - AIM: [aim_grid.jl](src/aim_grid.jl), [aim_projection.jl](src/aim_projection.jl), [aim_toeplitz.jl](src/aim_toeplitz.jl), [aim_operator.jl](src/aim_operator.jl)
  - ソルバ: [solver.jl](src/solver.jl), [block_krylov.jl](src/block_krylov.jl)
  - 入射・後処理: [incident.jl](src/incident.jl), [postprocess.jl](src/postprocess.jl)
- 本番計算のエントリポイント:
  - GRE / spheroid / single sphere スイープ: [viem_results/create_h5.jl](viem_results/create_h5.jl), [viem_results/run_viem.jl](viem_results/run_viem.jl), [viem_results/check_h5.jl](viem_results/check_h5.jl)
  - 2 球 doublet: [benchmarks/cas_v2/doublet_mstm/run_viem.jl](benchmarks/cas_v2/doublet_mstm/run_viem.jl) が generic aggregate ドライバ

## 2. 論文の計算マトリクス

波長固定 **λ₀ = 0.638 μm**、真空背景 `n_m = 1.0`。

| 形状 | 形状パラメータ | 配向プロトコル |
| --- | --- | --- |
| 単球 | 球 | 単配向 + 球面一様多配向（RHS 数スイープ 1,4,16,64） |
| 2 球クラスター | 等半径 doublet, **touching gap = 0.1 R** along doublet axis (README と同条件) | 単配向 + 球面一様多配向 + 代表配向 β = 0°, 30°, 60°, 90° |
| 扁平回転楕円体 (oblate) | b/c = 3, a/b = 1 | 単配向 + 球面一様多配向 + 代表配向 |
| GRE | a/b = b/c = 1, β_gre = 0.2 | 単配向 + 球面一様多配向 |

| 物質 | n_p @ λ=0.638 μm | a_eq (μm) |
| --- | --- | --- |
| 低屈折率 | 1.5 + 0.01i | 0.05, 0.1, 0.2, 0.5 |
| 高屈折率 | 3.17 + 0.16i | 0.05, 0.1, 0.2, 0.5 |
| Au (J&C 1972) | **0.17525 + 3.4830i** (λ=0.638 μm でハードコード) | 0.05, 0.1, 0.2 (0.5 μm は対象外) |

観測量:

- **Qext, Qsca, Qabs** (体積等価半径基準、規格化 `Q = C / (π a_eq²)`)
- **CAS-v2 複素散乱振幅**: `S_fw_θ`, `S_fw_φ`, `S_bk` (μm 単位、block-DDA_Py 互換定義)

### 形状パラメータの換算

- **2 球 doublet の半径換算**: 体積等価半径 `a_eq` から monomer 半径 `R = a_eq / 2^(1/3) ≈ 0.7937 · a_eq`、軸方向 gap `g = 0.1 · R` (README Benchmark 節と同比)。a_eq = 0.05, 0.1, 0.2, 0.5 μm → R ≈ 0.0397, 0.0794, 0.1587, 0.3969 μm。
- **扁平回転楕円体**: 3 半軸 `(a, b, c)`, `a = b`, `b/c = 3` → `c = (a_eq³ / (b/c)²)^(1/3) = a_eq / 9^(1/3)`, `a = b = 3c`。
- **GRE**: `GREParams(a_eq=a_eq, ab_ratio=1.0, bc_ratio=1.0, beta=0.2)` でそのまま構築。

## 3. メッシュ指定プロトコル

[src/gre_mesh.jl:84](src/gre_mesh.jl#L84) の `adaptive_lc` と [src/aggregate_mesh.jl:587](src/aggregate_mesh.jl#L587) の `adaptive_lc_aggregate` を使う。公式は

```text
lc = min( λ₀ / (|m_p|_max · N_pw),    # 波長制約, N_pw = 10 (既定)
          c_min / N_per_radius,        # 幾何制約, N_per_radius = 3 (既定、c_min = 最小半軸)
          0.3 · c_min / 3 )            # GRE β > 0 時の相関長制約 (c_min 基準)
```

- `|m_p|` は anisotropy 対応のモジュラス最大（Au は `|0.17525 + 3.4830i| ≈ 3.488` → lc ≈ 0.018 μm、保守側）
- 本番スイープは既定 `N_pw = 10`, `N_per_radius = 3`
- これらは `gre_mesh(...; wl_0, m_p_max, N_pw)` / `mesh_sphere_aggregate(...; wl_0, m_p_max)` の内部で自動呼び出し

## 4. lc 収束性の評価（論文 Figure 用）

各 (物質 × 形状) の組み合わせについて、**a_eq = 0.1 μm を代表サイズ**として lc を 5 段階スイープして Qext, Qsca, Qabs, |S_fw_θ| の収束性を測る:

```text
factor = 1.5, 1.0, 0.7, 0.5, 0.35   # adaptive_lc を 1.0 とする倍率
lc = factor × adaptive_lc(...)
```

- 本番大規模スイープの lc は **factor = 0.7** を採用（収束済み領域の中央値寄り）
- 結果は `viem_results/paper/convergence_{shape}_{material}.hdf5` に保存（スキーマは本番と共通、`lc_factor` attribute 追加）
- 論文 Figure には VIEM の O(h²) と DDA の O(h^0.5) の対比（README [:148-155](README.md#L148-L155) 参照実績）を掲載

## 5. 実行前チェックリスト（一計算 < 24 h を強制）

HDF5 スイープを起動する前に必ず:

1. **N_tet / N_DOF の見積**: GRE/sphere/spheroid は `adaptive_lc` と形状から、doublet は `aggregate_mesh` の実メッシュ生成で確認
2. **メモリ見積**: [README.md:619-645](README.md#L619-L645) の phase-A 実測から `RSS ≈ 100 kB/DOF` 程度を目安とし、`N_DOF × N_RHS × 2 (complex)` を block-Krylov の Krylov 空間分として加算
3. **Wall-time 見積**: setup 時間は N_DOF に対し実質線形、per-iteration cost は `N_DOF log N_DOF × N_RHS`。BiCGSTAB 反復数の目安は過去実績（N_RHS=1 で数十〜数百回、N_RHS 増で緩やかに増加）
4. **推定 > 24 h または RSS > マシン物理メモリ → 必ずユーザーに確認**

見積ツール: [viem_results/estimate_cost.jl](viem_results/estimate_cost.jl) (新規作成予定)。引数は `create_h5.jl` と同じパラメータスキーマで、各スイープ点の (N_DOF, RSS_est, t_setup_est, t_solve_est_per_orient) を表示しスイープ全体の合計も出す。

## 6. HDF5 スイープ運用規約

- 1 HDF5 = 1 物質 × 1 形状（Au, 低屈折率, 高屈折率それぞれ別ファイル、doublet も別）
- ファイル命名: `viem_results/paper/{shape}_{material}.hdf5`
  - `sphere_n15.hdf5`, `sphere_n317.hdf5`, `sphere_Au.hdf5`
  - `doublet_n15.hdf5`, `doublet_n317.hdf5`, `doublet_Au.hdf5`
  - `oblate_n15.hdf5`, `oblate_n317.hdf5`, `oblate_Au.hdf5`
  - `gre_n15.hdf5`, `gre_n317.hdf5`, `gre_Au.hdf5`
  - lc 収束: `convergence_{shape}_{material}.hdf5`
- 既存 HDF5 の C_ext ≠ 0 エントリは `run_viem.jl` の再計算スキップ機能に従い中断再開可能
- 多配向 block-Krylov の **反復数 と 1 配向あたり end-to-end 所要時間** は別スクリプト [viem_results/paper/run_rhs_scaling.jl](viem_results/paper/run_rhs_scaling.jl) で診断:
  - RHS 数スイープ `L = 1, 2, 4, 8, 16, 32`（2 のべき乗、計 6 点で iter 数のスケーリングを解像）
  - **2 ソルバを同時計測**: `block-BiCGSTAB` (`:aim_bicgstab`) と `block-GMRES` (`:aim_gmres`) を同じ shape slot, 同じメッシュ + projection + mass で評価
  - 結果は `/target/rhs_scaling/{bicgstab,gmres}/` のサブグループに `iters`, `converged`, `t_total_s`, `t_end2end_per_orient_s` を書き込み（`L_values`, `n_dof`, `n_tet` は両メソッド共通として親グループに）
  - 各 shape slot ごとに測定（worst-case mesh 共有、固定 RNG seed の uniform-sphere 配向を nested に L=1 ⊂ L=2 ⊂ L=4 ⊂ L=8 ⊂ L=16 ⊂ L=32）
  - L=64 はパイロット計測で block-BiCGSTAB の stagnation を頻発したため除外（near-linearly-dependent RHS による breakdown、[src/block_krylov.jl:62-65](src/block_krylov.jl#L62-L65) 既知制約）。L=32 を実用上限として採用。論文の Figure では BiCGSTAB / GMRES の収束反復数を併記し、BiCGSTAB が早く解ける一方 GMRES は monotone な収束を保証する点を比較
- **doublet 厳密解（MSTM 参照）は別 HDF5 で管理**: [viem_results/paper/run_mstm_reference.jl](viem_results/paper/run_mstm_reference.jl) を MSTMforCAS.jl env (`~/Julia/MSTMforCAS.jl`) で実行し、対応する `mstm_<basename>.hdf5` を生成
  - `truncation_order = 15`（[README.md:475-482](README.md#L475-L482) 準拠、Au 含む両材料で完全収束）
  - スキーマは `/target/{a_eq_um, R_monomer_um, gap_um, beta_rad, observables/{Q_*, S_fw_*, S_bk}, diagnostics/{n_iterations, converged}}`、軸対称性により (a_eq, β) の 2D グリッドのみ（α/γ は VIEM の解析展開と物理的に一致するため不要）
  - 体積等価球の Mie 値は run_viem.jl の `compute_mie_reference` が自動的に書き込む（doublet では Rayleigh サニティチェック用途、厳密解ではない）

## 7. 不変条件（回帰ガード）

以下を壊す変更は行わない:

- [test/test_mie_validation.jl](test/test_mie_validation.jl), [test/test_cas_v2.jl](test/test_cas_v2.jl), [test/test_block_krylov.jl](test/test_block_krylov.jl), [test/test_aggregate_mesh.jl](test/test_aggregate_mesh.jl) は常時グリーン
- block-DDA_Py 互換 HDF5 スキーマ（`C_ext`, `C_abs`, `C_sca` (= C_ext − C_abs), `S_fw_PCAS_theta`, `S_fw_PCAS_phi`, `S_bk_OCBS` ほか [viem_results/create_h5.jl:17-28](viem_results/create_h5.jl#L17-L28)）
- `Manifest.toml` は Julia **1.12.5** で固定（`juliaup override set -p <project> 1.12.5` 済み）。`Pkg.update()` / `Pkg.resolve()` を不用意に走らせない

## 8. Julia コーディング・パフォーマンスのガイドライン

既存モジュールを変更する場合は以下を維持する:

- **型安定性**: `@code_warntype` で Any 推論が出ないこと
- **メモリ割り当て最小化**: ループ内の再割り当てを避け、`.=`, `@views`, `mul!` で in-place 化
- **マルチスレッド**: 近傍場行列や Duffy 変換のような独立ループは `Threads.@threads` を使う。本番実行は `julia --project=. -t auto`
- **二言語化を避ける**: 計算コアは Julia で完結。Python との連携は HDF5 経由に限定

## 9. 参考資料（実装済み・Q&A 用）

以下はすべて実装に反映済みのため、**再実装の拠り所ではなく「なぜこの数式か」「なぜこの定数か」の参照用**:

- `Integral-Equation-Methods-for-Electromagnetics.pdf` 第 6 章（VIEM 定式化、特異点積分）— `.claude/reference/`
- `Essentials of Computational Electromagnetics.pdf`（AIM 定式化、グリッド射影）— `.claude/reference/`
- `s00466-009-0424-1.pdf` (Mousavi & Sukumar 2010、一般化 Duffy 変換)— `.claude/reference/`
- [docs/theory_note.tex](docs/theory_note.tex) / `docs/theory_note.pdf`（本ソルバの理論ノート、セクション番号で参照）
- [README.md](README.md)（実装方針、ベンチマーク結果、block-DDA_Py との定義対応）
- `block-DDA_Py` — `~/Python_in_WSL/block-DDA_Py`（出力フォーマット互換の参照元）
