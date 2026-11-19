# 物理光学・計算電磁気学ソルバー (VIEM-AIM in Julia) — 論文計算フェーズ運用ガイド

## 1. プロジェクト現状 (post-implementation)

本リポジトリはソルバの実装・検証を完了している。**新規モジュールを設計する段階ではない**；論文掲載用の本番計算を `src/` の既存実装で実行するフェーズ。

- 検証状況 (2026-11-19, v0.8.2): 14,780 / 14,780 テスト全パス on Julia 1.12.5（v0.8.1 の 14,749 件 + 集合体メッシュ 31 件）。
- 検証状況 (2026-11-19, v0.8.1): 14,749 / 14,749 テスト全パス（v0.8.0 の 14,720 件 + 異方性 AIM 29 件）。
- 検証状況 (2026-11-19, v0.8.0): 14,720 / 14,720 テスト全パス（v0.7.9 までの 14,439 件 + shift スイープ 238 件 + `S_bk` 43 件）。
- 検証状況 (2026-04-21): 14,422 / 14,422 テスト全パス on Julia 1.12.5。Mie 球面 (低/高屈折率・Au)、CAS-v2 2 球 doublet (vs MSTM)、AIM vs 直接法、block-DDA_Py 出力互換の各検証で回帰なし。
- 実装モジュール (src/):
  - メッシュ・基底: [gmsh_io.jl](src/gmsh_io.jl), [mesh.jl](src/mesh.jl), [swg.jl](src/swg.jl), [gre_mesh.jl](src/gre_mesh.jl), [aggregate_mesh.jl](src/aggregate_mesh.jl), [spheroid_sweep_io.jl](src/spheroid_sweep_io.jl)
  - 特異積分: [quadrature.jl](src/quadrature.jl), [duffy.jl](src/duffy.jl), [triangle_quad.jl](src/triangle_quad.jl), [surface_integrals.jl](src/surface_integrals.jl)
  - Green 関数・インピーダンス: [green.jl](src/green.jl), [impedance.jl](src/impedance.jl)
  - AIM: [aim_grid.jl](src/aim_grid.jl), [aim_projection.jl](src/aim_projection.jl), [aim_toeplitz.jl](src/aim_toeplitz.jl), [aim_operator.jl](src/aim_operator.jl)
  - ソルバ: [solver.jl](src/solver.jl), [block_krylov.jl](src/block_krylov.jl)
  - 入射・後処理: [incident.jl](src/incident.jl), [postprocess.jl](src/postprocess.jl)
  - 屈折率スイープ (v0.8.0): [shifted_sweep.jl](src/shifted_sweep.jl)（`block_gmres_shifted` は block_krylov.jl、`aim_K0_mvp` は aim_operator.jl）
- 本番計算のエントリポイント:
  - GRE / spheroid / single sphere スイープ: [viem_results/create_h5.jl](viem_results/create_h5.jl), [viem_results/run_viem.jl](viem_results/run_viem.jl), [viem_results/check_h5.jl](viem_results/check_h5.jl)
  - 2 球 doublet の本番: 同じ [viem_results/run_viem.jl](viem_results/run_viem.jl)（`shape_kind = "doublet"`、テンプレートは `viem_results/paper/doublet_*.jl`）。[benchmarks/cas_v2/doublet_mstm/run_viem.jl](benchmarks/cas_v2/doublet_mstm/run_viem.jl) は README の固定条件ベンチマーク（R = 0.030 μm）用

## 2. 論文の計算マトリクス

波長固定 **λ₀ = 0.638 μm**、真空背景 `n_m = 1.0`。

| 形状 | 形状パラメータ | 配向プロトコル |
| --- | --- | --- |
| 単球 | 球 | 単配向 + 球面一様多配向（RHS 数スイープ 1, 2, 4, 8, 16, 32, 64, 128） |
| 2 球クラスター | 等半径 doublet, **touching gap = 0.1 R** along doublet axis (README と同条件) | 単配向 + 球面一様多配向 + 代表配向 β = 0°, 30°, 60°, 90° |
| 扁平回転楕円体 (oblate) | b/c = 3, a/b = 1 | 単配向 + 球面一様多配向 + 代表配向 |
| GRE | a/b = b/c = 1, β_gre = 0.2 | 単配向 + 球面一様多配向 |

| 物質 | n_p @ λ=0.638 μm | a_eq (μm) |
| --- | --- | --- |
| n15 (低屈折率) | 1.5 + 0.01i | 0.05, 0.1, 0.2, 0.4 |
| n20 (paper "high", v0.7.5〜) | **2.0 + 0.0i** | 0.05, 0.1, 0.2, 0.4 |
| n317 (legacy high, ≤ v0.7.4) | 3.17 + 0.16i | (新規計算には使わない。`*_n317.hdf5` はこのリポジトリの作業ツリーには無い、2026-11-19 確認) |
| Au (J&C 1972) | **0.17525 + 3.4830i** (λ=0.638 μm でハードコード) | 0.05, 0.1, 0.2 (**0.5 μm は廃止** — Au の `\|m_p\|≈3.49` で波長制約 lc≈0.018 μm が支配し N_DOF 爆発、1 スロットで 24 h 超え確実のため) |

**v0.7.5 での材料差替え**: paper "high" を n317 (|m_p|≈3.17) → n20 (|m_p|=2.0) に引き下げ。実測 (DDA) で `gre × n317 × r_v=0.4 × L=100` が peak RSS 215 GB / wall 195 min を消費し、block-Krylov stagnation (iter=100, err~3e-3) も頻発したため。n20 は非吸収・中屈折率で両ソルバとも安定収束の見込み。定数 `N_HIGH` と `run_lc_convergence.jl` の `n317` 引数は残してある。

**v0.7.6 での MAXITER 拡張**: 全 runner (`run_viem.jl`, `run_lc_convergence.jl`, `run_rhs_scaling.jl`) と `MAXITER_HISTORY` を 100 → 200 に bump。理由: n20 化で典型 slot は ~30-50 iter で収束見込みなので 200 cap は安全マージン、Au plasmonic と RHS-scaling L=64-128 の stagnation 領域に追加余地を与えるため。v0.7.4 の `O(kL²)` block-Givens fix で per-iter コスト線形のため maxiter 倍化のオーバヘッドは限定的。

**v0.7.7 での volume-preserving rescale**: `gre_mesh`, `gre_mesh_with_field`, `mesh_sphere_aggregate` がメッシュ生成後に `apply_scale!(mesh, r_v_base / r_ve)` で節点座標を等方スケールし `V_mesh = (4π/3)·r_v_base³` を**厳密**に満足させるよう変更。block-DDA_Py 側 `Target.__init__` の `d_adj` 再スケールと対称化し、Rayleigh 領域の C_ext/C_abs/|S_fw| に出る discretisation volume bias (~0.5-3%) を除去。`r_ve` が HDF5 上で `r_v_base` と機械精度で一致するため paper の 1:1 相関プロットが clean に。`mesh_sphere_aggregate` のみ `rescale_to_target_volume::Bool=true` キーワードで無効化可能 (overlap geometry test 用)。詳細は [docs/paper_simulation_conditions_viem.md §5.2](docs/paper_simulation_conditions_viem.md) 参照。

観測量:

- **Qext, Qsca, Qabs** (体積等価半径基準、規格化 `Q = C / (π a_eq²)`)
- **CAS-v2 複素散乱振幅**: `S_fw_θ`, `S_fw_φ`, `S_bk` (μm 単位、block-DDA_Py 互換定義)

### 形状パラメータの換算

- **2 球 doublet の半径換算**: 体積等価半径 `a_eq` から monomer 半径 `R = a_eq / 2^(1/3) ≈ 0.7937 · a_eq`、軸方向 gap `g = 0.1 · R` (README Benchmark 節と同比)。a_eq = 0.05, 0.1, 0.2, 0.4 μm → R ≈ 0.0397, 0.0794, 0.1587, 0.3175 μm。メッシュは `mesh_sphere_aggregate(...; overlap_factor = 0)` で作り、中心間距離は `2.1 R`（§12 参照）。
- **扁平回転楕円体**: 3 半軸 `(a, b, c)`, `a = b`, `b/c = 3` → `c = (a_eq³ / (b/c)²)^(1/3) = a_eq / 9^(1/3)`, `a = b = 3c`。
- **GRE**: `GREParams(a_eq, 1.0, 1.0, 0.2)` でそのまま構築（位置引数の順は `r_v_base, bc_ratio, ab_ratio, beta`。キーワード形式は無い）。

## 3. メッシュ指定プロトコル

[src/gre_mesh.jl](src/gre_mesh.jl) の `adaptive_lc` と [src/aggregate_mesh.jl](src/aggregate_mesh.jl) の `adaptive_lc_aggregate` を使う。公式は

```text
lc = min( λ₀ / (|m_p|_max · N_pw),    # 波長制約, N_pw = 10 (既定)
          c_min / N_per_radius,        # 幾何制約, N_per_radius = 3 (既定、c_min = 最小半軸)
          0.3 · c_min / 3 )            # GRE β > 0 時の相関長制約 (c_min 基準)
```

- `|m_p|` は anisotropy 対応のモジュラス最大（Au は `|0.17525 + 3.4830i| ≈ 3.488` → lc ≈ 0.018 μm、保守側）
- 本番スイープは既定 `N_pw = 10`, `N_per_radius = 3`（GRE の `adaptive_lc` では 3 は固定値で、キーワードは `adaptive_lc_aggregate` / `mesh_sphere_aggregate` だけにある）
- これらは `gre_mesh(...; wl_0, m_p_max, N_pw)` / `mesh_sphere_aggregate(...; wl_0, m_p_max)` の内部で自動呼び出し

## 4. lc 収束性の評価（論文 Figure 用）

各 (物質 × 形状) の組み合わせ（形状は sphere / oblate / gre。doublet は対象外）について、**a_eq = 0.1 μm を代表サイズ**として lc を 5 段階スイープして Qext, Qsca, Qabs, |S_fw_θ| の収束性を測る:

```text
factor = 1.5, 1.0, 0.7, 0.5, 0.35   # adaptive_lc を 1.0 とする倍率
lc = factor × adaptive_lc(...)
```

- 本番大規模スイープの lc は **factor = 1.0**（`adaptive_lc` そのまま。`run_viem.jl` は `lc` を渡さない）
- 結果は `viem_results/paper/convergence_{shape}_{material}.hdf5` に保存。本番ファイルとは別のレイアウトで、`/target/lc_convergence/`（`lc_factor`, `lc`, `n_tet`, `n_dof`, 観測量、cost）と `/target/reference/`（Mie）を持つ
- 論文 Figure には VIEM の O(h²) と DDA の O(h^0.5) の対比（[docs/benchmark_results.md §9.2](docs/benchmark_results.md) の実績）を掲載

## 5. 実行前チェックリスト（一計算 < 24 h を強制）

HDF5 スイープを起動する前に必ず:

1. **N_tet / N_DOF の見積**: GRE/sphere/spheroid は `adaptive_lc` と形状から、doublet は `aggregate_mesh` の実メッシュ生成で確認
2. **メモリ見積**: [docs/benchmark_results.md §8.3](docs/benchmark_results.md) の phase-A 実測から `RSS ≈ 100 kB/DOF` 程度を目安とし、`N_DOF × N_RHS × 2 (complex)` を block-Krylov の Krylov 空間分として加算
3. **Wall-time 見積**: setup 時間は N_DOF に対し実質線形、per-iteration cost は `N_DOF log N_DOF × N_RHS`。block GMRES の反復数の目安は過去実績（誘電体・酸化鉄で 20–90 回、Au の N_RHS=1 で 200 回超、N_RHS=64 で約 100 回）
4. **推定 > 24 h または RSS > マシン物理メモリ → 必ずユーザーに確認**

見積ツール: [viem_results/estimate_cost.jl](viem_results/estimate_cost.jl)。引数はスイープ用 HDF5 のパスで、各スロットの (N_DOF, RSS_est, t_setup_est, t_solve_est) とスイープ全体の合計を表示し、24 h 超または空きメモリの 90 % 超のスロットに印を付ける。

## 6. HDF5 スイープ運用規約

- 1 HDF5 = 1 物質 × 1 形状（Au, 低屈折率, 高屈折率それぞれ別ファイル、doublet も別）
- ファイル命名: `viem_results/paper/{shape}_{material}.hdf5`
  - `sphere_n15.hdf5`, `sphere_n20.hdf5`, `sphere_Au.hdf5`
  - `doublet_n15.hdf5`, `doublet_n20.hdf5`, `doublet_Au.hdf5`
  - `oblate_n15.hdf5`, `oblate_n20.hdf5`, `oblate_Au.hdf5`
  - `gre_n15.hdf5`, `gre_n20.hdf5`, `gre_Au.hdf5`
  - lc 収束: `convergence_{shape}_{material}.hdf5`
- 既存 HDF5 の計算済みスロット（`S_fw_PCAS_mie` の虚部が 0 でないもの）は `run_viem.jl` がスキップするので、中断再開できる
- **`/target/cost/` グループで end-to-end cost を per-slot 記録（block-DDA_Py 側と対称、v0.7.3+）**。[viem_results/paper/_common.jl](viem_results/paper/_common.jl) の `create_paper_h5` が 12 dataset（`t_build_s`, `t_setup_s`, `t_solve_s`, `t_total_s`, `peak_rss_bytes`, `n_tet`, `n_dof`, `mean_edge_length`, `iters`, `converged`, `solver_err`、および v0.7.5 からの `residual_history`）を shape_cond 次元で作成し、[run_viem.jl](viem_results/run_viem.jl) が per-slot で埋める。peak RSS は [viem_results/rss_monitor.jl](viem_results/rss_monitor.jl) の daemon-task サンプラ（`/proc/self/status:VmRSS` を 0.2s 間隔でポーリング、`reset!` で per-slot ベースラインを切り替え）経由で取得。DDA 側 ([~/Python/block-DDA_Py/utils/rss_monitor.py](file:///home/moteki/Python/block-DDA_Py/utils/rss_monitor.py)) と `/target/cost/` の 8 フィールド（時間・peak RSS・iters・converged・solver_err）は bit-for-bit 一致、残り 3 つは VIEM 側名称 (`n_tet`/`n_dof`/`mean_edge_length`) で DDA の (`n_cuboid`/`n_occ`/`lattice_lf`) と 1:1 対応。lc 収束 HDF5 の `lc_convergence/` サブグループにも同じ 8 フィールドを書き込む。RHS-scaling の `/target/rhs_scaling/gmres/` は `iters`, `converged`, `t_total_s`, `t_end2end_per_orient_s`, `peak_rss_bytes`, `residual_history` だけを持つ
- 後方互換: `/target/cost/` を持たない古い HDF5 は `run_viem.jl` 起動時にエラーで拒否。`create_paper_h5` または `create_h5.jl`（v0.8.0 から cost group を書く）で再生成する
- 多配向 block-Krylov の **反復数 と 1 配向あたり end-to-end 所要時間** は別スクリプト [viem_results/paper/run_rhs_scaling.jl](viem_results/paper/run_rhs_scaling.jl) で診断:
  - RHS 数スイープ `L = 1, 2, 4, 8, 16, 32, 64, 128`（2 のべき乗、計 8 点で iter 数のスケーリングを解像）
  - **v0.7.6 から GMRES 単独計測**: `block-GMRES` (`:aim_gmres`) のみ。論文のスコープからソルバ依存性軸を外し、shape × material × N_DOF のスケーリングに焦点を絞るため。BiCGSTAB を再度比較したい場合は `run_rhs_scaling.jl::METHODS` に `(:bicgstab, :aim_bicgstab)` を追記すれば schema 変更なしで復活可
  - 結果は `/target/rhs_scaling/gmres/` サブグループに `iters`, `converged`, `t_total_s`, `t_end2end_per_orient_s`, `peak_rss_bytes`, `residual_history` を書き込み（`L_values`, `n_dof`, `n_tet` は親グループに）
  - 各 shape slot ごとに測定（worst-case mesh 共有、固定 RNG seed の uniform-sphere 配向を nested に L=1 ⊂ L=2 ⊂ … ⊂ L=128）
- **本番ソルバのデフォルトは v0.7.1 から `:aim_gmres`**（`solve_cas_v2_orientations` の method キーワードおよび [viem_results/run_viem.jl](viem_results/run_viem.jl) の `SOLVER_METHOD`）。理由: GRE 本番で block solve L=100、N_DOF 数十万の条件下では BiCGSTAB の stagnation リスクが高く、論文全 44 スロットの完走を優先して monotone 収束保証のある GMRES を選択。パフォーマンス重視のケースでは明示的に `:aim_bicgstab` を指定する
- **doublet 厳密解（MSTM 参照）は別 HDF5 で管理**: [viem_results/paper/run_mstm_reference.jl](viem_results/paper/run_mstm_reference.jl) を MSTMforCAS.jl env (`~/Julia/MSTMforCAS.jl`) で実行し、対応する `mstm_<basename>.hdf5` を生成
  - `truncation_order = 15`（[README.md](README.md) の doublet ベンチマーク節に準拠、Au 含む両材料で完全収束）
  - スキーマは `/target/{a_eq_um, R_monomer_um, gap_um, beta_rad, observables/{Q_*, S_fw_*, S_bk}, diagnostics/{n_iterations, converged}}`、軸対称性により (a_eq, β) の 2D グリッドのみ（α/γ は VIEM の解析展開と物理的に一致するため不要）
  - 体積等価球の Mie 値は run_viem.jl の `compute_mie_reference` が自動的に書き込む（doublet では Rayleigh サニティチェック用途、厳密解ではない）

## 7. 不変条件（回帰ガード）

以下を壊す変更は行わない:

- [test/test_mie_validation.jl](test/test_mie_validation.jl), [test/test_cas_v2.jl](test/test_cas_v2.jl), [test/test_block_krylov.jl](test/test_block_krylov.jl), [test/test_aggregate_mesh.jl](test/test_aggregate_mesh.jl) は常時グリーン
- block-DDA_Py 互換 HDF5 スキーマ（`C_ext`, `C_abs`, `C_sca` (= C_ext − C_abs), `S_fw_PCAS_theta`, `S_fw_PCAS_phi`, `S_bk_OCBS` ほか。一覧は [viem_results/create_h5.jl](viem_results/create_h5.jl) 冒頭のコメント）
- `Manifest.toml` は Julia **1.12.5** で固定（`juliaup override set -p <project> 1.12.5` 済み）。`Pkg.update()` / `Pkg.resolve()` を不用意に走らせない

## 8. Julia コーディング・パフォーマンスのガイドライン

既存モジュールを変更する場合は以下を維持する:

- **型安定性**: `@code_warntype` で Any 推論が出ないこと
- **メモリ割り当て最小化**: ループ内の再割り当てを避け、`.=`, `@views`, `mul!` で in-place 化
- **マルチスレッド**: 近傍場行列や Duffy 変換のような独立ループは `Threads.@threads` を使う。本番実行は `julia --project=. -t auto`
- **二言語化を避ける**: 計算コアは Julia で完結。Python との連携は HDF5 経由に限定

## 9. 参考資料（実装済み・Q&A 用）

以下はすべて実装に反映済みのため、**再実装の拠り所ではなく「なぜこの数式か」「なぜこの定数か」の参照用**:

- Volakis & Sertel, *Integral Equation Methods for Electromagnetics* 第 6 章（VIEM 定式化、特異点積分）— `.claude/reference/`（ファイル名は `Integral-Equation-Methods-for-Electromagnetics John-L.-Volakis-Kubilay-Sertel-.pdf`）
- Sheng & Song, *Essentials of Computational Electromagnetics*（AIM 定式化、グリッド射影）— `.claude/reference/`（ファイル名は `Essentials of Computational Electromagnetics-Sheng and Song.pdf`）
- `s00466-009-0424-1.pdf` (Mousavi & Sukumar 2010、一般化 Duffy 変換)— `.claude/reference/`
- [docs/theory_note.tex](docs/theory_note.tex) / `docs/theory_note.pdf`（本ソルバの理論ノート、セクション番号で参照）
- [README.md](README.md)（実装方針、ベンチマーク結果、block-DDA_Py との定義対応）
- `block-DDA_Py` — `~/Python/block-DDA_Py`（出力フォーマット互換の参照元、姉妹プロジェクト [CLAUDE.md](file:///home/moteki/Python/block-DDA_Py/CLAUDE.md)）

## 10. 屈折率スイープ（shifted block GMRES、v0.8.0）

等方・均質粒子では `Z(m_p) = (κ/ε_bg)(τM − K)`、`τ = 1/((m_p/m_m)² − 1)`。`M⁻¹` 右前処理により全 m_p が `K M⁻¹` の shift になり、1 回のブロック Arnoldi で解ける。

- エントリポイント: `solve_cas_v2_orientations_shifted(basis, eulers; wl_0, m_m, m_p_list, ...)`
- ドライバ: `benchmarks/cas_v2/{ironoxide,spheroid}_sweep_h5.jl --shifted-ri`、[viem_results/run_viem.jl](viem_results/run_viem.jl) `--shifted-ri`（block-DDA_Py の `run_dda.py --shifted-ri` に対応）
- 既定動作は不変。`--shifted-ri` なしの出力は変更前と一致する（`benchmarks/shifted/results/` の `p3_driver_check.txt`, `run_viem_shifted_check.txt`）
- 収束判定は現行と同じ `‖B − Z D‖_F/‖B‖_F < tol`。メッシュは屈折率軸で共通（リスト内の最大 |m_p| で作る）。既定モードでも共通メッシュなので、shift モードの値は既定モードとソルバ許容誤差の範囲で一致する
- 非対応: 異方性（`--form-biref`, `--biaxial`、`m_p_xyz` の 3 成分が異なる行）、非均質粒子、`:aim_bicgstab`。`run_viem.jl` は異方性の行を従来どおり 1 つずつ解く
- バッチはコントラストの近い m_p でまとめる（反復数は最難の m_p で決まる）。Au と誘電体の混在は未測定
- `run_viem.jl --shifted-ri` の `/target/cost/`: 同じバッチで解いた行の `t_solve_s`, `t_total_s` はバッチ時間を行数で割った値、`peak_rss_bytes` はバッチのピーク。`iters`, `converged`, `solver_err`, `residual_history` は屈折率ごと。モードは `/target` の属性 `shifted_ri` に記録し、1 ファイルに 2 つのモードを混在させない
- 通常求解の `precond = :mass_right`（`M⁻¹` 右前処理）はオプション。既定は `:none`
- 回帰ガード: [test/test_shifted_krylov.jl](test/test_shifted_krylov.jl), [test/test_shifted_sweep.jl](test/test_shifted_sweep.jl) は常時グリーン
- ベンチマーク: [benchmarks/shifted/run_shift_scaling.jl](benchmarks/shifted/run_shift_scaling.jl)。結果は docs/benchmark_results.md §11、理論は docs/theory_note.pdf 第 11 節。実測の加速率は N_m = 3 / 10 / 30 で約 3 / 10 / 29–30 倍

## 11. 後方散乱振幅 `S_bk` の定義（v0.8.0 で訂正）

- `S_bk`（HDF5 の `S_bk_OCBS`）は `(−S_bk_θ + S_bk_φ)/2 = (−S11 + S22)(180°)/2`。Moteki and Adachi (2024) Eq. (28) の erratum の修正式で、block-DDA_Py v0.8.2 以降と同じ
- v0.7.9 以前は印刷版の `1/√2` を使っており、`S_bk` は現行の √2 倍だった（位相、VIEM/参照値の比、前方散乱振幅、断面積は同じ）
- `viem_results/paper/` の HDF5 はすべて修正式に揃えてある（2026-11-19、ユーザー指示）。`doublet_*.hdf5` は v0.8.2 で作り直し。それ以外（sphere / oblate / gre / pilot の VIEM と Mie、MSTM の `mstm_doublet_*.hdf5`、T-matrix の `tmm_oblate*.hdf5`、`convergence_*.hdf5`）は保存値に 1/√2 を掛けて変換した。再計算はしておらず、後方散乱振幅以外のデータセットは元のまま。属性 `S_definition`（無いファイルは `S_bk_definition`）と `S_bk_conversion_note` で確認できる。v0.7.8 が作った元のファイルは git 履歴の `416cece`外部にある v0.7.9 以前のファイルと比べるときは、その `S_bk` に 1/√2 を掛ける
- 新しく作る HDF5 の `S_definition` 属性は `.../2`。`run_viem.jl` は旧定義の結果が入ったファイルへの追記を拒否する
- 回帰ガード: [test/test_backward_amplitude.jl](test/test_backward_amplitude.jl) は常時グリーン

## 12. v0.8.1 / v0.8.2 の修正（2026-11-19）

- **doublet の形状**: 中心間距離 `2.1 R`、gap `0.1 R`（文書と MSTM 参照どおり）
  - v0.8.2: `mesh_sphere_aggregate` は、離れた部分からなる集合体を部分ごとに体積合わせする（`_rescale_to_monomer_volumes!`）。各球を自分の中心のまわりで公称体積に合わせるので、中心間距離はメッシュの粗さにも `overlap_factor` にも依らない。つながった集合体は従来どおり原点まわりの一様スケール
  - 経緯: v0.8.0 までの `run_viem.jl` は既定の `overlap_factor = 0.02` と全体の一様スケールで、中心間距離が `2.06–2.08 R`（表面間の gap `0.0595 R`）だった。v0.8.1 は `overlap_factor = 0` にしたが一様スケールのままで `2.10–2.13 R`。v0.8.2 で `2.100 R`
  - `viem_results/paper/doublet_*.hdf5` は v0.8.2 で作り直した（2026-11-19、supermicro-647）。`S_bk` は修正式、`/target` に属性 `doublet_gap_over_R = 0.1`。v0.7.8 の本番ファイルは git 履歴（`416cece`）にある。cost の値は他の論文用ファイルと計算機が違う
  - MSTM との差（β 5 点の最大）: Au の `C_ext` が 13 % / 5 % / 5 % → 1.5 % / 2.0 % / 1.5 %、n15 の `S_bk`（a_eq 0.1, 0.2 μm）が 4.7 % / 16 % → 0.3 % / 4.7 %。誘電体の小粒径の `S_fw` は 0.2–0.9 % → 0.4–1.7 %（旧メッシュの中心間距離が粗いメッシュでたまたま 2.1 R に近かった）
  - Au の doublet は 3 サイズとも 200 反復で未収束（残差 8e-5〜5.7e-4）。v0.7.8 の本番でも同じ
  - `run_viem.jl` は、属性 `doublet_gap_over_R` が無く結果が入っている doublet ファイルへの追記を拒否する
  - README の doublet ベンチマーク（`benchmarks/cas_v2/doublet_mstm/`、docs/benchmark_results.md §7–§8）の数値は v0.8.1 以前のメッシャーのもの。未再実行
- **異方性の AIM 経路**: 質量項は `Σ_α (1/ε_α) M_α`（`AIMOperator.mass_weighted`、`assemble_weighted_mass_matrix`）。v0.8.0 までの AIM 経路は `1/ε_α` の平均を使っており、異方性粒子の観測量が 5–25 % ずれていた。`:dense` は以前から正しい。等方の結果は変わらない
  - v0.8.0 以前に `:aim_gmres` で作った異方性の結果（`ironoxide_sweep_h5.jl --form-biref` / `--biaxial` の表、`form_birefringence_probe.jl`、`biaxial_probe.jl`）は作り直しが要る
  - 回帰ガード: [test/test_anisotropic.jl](test/test_anisotropic.jl) の AIM 対 密行列の比較
- **`run_viem.jl` の失敗時処理**: 既定モードでソルバが例外を出したとき、NaN を書いた後に止まっていた（番兵に `residual_history` が無かった）。修正済みで、NaN を埋めて次の条件へ進む

