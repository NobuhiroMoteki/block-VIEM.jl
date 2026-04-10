
# 物理光学・計算電磁気学ソルバー開発プロジェクト (VIEM-AIM in Julia)

## プロジェクト概要
本プロジェクトは、任意形状・高屈折率の非球形粒子（酸化鉄の癒着凝集体や表面凹凸のある金ナノ粒子など）による光散乱問題を解くための、Julia言語による体積積分方程式法（VIEM）ソルバーの新規開発である。
既存の `block-DDA_Py` (離散双極子近似) の代替となる高精度・高効率なソルバーを目指す。

## コア仕様（実装要件）
1. **離散化**: Gmshを用いた四面体要素（Tetrahedral mesh）。ベクトル基底関数としてSWG（Schaubert-Wilton-Glisson）基底を採用する。
2. **特異点積分**: ガウスの定理で特異性を 1/R に緩和（部分積分）したのち、自己項（Self-term）および近接項（Adjacent-term）の数値積分には **Duffy変換（Duffy's transformation）** を適用し、特異性をキャンセルしてガウス求積を行う。
3. **高速化（MVP-FFT）**: AIM（Adaptive Integral Method）を実装し、四面体上の分極を直交グリッドに射影（Taylor展開等によるモーメントマッチング）する。遠方場はFFTを用いたToeplitz行列の行列ベクトル積（MVP）で計算し、近傍場は直接積分で補正（Precorrection）する。
4. **反復ソルバーと多配向一括計算**: Krylov部分空間法（BiCGSTAB等）を使用する。最終的には `block-DDA_Py` と同様に、多数の入射角・偏光状態（複数右辺ベクトル）を一括して効率的に解くため、Block-KrylovソルバーまたはT行列（遷移行列）的なアプローチを実装する。
5. **出力物理量**: 消散・散乱・吸収断面積、複素散乱振幅、とくにCAS-v2の観測量など。出力フォーマットや計算ロジックは `block-DDA_Py` を踏襲する。

## Juliaコーディング・パフォーマンスのガイドライン
* **型安定性（Type Stability）**: `@code_warntype` を意識し、Any型が推論されないようにする。
* **メモリ割り当ての最小化**: ループ内での配列の再割り当てを避け、In-place演算（`.=`, `@views`, `mul!` など）を徹底する。
* **マルチスレッド**: 近傍場行列の計算やDuffy変換のループなど、互いに独立した重い処理には `Threads.@threads` を積極的に用いる。
* **二言語問題の回避**: 計算のコア部分は全てJuliaで完結させる。

## 開発フェーズ（Step-by-Step）
Agentは以下の順序でモジュールを設計・実装し、各ステップで単体テスト（Unit Test）を書くこと。
1. `Mesh/Geometry`: Gmshファイルの読み込み (`Gmsh.jl`推奨) と四面体要素、SWG基底関数のデータ構造構築。
2. `Integration`: Duffy変換を用いた特異点・近接項の体積積分エンジンの実装とテスト（解析解との比較）。
3. `AIM`: 直交仮想グリッドの生成、射影行列（W）の構築、およびFFT-MVPの実装。
4. `Solver`: 近傍場補正行列の構築、Krylovソルバーとの結合（IterativeSolvers.jl や Krylov.jl を使用）。
5. `PostProcess`: 遠方界への放射積分と `block-DDA_Py` 互換の散乱特性の計算。

## 参考資料リスト（RAG用コンテキスト）
Agentは必要に応じて以下の資料を参照すること。これらはプロジェクトディレクトリ内の特定のフォルダ（例: `.claude/reference/`）に配置されている。
* **`Integral-Equation-Methods-for-Electromagnetics.pdf`**: 第6章のVIEM定式化と特異点積分の基本概念。`.claude/reference/`）に配置。
* **`Essentials of Computational Electromagnetics.pdf`**: AIMの定式化、グリッド射影の具体的な数式。`.claude/reference/`）に配置。
* **`s00466-009-0424-1.pdf`** ("Generalized Duffy transformation...", Mousavi & Sukumar, 2010): VIEMの近接項・自己項における 1/R 特異性を解消するための、四面体から立方体への座標変換とヤコビアンの式。`.claude/reference/`）に配置。
* **`block-DDA_Py`**: 既存の離散双極子法のPythonコード。入力パラメータ、出力パラメータ（CAS-v2の観測量など）の単位、計算ロジックや多配向の解の一括計算、FFT-MVPのアルゴリズムのリファレンスとして使用。プロジェクトフォルダの外の`~Python_in_WSL/block-DDA_Py`に配置。
