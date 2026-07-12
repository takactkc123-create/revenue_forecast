# 個人住民税予測モデル

## 概要

住民税は自治体の主要財源のひとつである。翌年度の税収をどれだけ正確に見通せるかは、予算編成の精度に直結する。

このプロジェクトは、**個人の収入・控除情報をもとに翌年の住民税額を予測する機械学習モデル**を構築するパイプラインである。LightGBM（勾配ブースティング）を用いて個人レベルの税額を推定し、それを集計することで自治体全体の税収見込みを算出する。

実データがなくても動かせるよう、日本全体の統計データをもとにしたダミーデータ生成機能も内蔵している。

---

## クイックスタート

```bash
# 実データがある場合（03から開始）
uv run python 03_feature_eng.py
uv run python 04_model_train.py --retrain-all   # walk-forward検証 + 全年再学習
uv run python 05_predict_2026.py
uv run python 06_trend_correction.py
uv run python 07_visualize.py

# ダミーデータで試す場合（01から開始）
uv run python 01_generate_dummy.py
uv run python 03_feature_eng.py
uv run python 04_model_train.py --retrain-all
uv run python 05_predict_2026.py
uv run python 06_trend_correction.py
uv run python 07_visualize.py
```

---

## ファイル構成

```
.
├── config.py                  # 全設定値の一元管理（ここだけ触ればパラメータ調整可能）
├── tax_reform.py              # 税制改正補正ロジック（共通モジュール）
│
├── 01_generate_dummy.py       # ダミーデータ生成
├── 02_datacheck.py            # データチェック
├── 03_feature_eng.py          # 特徴量エンジニアリング
├── 04_model_train.py          # モデル学習・精度検証
├── 05_predict_2026.py         # 翌年度予測
├── 06_trend_correction.py     # マクロ補正（トレンド・税制改正）
├── 07_visualize.py            # グラフ出力
│
├── data/
│   ├── individual_raw.csv              # 入力データ（実データ or ダミー）
│   ├── individual_prepared.csv         # 特徴量追加済みデータ
│   ├── tax_reform_config.csv           # 税制改正補正ルール
│   ├── yearly_result.csv               # 年度別合算精度
│   ├── val_result.csv                  # 個人別検証結果
│   ├── walkforward_result_04.csv       # walk-forward各フォールドの精度（--walkforward/--retrain-all時のみ）
│   ├── prediction_2026.csv             # 個人別予測値
│   └── prediction_adjusted_2026.csv    # 補正後予測値
│
├── models/
│   ├── lgbm_model.txt         # 学習済みモデル
│   └── model_config.csv       # 特徴量・パラメータ・検証モード記録
│
└── results/
    ├── fig1_yearly_accuracy.png          # 年度別合算精度グラフ
    ├── fig2_error_distribution.png       # 個人税額誤差分布
    ├── fig3_age_breakdown_2026.png       # 年齢区分別税額
    ├── fig4_summary_2026.png             # 予測サマリー
    ├── fig5_metrics_dashboard.png        # 評価指標テーブル＋年度別誤差率棒グラフ
    ├── fig6_tax_timeseries_2026.png      # 税収実績推移＋予測・信頼区間の時系列グラフ
    └── fig7_walkforward_report.png       # walk-forward検証レポート（--walkforward/--retrain-all時のみ）
```

---

## 実行順序と入出力

### パターン A：実データを使う場合

> 住民税システムや基幹系から抽出したCSVが手元にある場合

| ステップ | スクリプト | 入力 | 出力 |
|---:|---|---|---|
| 1 | `03_feature_eng.py` | `data/individual_raw.csv` | `data/individual_prepared.csv` |
| 2 | `04_model_train.py` | `individual_prepared.csv` | `lgbm_model.txt`, `yearly_result.csv` |
| 3 | `05_predict_2026.py` | `lgbm_model.txt`, `individual_prepared.csv` | `prediction_2026.csv` |
| 4 | `06_trend_correction.py` | `prediction_2026.csv`, `yearly_result.csv` | `prediction_adjusted_2026.csv` |
| 5 | `07_visualize.py` | 上記CSV群 | `results/*.png` |

実データCSVに必要な列は `03_feature_eng.py` 冒頭のドキュメントを参照。SQLでの抽出方法は `00_SQL/` 配下を参照。

### パターン B：ダミーデータで試す場合

> 実データがない・PoC段階・動作確認をしたい場合

| ステップ | スクリプト | 入力 | 出力 |
|---:|---|---|---|
| 0 | `01_generate_dummy.py` | `config.py` の設定値 | `data/individual_raw.csv` |
| 1〜5 | パターンAと同じ | — | — |

---

## 各ファイルの詳細

### `config.py` — 設定値の一元管理

すべてのパラメータはここに集約されている。コードを読まなくても、このファイルの数値を変えるだけで動作を調整できる。

| 設定グループ | 主な設定項目 | 変更タイミング |
|---|---|---|
| 年度設定 | `TRAIN_YEARS`, `TEST_YEAR`, `PREDICT_YEAR` | 年度更新時 |
| モデル | `LGBM_PARAMS` | 精度チューニング時 |
| 税控除推計 | `FURUSATO_PARAMS`, `HOUSING_PARAMS` | 実績データ取得後 |
| ダミーデータ規模 | `N_PER_YEAR`, `TURNOVER_RATE` | 計算コスト調整時 |
| 人口増減 | `POPULATION_GROWTH_RATES` | 年度別人口規模を変えたいとき |
| 男女比 | `GENDER_RATIO` | 対象自治体の実態に合わせるとき |
| 年齢構成 | `AGE_GROUPS`, `AGE_WEIGHTS` | 対象自治体の実態に合わせるとき |
| 賃金上昇率 | `APPLY_WAGE_GROWTH`, `SALARY_GROWTH_RATES`, `PENSION_GROWTH_RATES` | 年別の給与・年金上昇率を実績に合わせるとき |
| 検証モード | `VALIDATION_MODE`, `WF_MIN_TRAIN_YEARS` | CLIオプション未指定時のデフォルト設定 |

---

### `01_generate_dummy.py` — ダミーデータ生成

実データがない状態でもモデルの動作確認ができるよう、統計的に現実に近いダミーデータを生成する。

**生成ロジックの工夫**

- **給与収入**：国税庁「民間給与実態統計調査（令和6年分）第14図」の年齢5歳刻み・男女別の平均給与を参照して収入分布を設定。単一の平均値ではなく年代・性別ごとに分けることで実態に近い分布を再現している。
- **年金収入**：厚生労働省「厚生年金保険・国民年金事業の概況（令和5年度）」から年齢区分別の平均受給額（国民年金＋厚生年金の合計）を参照。60歳未満はゼロとし受給開始年齢を考慮している。
- **医療費控除の申告率**：厚生労働省の年齢階級別医療費支出割合（44歳以下17.9% / 45〜64歳22.0% / 65歳以上60.1%）をもとに確率を設定し、全体の申告率が約29%になるよう調整している。
- **年齢構成**：総務省統計局2024年データの20歳以上人口の年齢5歳刻み比率を使用。
- **男女比**：e-Stat（政府統計の総合窓口）より20歳以上の人口比率を参照し、女性51.7% / 男性48.3%に設定。
- **人口推移**：`POPULATION_GROWTH_RATES`による年別の人口倍率で各年のレコード数を調整。在籍者の一定割合（`TURNOVER_RATE`）が毎年入れ替わる流入・退出モデルを採用している。
- **賃金上昇率**：`APPLY_WAGE_GROWTH = True` のとき、給与（`SALARY_GROWTH_RATES`）と年金（`PENSION_GROWTH_RATES`）に年別の上昇率を個別適用する。在籍継続者には前年比の上昇率、新規流入者には基準年からの累積上昇率を適用する。`False` に設定すると全年一律 +0.5%/年の旧動作に戻る。

**留意点**

- 特定バイアスを避けるため、収入・控除の設定値はすべて全国統計から導出している（個別自治体の特性は反映していない）。
- 新規流入者の収入モデルは給与主体に簡略化しており、年金・事業・不動産等の複合保有者は既存在籍者に集中する設計である。

---

### `03_feature_eng.py` — 特徴量エンジニアリング

生データに含まれない「モデルへの入力として有効な列」を追加する。

**主な追加特徴量**

| 特徴量 | 内容 |
|---|---|
| `income_total` | 全所得種別の合計 |
| `has_salary` 等 | 所得種別フラグ（給与・事業・年金・不動産・配当） |
| `deduct_total` | 所得控除合計 |
| `deduct_rate` | 控除合計 ÷ 合計所得（税負担の軽さを表す比率） |
| `age_num` | 年齢5歳刻み区分の数値（1〜13） |
| `is_continuing` | 前年データあり（継続者）フラグ |
| `prev_tax_amount` | 前年税額（時系列での変化をモデルに伝える） |
| `income_yoy_change` | 合計所得の前年差額 |

**留意点**

- `muni_kintowari`（均等割）・`pref_tokuwari`（所得割）など税額内訳列は、合計すると目的変数と一致するため特徴量から除外している（リーク防止）。

---

### `04_model_train.py` — モデル学習・精度検証

**モデル選定：LightGBM**

住民税の計算は給与や控除の組み合わせによる非線形な計算体系であり、木ベースのアンサンブルモデルが適合しやすい。LightGBMは大量の数値列に対して高速かつ高精度で、外れ値（高額所得者）への耐性もある。

**検証モード（CLIオプションで切り替え）**

| オプション | 動作 | 使いどころ |
|---|---|---|
| `--standard`（デフォルト） | `TRAIN_YEARS` で学習 → `TEST_YEAR` で評価（1回のホールドアウト） | 動作確認・手早い精度確認 |
| `--walkforward` | fold1〜fold4 の時系列クロスバリデーション | モデルの安定性確認・バイアス検出 |
| `--retrain-all` | walk-forward 検証 → `TRAIN_YEARS + TEST_YEAR` 全年で再学習 | **実データ運用時の推奨フロー** |

`--retrain-all` の場合、walk-forward の最終フォールド（fold4）のアウトオブサンプル予測を `val_result.csv` として使用するため、評価の公平性は保たれる。最終モデルは全年データを学習済みのため、直近年（2025年）のパターンも反映した状態で2026年を予測できる。

各フォールドの精度は `data/walkforward_result_04.csv` に保存される。

**評価指標**

| 指標 | 意味 | 役割 |
|---|---|---|
| MAE | 個人1人あたりの予測誤差（円） | 誤差の絶対規模の把握 |
| WMAPE | 集計レベルの誤差率（非課税者含む） | **税収予測の主指標** |
| MAPE | 課税者のみの誤差率 | 参考指標 |
| R² | 予測の当てはまり度（1.0が完全一致） | 参考指標 |

> **WMAPEを主指標とする理由**：非課税者（税額=0円）を含む全体で集計したときの誤差率であり、自治体が管理する「税収合計」の予測精度と等価である。

**留意点**

- 税制改正がある年のラベルは `tax_reform_config.csv` の `label_correction` で補正してから学習する（改正の影響を過去年に誤帰属させない）。
- 非課税基準（地方税法第295条）以下の予測値は強制的に0円に上書きされる。
- `year` は特徴量（FEATURE_COLS）に含まれない。木モデルは訓練範囲外の年値を外挿できないためであり、年ごとの経済動向はダミーデータの上昇率設定や実データの特徴量分布として取り込む設計になっている。

---

### `05_predict_2026.py` — 翌年度予測

直近年のデータをベースに給与・所得トレンドを外挿して2026年の特徴量を生成し、学習済みモデルで予測する。

**主なオプション**

```bash
python 05_predict_2026.py --year 2027           # 予測年の変更
python 05_predict_2026.py --file data/xxx.csv   # 実データCSVを直接指定
python 05_predict_2026.py --wage-rate 0.025     # 給与上昇率を直接指定
```

**留意点**

- `--file` オプションで2026年の実データCSVを渡せば、外挿なしで実データベースの予測が可能である。
- コンフォーマル予測（信頼区間）を付与するため、予測値だけでなく上限・下限も出力される。
- 住宅ローン控除の期間満了（年間退出率: `HOUSING_PARAMS["annual_exit_rate"]`）を考慮した自動減衰処理がある。

---

### `06_trend_correction.py` — マクロ補正

個人レベルの予測では吸収しきれない集計レベルのズレを補正する。

**補正の種類**

| 補正 | 内容 |
|---|---|
| トレンド補正 | 年度別合算の系統的な過大・過小傾向を乗率で補正 |
| 税制改正マクロ補正 | 扶養控除要件引き上げ・特定親族特別控除等の影響を推計して加減算 |

補正ルールは `data/tax_reform_config.csv` で管理されており、新たな税制改正が生じた場合はCSVに行を追加するだけで対応可能（コード修正不要）。

---

### `07_visualize.py` — グラフ出力

| グラフ | ファイル名 | 内容 |
|---|---|---|
| Fig1 | `fig1_yearly_accuracy.png` | 年度別：実測 vs 予測 折れ線 |
| Fig2 | `fig2_error_distribution.png` | 個人税額誤差のヒストグラム |
| Fig3 | `fig3_age_breakdown_2026.png` | 年齢区分別の合計税額・人員（棒グラフ） |
| Fig4 | `fig4_summary_2026.png` | 予測中央値・低い見積もり・高い見積もり（95%信頼区間）＋数値テーブル |
| Fig5 | `fig5_metrics_dashboard.png` | WMAPE・RMSE 等の評価指標テーブル＋年度別誤差率棒グラフ |
| Fig6 | `fig6_tax_timeseries_2026.png` | 2020〜前年度の実績推移＋予測年度の予測値・95%信頼区間を重ねた時系列グラフ |
| Fig7 | `fig7_walkforward_report.png` | walk-forward各フォールドの精度テーブル＋誤差率棒グラフ（`--walkforward`/`--retrain-all`時のみ出力） |

---

### `tax_reform.py` — 税制改正補正モジュール

04〜06から呼び出される共通モジュール。以下の主要計算式を提供する。

- 給与所得控除（年次別のブラケット計算）
- 公的年金等控除（65歳未満・以上で異なる計算式）
- 基礎控除（所得水準による逓減）
- ふるさと納税の住民税控除推計
- 非課税判定（地方税法第295条）

**留意点**

- 住民税と所得税では控除額の上限・計算式が異なる（例：生命保険料控除の上限は住民税70,000円〔一般・介護医療・個人年金の3区分合計〕、所得税120,000円）。
- 定額減税（2024年）は `active: False` で無効化済み。実データを投入する際は2024年の税額を定額減税前の水準に加工してから使用すること。

---

## 設定値の変更ガイド

### 年度を更新したいとき

```python
# config.py
TRAIN_YEARS  = [2020, 2021, 2022, 2023, 2024, 2025]  # 訓練年を追加
TEST_YEAR    = 2026                                     # テスト年を更新
PREDICT_YEAR = 2027                                     # 予測年を更新
```

あわせて `POPULATION_GROWTH_RATES` にも新しい年のエントリを追加する。

### ダミーデータの規模を変えたいとき

```python
# config.py
N_PER_YEAR    = 100_000   # 1年あたり生成レコード数。増やすと精度が安定するが実行時間も増加
TURNOVER_RATE = 0.05      # 年間の入れ替わり率（5% = 毎年5%が転出・5%が転入）
```

### 対象自治体の実態に合わせたいとき

```python
# config.py
AGE_WEIGHTS = [...]            # 年齢構成比を自治体の住基データに合わせて変更
GENDER_RATIO = [0.517, 0.483]  # 女性・男性の比率（e-Stat から取得）
POPULATION_GROWTH_RATES = {    # 年別の対基準年人口倍率
    2020: 1.000,
    2021: 0.999,
    ...
}
```

### 賃金上昇率を実績値に合わせたいとき

```python
# config.py
APPLY_WAGE_GROWTH = True   # False にすると全年一律 +0.5%/年に戻る

SALARY_GROWTH_RATES = {    # 給与：厚労省「毎月勤労統計調査」等の前年比を参照
    2020: 1.000,
    2021: 1.018,
    2022: 1.021,
    2023: 1.036,
    2024: 1.051,
    2025: 1.057,
}

PENSION_GROWTH_RATES = {   # 年金：厚労省「年金改定率」を参照
    2020: 1.000,
    2021: 0.999,
    2022: 0.996,
    2023: 1.019,
    2024: 1.027,
    2025: 1.019,
}
```

### walk-forward 検証のデフォルトモードを変えたいとき

```python
# config.py（CLIオプション未指定時のデフォルト）
VALIDATION_MODE   = "retrain_all"   # "standard" / "walkforward" / "retrain_all"
WF_MIN_TRAIN_YEARS = 2              # fold1 の最低訓練年数
```

### 税制改正に対応したいとき

1. `data/tax_reform_config.csv` に行を追加（`reform_name`, `effective_year`, `param_key`, `param_value` 等）
2. 新しい計算式が必要な場合のみ `tax_reform.py` に関数を追加

---

## 環境

```bash
# 依存パッケージのインストール
uv sync
```

主要ライブラリ：`lightgbm`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`

Python 3.11以上を想定している。

---

## 注意事項

- このモデルはダミーデータをもとにした概念検証用途で構築されている。実際の税収予測に使用する際は、実データによる再学習と十分な検証を行うこと。
- 住民税の計算式は自治体・年度によって一部異なる場合がある（寒冷地加算等）。本実装は地方税法に基づく標準税率を前提としている。
- 個人情報を含む実データを扱う場合は、データの取扱いポリシーに従うこと。
