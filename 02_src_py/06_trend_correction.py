"""
06_trend_correction.py
個人住民税予測モデル - Step6: トレンド・税制改正マクロ補正

【補正の種類】
  A. トレンド補正（wage_trend_factor）
     - 年度別合算予測の系統的な過大・過小傾向を緩和する乗率補正
     - 05 の過去年検証（05out_forecast_backtest.csv。05 と同じ方法で過去年を予測した誤差）から自動算出（オプションで上書き可）

  B. 税制改正マクロ補正（tax_reform.py の REFORMS の macro_correction）
     - 扶養要件引き上げ（2026年〜）: 扶養控除新規取得者の増加分を推計して加算
     - 特定親族特別控除（2026年〜）: 19-22歳扶養親族への追加控除を推計して減算

【合計の信頼区間】
  05 の過去年検証の年ごとのぶれ（誤差率の標準偏差）から t 分布で作る（compute_yearly_ci）。
  将来の伸び率の当て外れなど外挿の誤差を含むため、個人別の区間（split conformal）の合計より広い。
  個人別CSVの個人ごとの区間は今までどおり（モデルの個人単位の誤差）なので、その合計とサマリーの区間は一致しない。
  --no-trend のときも幅は同じ方法で作るが、平均の偏りを打ち消さないので区間の中心はずれたままになる。

【import】
  data/05out_prediction_YYYY.csv      ← 05 の出力
  data/05out_forecast_backtest.csv    ← 05 の出力（過去年検証）
  data/03out_individual_prepared.csv  ← 03 の出力

【export】
  data/06out_prediction_adjusted_YYYY.csv          ← 補正後の個人別予測値
  data/06out_prediction_adjusted_summary_YYYY.csv  ← 補正後の合計サマリー

【使い方】
  ※ 02_src_py/ の中で実行する例。プロジェクトルートからは uv run python 02_src_py/06_trend_correction.py でも実行できる。
  uv run python 06_trend_correction.py
  uv run python 06_trend_correction.py --year 2026
  uv run python 06_trend_correction.py --no-trend   # トレンド補正をスキップ
  uv run python 06_trend_correction.py --factor 0.98  # トレンド補正乗率を直接指定

【人口減少（死亡・転出）を反映したいとき → --factor を使う】 2026-09-12追記
  05 は「予測年の対象者 ＝ 直近年の対象者そのまま」という前提で予測する
  （05 の estimate_next_year() が直近年のレコードをコピーして所得だけ伸ばすため、
   死亡・転出による減少も転入による増加も反映されない）。
  このため人口が減少している自治体では予測が過大になる。その分を打ち消すには
  --factor に「1 − 想定減少率」を渡す。

    uv run python 06_trend_correction.py --factor 0.988   # 対象者が年1.2%減る想定
    uv run python 06_trend_correction.py --factor 0.995   # 年0.5%減る想定

  乗率は個人別予測値と信頼区間の両方に一律で掛かる。
  根拠値は実データなら「前年にいたIDのうち翌年消えた割合 −  新規に現れたIDの割合」
  （＝消滅率と新規率の差引き）から算出できる。

  ※ 引数なしで実行したときの自動算出（compute_trend_factor）は、05 の過去年検証の
    平均誤差率（所得の外挿・対象者の固定などをすべて含む誤差）を打ち消すものである。
    人口の増減も誤差に含まれるが、人口として個別には見ていない。
    --factor を指定すると自動算出は使わない（人口の想定と自動算出は併用できない）。
  ※ tax_reform.py の REFORMS の macro_correction に人口減少の行を追加しても効かない
    （apply_macro_reforms が扱うのは dependent_income_limit /
     special_dependent_allowance の2つのみ。それ以外は「未実装」警告を出して無視する）。
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

# data/ 等の相対パスはカレントディレクトリ基準のため、02_src_py/ の中から実行した場合は
# プロジェクトルートへ戻す（2026-09-17追加）。ルートから実行した場合は何もしない。
# import は sys.path（スクリプトの置き場所）を見るため、chdir しても config / tax_reform は読める。
if os.path.basename(os.getcwd()) == "02_src_py":
    os.chdir("..")

from tax_reform import load_reforms, print_reform_summary
from config import (
    PREPARED_DATA_PATH,
    PREDICT_YEAR, TARGET_COL, CONFORMAL_COVERAGE,
)

BACKTEST_PATH = "data/05out_forecast_backtest.csv"   # 05 の過去年検証（トレンド補正の根拠）


# ─── トレンド補正乗率の算出 ───────────────────────────────────────────────────
def compute_trend_factor(bt_df: pd.DataFrame) -> tuple[float, str]:
    """
    05 の過去年検証（05out_forecast_backtest.csv）の平均誤差率からトレンド補正乗率を算出する。
    過去年検証は、05 と同じ方法（直近年をコピーして所得を伸ばす）で過去の各年を予測し、実績と比べたもの。

    誤差率 = (予測 − 実績) / 実績
    乗率   = 1 / (1 + 平均誤差率)

    例: 平均誤差率 +2% → 乗率 0.980（予測を 2% 引き下げ）
        平均誤差率 -1% → 乗率 1.010（予測を 1% 引き上げ）

    「人口補正後誤差率_%」の列があればそちらを使う（人口乗率を掛けた後に残る誤差だけを補正し、二重補正を防ぐ）。

    Returns:
        (乗率, 根拠の説明文)
    """
    if bt_df.empty:
        return 1.0, "過去年検証の結果が空 → 補正なし"
    err_col  = "人口補正後誤差率_%" if "人口補正後誤差率_%" in bt_df.columns else "誤差率_%"
    mean_err = bt_df[err_col].mean()
    std_err  = bt_df[err_col].std()
    factor   = round(1.0 / (1.0 + mean_err / 100.0), 4)
    years    = f"{bt_df['予測年度'].min()}〜{bt_df['予測年度'].max()}"
    basis    = f"05方式の過去年検証 {years} 平均{err_col.removesuffix('_%')} {mean_err:+.2f}%（標準偏差 {std_err:.2f}%）"
    return factor, basis


def compute_yearly_ci(bt_df: pd.DataFrame, total_oku: float, coverage: float):
    """
    05 の過去年検証の年ごとのぶれから、合計の信頼区間を作る。

    区間 = 最終予測 ×（1 ± t × 標準偏差 × √(1 + 1/年数)）
      - 標準偏差: 過去年検証の誤差率の標本標準偏差（将来の伸び率の当て外れなど、外挿の誤差を含む）
      - t: 自由度（年数 − 1）の t 分布の係数。年数が少ないほど大きく、年数が増えると正規分布の 1.96 に近づく
      - √(1 + 1/年数): トレンド補正に使う平均誤差率も同じ年数から推計しているため、そのぶれを含める分

    個人別の区間（split conformal）はモデルの個人単位の誤差だけを表すので、合計の区間にはこちらを使う。
    そのため、個人別の区間の合計とは一致しない。

    Returns:
        (下限_億円, 上限_億円, 方式の説明文)。年数が2年未満なら None
    """
    err_col = "人口補正後誤差率_%" if "人口補正後誤差率_%" in bt_df.columns else "誤差率_%"
    errs    = bt_df[err_col].dropna()
    n       = len(errs)
    if n < 2:
        return None
    sd   = errs.std() / 100.0
    k    = stats.t.ppf(0.5 + coverage / 2, n - 1)
    half = k * sd * np.sqrt(1 + 1 / n)
    method = f"過去年検証の年ごとのぶれ（t分布・{n}年、標準偏差 {sd * 100:.2f}%）"
    return total_oku * (1 - half), total_oku * (1 + half), method


# ─── 税制改正マクロ補正 ───────────────────────────────────────────────────────
    """
    個人レベルで反映できない税制改正を集計レベルで補正する。
    tax_reform.py の REFORMS の macro_correction タイプを読み込んで適用する。

    Returns:
        (補正後合計_億円, 補正明細リスト)
    """

def apply_macro_reforms(
    pred_total_oku: float,
    n_persons: int,
    target_year: int,
    df_prep: pd.DataFrame,
) -> tuple[float, list]:
    reforms = load_reforms(
        target_year=target_year, reform_type="macro_correction"
    )
    print_reform_summary(reforms, label="macro_correction")

    adjustments = []
    total = pred_total_oku

    # 【有効にする場合】02_src_py/tax_reform.py の REFORMS で以下を設定する（このファイルの修正は不要）。
    # dependent_income_limit / special_dependent_allowance は reform_type=macro_correction・active=False で登録済み。
    #
    #   1. active を True に変更（変更後は uv run python 02_src_py/tax_reform.py で記録を更新）
    #   2. 必要に応じて params に下記 if/elif が参照するキーを追加する（未設定なら既定値で計算される）
    #        dependent_income_limit     : new_dependent_rate（既定0.002）, deduction_per_person（既定330000）
    #        special_dependent_allowance: target_rate（既定0.003）, deduction_per_person（既定450000）
    #      ※ 既存の old_limit/new_limit・age_from/age_to は改正内容の記録で、計算には使われない
    for r in reforms:
        name   = r["name"]
        params = r["params"]

        if name == "dependent_income_limit":
            # 扶養要件引き上げ（扶養可能所得上限 48万→58万円）
            # 新たに扶養に入る人数を推計し、1人あたり33万円の控除増 × 10% = 3.3万円減
            rate       = float(params.get("new_dependent_rate", 0.002))
            new_dep_n  = int(n_persons * rate)
            deduct_pp  = float(params.get("deduction_per_person", 330_000))
            tax_effect = -new_dep_n * deduct_pp * 0.10 / 1e8
            total     += tax_effect
            msg = (f"  扶養要件引き上げ: +{new_dep_n:,}人 × "
                   f"{deduct_pp/1e4:.0f}万控除 × 10% = {tax_effect:.3f}億円")
            print(msg)
            adjustments.append({"name": name, "effect_oku": round(tax_effect, 4), "memo": msg.strip()})

        elif name == "special_dependent_allowance":
            # 特定親族特別控除（19-22歳扶養親族への追加控除）
            rate        = float(params.get("target_rate", 0.003))
            target_n    = int(n_persons * rate)
            deduct_pp   = float(params.get("deduction_per_person", 450_000))
            tax_effect  = -target_n * deduct_pp * 0.10 / 1e8
            total      += tax_effect
            msg = (f"  特定親族特別控除: {target_n:,}人 × "
                   f"{deduct_pp/1e4:.0f}万控除 × 10% = {tax_effect:.3f}億円")
            print(msg)
            adjustments.append({"name": name, "effect_oku": round(tax_effect, 4), "memo": msg.strip()})

        else:
            print(f"  ⚠ 未実装の macro_correction: {name}")

    return total, adjustments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=PREDICT_YEAR,
                        help=f"予測年度（デフォルト: {PREDICT_YEAR}）")
    parser.add_argument("--no-trend", action="store_true",
                        help="トレンド補正をスキップする")
    parser.add_argument("--factor", type=float, default=None,
                        help="トレンド補正乗率を直接指定（例: 0.98）")
    args = parser.parse_args()

    print(f"=== 06: {args.year}年度 トレンド・マクロ補正 ===\n")

    pred_path = f"data/05out_prediction_{args.year}.csv"
    if not os.path.exists(pred_path):
        print(f"エラー: {pred_path} がありません。先に 05_predict_2026.py を実行してください。")
        return

    pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
    pred_col = "pred_tax_amount"
    pred_tax = pred_df[pred_col].values.astype(float)
    total_before_oku = pred_tax.sum() / 1e8
    n_persons = len(pred_df)

    print(f"補正前合計: {total_before_oku:.2f} 億円 ({n_persons:,}人)\n")

    # ── A. トレンド補正 ───────────────────────────────────────────────────────
    if args.no_trend:
        trend_factor = 1.0
        trend_basis  = "スキップ（--no-trend）"
        print("トレンド補正: スキップ（--no-trend）")
    elif args.factor is not None:
        trend_factor = args.factor
        trend_basis  = "直接指定（--factor）"
        print(f"トレンド補正乗率（直接指定）: {trend_factor:.4f}")
    elif os.path.exists(BACKTEST_PATH):
        bt_df        = pd.read_csv(BACKTEST_PATH, encoding="utf-8-sig")
        trend_factor, trend_basis = compute_trend_factor(bt_df)
        print(f"トレンド補正（自動算出）: {trend_basis} → 乗率 {trend_factor:.4f}")
    else:
        trend_factor = 1.0
        trend_basis  = f"{BACKTEST_PATH} なし → 補正なし"
        print(f"トレンド補正: {BACKTEST_PATH} なし → 乗率 1.0（補正なし）。先に 05_predict_2026.py を実行してください")

    pred_tax_after_trend = (pred_tax * trend_factor).round(0).astype(int)
    total_after_trend    = pred_tax_after_trend.sum() / 1e8
    print(f"  補正後合計: {total_after_trend:.2f} 億円"
          f"  (差分: {(total_after_trend - total_before_oku):+.3f}億円)\n")

    # ── B. 税制改正マクロ補正 ─────────────────────────────────────────────────
    print("── 税制改正マクロ補正 ──")
    df_prep = pd.read_csv(PREPARED_DATA_PATH, encoding="utf-8-sig")
    total_after_macro, adjustments = apply_macro_reforms(
        total_after_trend, n_persons, args.year, df_prep
    )
    macro_effect = total_after_macro - total_after_trend

    # マクロ補正の効果を個人別に按分（比率補正）
    if total_after_trend > 0 and macro_effect != 0:
        macro_factor = total_after_macro / total_after_trend
        pred_tax_final = (pred_tax_after_trend * macro_factor).round(0).astype(int)
    else:
        pred_tax_final = pred_tax_after_trend.copy()

    total_final = pred_tax_final.sum() / 1e8

    # ── 結果表示 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  補正前（05 出力）       : {total_before_oku:>8.2f} 億円")
    if trend_factor != 1.0:
        print(f"  トレンド補正後          : {total_after_trend:>8.2f} 億円  (× {trend_factor:.4f})")
    if adjustments:
        print(f"  税制改正マクロ補正      : {macro_effect:>+8.3f} 億円")
    print(f"  最終予測合計            : {total_final:>8.2f} 億円")
    print(f"{'='*60}")

    # ── CSV 出力 ─────────────────────────────────────────────────────────────
    out_df = pred_df.copy()
    out_df[pred_col] = pred_tax_final

    # 信頼区間列も同率で補正
    for col in out_df.columns:
        if col.startswith("pred_tax_lower") or col.startswith("pred_tax_upper"):
            out_df[col] = (out_df[col].values * trend_factor
                           * (macro_factor if adjustments else 1.0)).round(0).astype(int)

    out_path = f"data/06out_prediction_adjusted_{args.year}.csv"
    sum_path = f"data/06out_prediction_adjusted_summary_{args.year}.csv"

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 合計の信頼区間: 過去年検証の年ごとのぶれから作る（compute_yearly_ci）。
    # 個人別の区間（上で補正した列）はモデルの個人単位の誤差だけを表すので、その合計は参考として残す。
    pct       = int(round(CONFORMAL_COVERAGE * 100))
    lower_col = next((c for c in out_df.columns if c.startswith("pred_tax_lower")), None)
    upper_col = next((c for c in out_df.columns if c.startswith("pred_tax_upper")), None)
    ref_ci    = None
    if lower_col and upper_col:
        ref_ci = (out_df[lower_col].sum() / 1e8, out_df[upper_col].sum() / 1e8)

    yearly_ci = None
    if os.path.exists(BACKTEST_PATH):
        yearly_ci = compute_yearly_ci(pd.read_csv(BACKTEST_PATH, encoding="utf-8-sig"), total_final, CONFORMAL_COVERAGE)
    if yearly_ci is not None:
        ci_low, ci_high, ci_method = yearly_ci
    elif ref_ci is not None:
        ci_low, ci_high = ref_ci
        ci_method = f"個人別の区間の合計（{BACKTEST_PATH} がないか、2年未満のため）"
    else:
        ci_low = ci_high = ci_method = None

    ci_cols = {}
    if ci_method is not None:
        ci_cols = {
            f"CI下限_{pct}%_億円": round(ci_low, 2),
            f"CI上限_{pct}%_億円": round(ci_high, 2),
            "CI方式"            : ci_method,
        }
        print(f"  {pct}%信頼区間          : {ci_low:.2f} 〜 {ci_high:.2f} 億円（{ci_method}）")
    if ref_ci is not None:
        ci_cols["参考_個人区間合計_下限_億円"] = round(ref_ci[0], 2)
        ci_cols["参考_個人区間合計_上限_億円"] = round(ref_ci[1], 2)
    summary_rows = [{
        "予測年度"          : args.year,
        "補正前合計_億円"   : round(total_before_oku, 2),
        "トレンド補正乗率"  : trend_factor,
        "トレンド補正_根拠" : trend_basis,
        "マクロ補正効果_億円": round(macro_effect, 3),
        "最終予測合計_億円" : round(total_final, 2),
        **ci_cols,
        "予測人員"          : n_persons,
    }]
    for adj in adjustments:
        summary_rows[0][f"macro_{adj['name']}_億円"] = adj["effect_oku"]

    pd.DataFrame(summary_rows).to_csv(sum_path, index=False, encoding="utf-8-sig")

    print(f"\n→ {out_path} に補正後個人別予測を保存")
    print(f"→ {sum_path} に補正後サマリーを保存")
    print("\n次: python 07_visualize.py")


if __name__ == "__main__":
    main()
