"""
06_trend_correction.py
個人住民税予測モデル - Step6: トレンド・税制改正マクロ補正

【補正の種類】
  A. トレンド補正（wage_trend_factor）
     - 年度別合算予測の系統的な過大・過小傾向を緩和する乗率補正
     - 04 の yearly_result.csv に基づいて自動算出（オプションで上書き可）

  B. 税制改正マクロ補正（tax_reform_config.csv の macro_correction）
     - 扶養要件引き上げ（2026年〜）: 扶養控除新規取得者の増加分を推計して加算
     - 特定親族特別控除（2026年〜）: 19-22歳扶養親族への追加控除を推計して減算

【import】
  data/prediction_YYYY.csv       ← 05 の出力
  data/yearly_result.csv         ← 04 の出力
  data/individual_prepared.csv   ← 03 の出力

【export】
  data/prediction_adjusted_YYYY.csv      ← 補正後の個人別予測値
  data/prediction_adjusted_summary_YYYY.csv ← 補正後の合計サマリー

【使い方】
  python 06_trend_correction.py
  python 06_trend_correction.py --year 2026
  python 06_trend_correction.py --no-trend   # トレンド補正をスキップ
  python 06_trend_correction.py --factor 0.98  # トレンド補正乗率を直接指定
"""

import argparse
import os
import numpy as np
import pandas as pd
from tax_reform import load_reforms, print_reform_summary
from config import (
    PREPARED_DATA_PATH, REFORM_CONFIG_PATH,
    PREDICT_YEAR, TARGET_COL,
)

YEARLY_PATH = "data/yearly_result.csv"


# ─── トレンド補正乗率の算出 ───────────────────────────────────────────────────
def compute_trend_factor(yearly_df: pd.DataFrame) -> float:
    """
    04 の年度別合算精度（yearly_result.csv）を使い、
    訓練年の平均誤差率からトレンド補正乗率を算出する。

    誤差率 = (予測 − 実測) / 実測
    乗率   = 1 / (1 + 平均誤差率)

    例: 平均誤差率 +2% → 乗率 0.980（予測を 2% 引き下げ）
        平均誤差率 -1% → 乗率 1.010（予測を 1% 引き上げ）
    """
    train_rows = yearly_df[~yearly_df["is_test"]]
    if train_rows.empty:
        return 1.0
    mean_err_rate = train_rows["error_rate_pct"].mean() / 100.0
    factor        = 1.0 / (1.0 + mean_err_rate)
    return round(factor, 4)


# ─── 税制改正マクロ補正 ───────────────────────────────────────────────────────
    """
    個人レベルで反映できない税制改正を集計レベルで補正する。
    tax_reform_config.csv の macro_correction タイプを読み込んで適用する。

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
        REFORM_CONFIG_PATH, target_year=target_year, reform_type="macro_correction"
    )
    print_reform_summary(reforms, label="macro_correction")

    adjustments = []
    total = pred_total_oku

    # 【呼び出す場合】data/tax_reform_config.csv で以下の対応が必要（このファイルの修正は不要）。
    # 現状 dependent_income_limit / special_dependent_allowance は reform_type=feature_correction の行として active=False で登録されており
    # （05 の個人特徴量補正用、現在は年齢等のデータ不足によりスキップのみ）、そのままでは下記 if/elif には一致しない。
    # 
    #   1. reform_type を feature_correction → macro_correction に変更
    #   2. active を True に変更
    #   3. param_key/param_value を下記 if/elif が参照するキーに合わせる
    #      （既存の old_limit/new_limit 行はそのままでは使われないので、新しい行として追加するかparam_key ごと置き換える）
    #      
    #        dependent_income_limit     : new_dependent_rate（既定0.002）, deduction_per_person（既定330000）
    #        special_dependent_allowance: target_rate（既定0.003）, deduction_per_person（既定450000）,
    #                                      age_from（既定19）, age_to（既定22）※age_from/age_toは既存行を流用可
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

    pred_path = f"data/prediction_{args.year}.csv"
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
        print("トレンド補正: スキップ（--no-trend）")
    elif args.factor is not None:
        trend_factor = args.factor
        print(f"トレンド補正乗率（直接指定）: {trend_factor:.4f}")
    elif os.path.exists(YEARLY_PATH):
        yearly_df    = pd.read_csv(YEARLY_PATH, encoding="utf-8-sig")
        trend_factor = compute_trend_factor(yearly_df)
        mean_err_pct = yearly_df[~yearly_df["is_test"]]["error_rate_pct"].mean()
        print(f"トレンド補正（自動算出）: 訓練年平均誤差率 {mean_err_pct:+.2f}% → 乗率 {trend_factor:.4f}")
    else:
        trend_factor = 1.0
        print(f"トレンド補正: {YEARLY_PATH} なし → 乗率 1.0（補正なし）")

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

    out_path = f"data/prediction_adjusted_{args.year}.csv"
    sum_path = f"data/prediction_adjusted_summary_{args.year}.csv"

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary_rows = [{
        "予測年度"          : args.year,
        "補正前合計_億円"   : round(total_before_oku, 2),
        "トレンド補正乗率"  : trend_factor,
        "マクロ補正効果_億円": round(macro_effect, 3),
        "最終予測合計_億円" : round(total_final, 2),
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
