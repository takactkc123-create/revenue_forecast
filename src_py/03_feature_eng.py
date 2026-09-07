"""
03_feature_eng.py
=================
個人住民税予測モデル - Step3: 特徴量エンジニアリング

【実行方法】
  python 03_feature_eng.py
  python 03_feature_eng.py --file data/individual_raw.csv

【import】
data/individual_raw.csv（実データ or ダミーデータ）

【export】
  data/individual_prepared.csv（特徴量追加済みデータ）
     └─ 04_model_train.py で学習用データとして使用

"""

import argparse
import os
import numpy as np
import pandas as pd
from tax_reform import (
    compute_basic_deduction,
    estimate_furusato_resident_deduction,
)
from config import (
    RAW_DATA_PATH, PREPARED_DATA_PATH, SUMMARY_PATH,
    FURUSATO_PARAMS, AGE_MAP, ALL_INCOME_COLS,
)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)

    # ── リーク列の除外 ──────────────────────────────────────────────────────────
    # 均等割・所得割の合計 = 年税額（目的変数）のため説明変数に使用しない
    # 実データCSVにこれらの列が含まれている場合もここでdropする
    LEAK_COLS = ["muni_kintowari", "muni_tokuwari", "pref_kintowari", "pref_tokuwari", "shortfall"]
    df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])

    # ── 合計所得（ALL_INCOME_COLS に含まれる列を合計。存在しない列は0扱い）──────
    inc_cols = [c for c in ALL_INCOME_COLS if c in df.columns]
    df["income_total"] = df[inc_cols].sum(axis=1)

    # ── 所得種別フラグ ──────────────────────────────────────────────────────────
    # 各所得が0より大きければ1フラグをたてる
    # なお、給与所得は「income_salary_gross」列があればそれを優先し、なければ「income_salary」を使用する
    salary_base = (
        df["income_salary_gross"]
        if "income_salary_gross" in df.columns
        else df["income_salary"]
    )
    df["has_salary"]   = (salary_base > 0).astype(int)
    df["has_business"] = (df.get("income_business", 0) > 0).astype(int)
    df["has_pension"]  = (df.get("income_pension",  0) > 0).astype(int)
    df["has_property"] = (df.get("income_property", 0) > 0).astype(int)
    df["has_dividend"] = (df.get("income_dividend", 0) > 0).astype(int)
    # 分離課税所得のいずれかを保有するフラグ
    sep_cols = [c for c in ALL_INCOME_COLS if c.startswith("sep_") and c in df.columns]
    df["has_sep_income"] = (df[sep_cols].sum(axis=1) > 0).astype(int) if sep_cols else 0

    # ── 基礎控除（列がなければ自動計算）────────────────────────────────────────
    if "deduct_basic" not in df.columns:
        df["deduct_basic"] = (
            compute_basic_deduction(df["income_total"].values)
            .round(0).astype(int)
        )

    # ── 所得控除の合計 ──────────────────────────────────────────────────────────
    deduct_cols = [
        "deduct_social_ins", "deduct_small_biz_ins", "deduct_life_ins",
        "deduct_earthquake_ins", "deduct_casualty", "deduct_medical",
        "deduct_disability", "deduct_widow", "deduct_spouse", "deduct_spouse_special",
        "deduct_dependent", "deduct_basic", "deduct_working_student",
        "deduct_donation_income",
    ]
    df["deduct_total"] = sum(
        df[c] if c in df.columns else pd.Series(0, index=df.index)
        for c in deduct_cols
    )

    # ── 税額控除（列がなければ概算補完）────────────────────────────────────────
    if "deduct_tax_furusato" not in df.columns:
        df["deduct_tax_furusato"] = estimate_furusato_resident_deduction(
            df["taxable_income"].values,
            donation_rate=FURUSATO_PARAMS["donation_rate"],
            one_stop_ratio=FURUSATO_PARAMS["one_stop_ratio"],
        ).astype(int)
    if "deduct_tax_housing" not in df.columns:
        df["deduct_tax_housing"] = 0

    # ── 比率特徴量 ──────────────────────────────────────────────────────────────
    safe_total = df["income_total"].replace(0, np.nan)
    df["deduct_rate"]  = (df["deduct_total"]    / safe_total).fillna(0).clip(0, 1)
    df["taxable_rate"] = (df["taxable_income"]  / safe_total).fillna(0).clip(0, 1)

    # ── 年齢 → 数値区分 ─────────────────────────────────────────────────────────
    # age列（実年齢）があれば直接区分化、なければ age_group ラベルからマッピング
    if "age" in df.columns:
        df["age_num"] = pd.cut(
            df["age"],
            bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 200],
            labels=list(range(1, 14)),
            right=False,
        ).astype(int)
    else:
        df["age_num"] = df["age_group"].map(AGE_MAP).fillna(7).astype(int)

    # ── 前年データとの時系列結合 ────────────────────────────────────────────────
    prev = df[df["year"] < df["year"].max()][
        ["person_id", "year", "income_total", "tax_amount", "taxable_income"]
    ].copy()
    prev["year"] = prev["year"] + 1
    prev.columns = [
        "person_id", "year",
        "prev_income_total", "prev_tax_amount", "prev_taxable_income",
    ]
    df = df.merge(prev, on=["person_id", "year"], how="left")

    df["income_yoy_change"] = (df["income_total"] - df["prev_income_total"]).fillna(0)
    df["is_continuing"]     = df["prev_income_total"].notna().astype(int)

    # ── 徴収区分フラグ ──────────────────────────────────────────────────────────
    df["is_tokubetsu"] = (df["collection_type"] == 1).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=RAW_DATA_PATH,
                        help=f"入力CSVパス（デフォルト: {RAW_DATA_PATH}）")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    print("=== 03: 特徴量エンジニアリング ===\n")

    print(f"読込: {args.file}")
    df_raw = pd.read_csv(args.file, encoding="utf-8-sig")
    print(f"  {len(df_raw):,} 件 / {df_raw['year'].nunique()} 年分\n")

    print("── 年度別レコード数（前処理前） ──")
    print(df_raw.groupby("year")["tax_amount"].agg(
        件数="count",
        税額合計_億円=lambda x: round(x.sum() / 1e8, 2),
        一人あたり平均_万円=lambda x: round(x.mean() / 1e4, 1),
    ).to_string())

    print("\n前処理・特徴量生成中...")
    df_prep = preprocess(df_raw)
    print(f"  完了: {len(df_prep.columns)} 列")

    df_prep.to_csv(PREPARED_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"\n→ {PREPARED_DATA_PATH} に保存 ({len(df_prep):,} 件)")

    summary = df_raw.groupby("year")["tax_amount"].sum().reset_index()
    summary.columns = ["year", "tax_total"]
    summary["tax_total_oku"] = (summary["tax_total"] / 1e8).round(2)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"→ {SUMMARY_PATH} に年度別集計を保存")
    print("\n次: python 04_model_train.py")


if __name__ == "__main__":
    main()
