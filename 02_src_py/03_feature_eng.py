"""
03_feature_eng.py
=================
個人住民税予測モデル - Step3: 特徴量エンジニアリング

【実行方法】
  python 03_feature_eng.py
  python 03_feature_eng.py --file data/01out_individual_raw.csv

【import】
data/01out_individual_raw.csv（実データ or ダミーデータ）

【export】
  data/03out_individual_prepared.csv（特徴量追加済みデータ）
     └─ 04_model_train.py で学習用データとして使用

【生成特徴量の設計意図と検証結果 2026-09-12追記】
  preprocess() 内の各特徴量には「意図（なぜ実装したか）」と「効果（外すとどうなるか）」を
  コメントで併記している。効果の数値は 2026-09-11 に実施したアブレーション検証の結果
  （ダミーデータ597,000件 / TRAIN 2020-2024 → TEST 2025 の1回ホールドアウト / 各1試行）。
  検証条件・全結果一覧・注意点は memo.md「2026-09-11: 03_feature_eng.py の生成特徴量の
  アブレーション検証」を参照。

  要点:
    - 03の生成特徴量全体で重要度シェア31.2%。全て外すと MAE +84円(+7.8%)・WMAPE +0.043pt 悪化。
    - 効果が明確なのは 比率特徴量 > income_total > deduct_total の順。
    - 所得種別フラグ5個（has_sep_income以外）は重要度0.00で未使用、前年特徴量5個は
      外しても精度が落ちない（当年の所得・控除列と情報が重複）。
    - ただしダミーデータは tax_reform.py の計算式で税額を決定的に生成しており R2=0.9998 と
      異常に高いため、追加特徴量の効果が出にくい。実データでの再検証が必要（特に前年特徴量）。
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
    # 意図: 非課税判定（地方税法295条）も各種控除の判定も「合計所得金額」が基準。個別所得列だけ
    #       与えると木モデルは加算を閾値分割の組合せでしか近似できないため、合計を明示的に渡して
    #       非課税ライン付近の分割を1回で済ませる。
    # 効果: 除外すると MAE +29円 / WMAPE +0.015pt 悪化（重要度4.60%）。意図どおり寄与している。
    inc_cols = [c for c in ALL_INCOME_COLS if c in df.columns]
    df["income_total"] = df[inc_cols].sum(axis=1)

    # ── 所得種別フラグ ──────────────────────────────────────────────────────────
    # 各所得が0より大きければ1フラグをたてる
    # なお、給与所得は「income_salary_gross」列があればそれを優先し、なければ「income_salary」を使用する
    # 意図: 給与所得者／年金受給者／事業所得者という納税者類型ごとに税額構造が違うという仮説。
    # 効果: has_sep_income 以外の5個は重要度0.00＝一度も分割に使われていない。6個まとめて除外しても
    #       WMAPE +0.001pt / MAE +3円 のみで、`income_salary_gross > 0` のような分割を木が自力で
    #       作れるため冗長だった。複数列の合計を要する has_sep_income だけは例外的に0.73%使用。
    #       → 5個は削除候補だが、実データで同じ結果になるかは未確認のため現状は残している。
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
    # 意図: 「課税標準額＝合計所得−所得控除合計」という税額計算の骨格そのもの。14種類の控除列の
    #       和を明示的に渡す（理由は income_total と同じく木モデルが加算を苦手とするため）。
    # 効果: 除外すると MAE +24円 / WMAPE +0.012pt 悪化。重要度5.06%で03生成特徴量の中で最大。
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
    # 意図: 控除率・課税所得率という正規化指標。所得水準が違っても「控除の厚い人／薄い人」を同一
    #       スケールで比較でき、絶対額のみの場合に必要な所得帯ごとの閾値分割を節約できる。
    # 効果: 2個まとめて除外すると MAE +35円 / WMAPE +0.018pt 悪化し、03生成特徴量の中で最も
    #       効果が大きい（deduct_rate 4.67% / taxable_rate 2.67%）。設計意図が検証で裏付けられた。
    safe_total = df["income_total"].replace(0, np.nan)
    df["deduct_rate"]  = (df["deduct_total"]    / safe_total).fillna(0).clip(0, 1)
    df["taxable_rate"] = (df["taxable_income"]  / safe_total).fillna(0).clip(0, 1)

    # ── 年齢 → 数値区分 ─────────────────────────────────────────────────────────
    # age列（実年齢）があれば直接区分化、なければ age_group ラベルからマッピング
    # 意図: age_group は文字列カテゴリでLightGBMに直接渡せないため、順序性（若年→高齢）を保った
    #       数値に変換する。年金受給開始年齢（65歳）や扶養控除の年齢要件との関係を捉える狙い。
    # 効果: 除外しても MAE +4円 / WMAPE +0.002pt と寄与は小さい（重要度1.20%）。ダミーデータでは
    #       年齢の影響が所得列（年金収入等）に既に織り込まれているためと考えられる。
    if "age" in df.columns:
        # 2026-09-12変更: 旧「80以上」を 80-84/85-89/90over に分割したため13区分→15区分
        df["age_num"] = pd.cut(
            df["age"],
            bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 200],
            labels=list(range(1, 16)),
            right=False,
        ).astype(int)
    else:
        df["age_num"] = df["age_group"].map(AGE_MAP).fillna(7).astype(int)

    # ── 前年データとの時系列結合 ────────────────────────────────────────────────
    # 意図: 住民税は前年所得課税であり、実務上「前年の税額」が翌年予測の最強の予測子になるという
    #       想定。転入・新規課税者（前年データなし）は is_continuing で区別する。
    # 効果: 重要度は高い（prev_tax_amount 4.18% / income_yoy_change 3.98%）が、5個まとめて除外
    #       しても WMAPE -0.001pt / MAE -1円 と精度が落ちない（誤差範囲で微改善）。当年の所得・
    #       控除列が完全に揃っているため情報が重複しており、木が「使いやすいので使っているだけ」
    #       の状態と考えられる。
    #       ※実データでは申告遅れ・未申告で当年データが不完全なケースがあり前年値の価値が変わる
    #         可能性が高いため、削除せず残している。実データでの再検証が必要。
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
    # 意図: 給与天引き（特別徴収）か自主納付（普通徴収）かで納税者層（給与所得者 vs 年金・事業
    #       所得者）が分かれるという想定。
    # 効果: 除外しても MAE +4円 / WMAPE +0.002pt、重要度0.11%とほぼ未使用。元の collection_type
    #       列と情報が重複している。
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
