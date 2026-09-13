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

【生成特徴量の設計意図と検証結果 2026-09-12追記 / 2026-09-13 数値更新】
  preprocess() 内の各特徴量には「意図（なぜ実装したか）」と「効果（外すとどうなるか）」を
  コメントで併記している。効果の数値は 2026-09-13 に再実行したアブレーション検証の結果
  （ダミーデータ生成の不具合修正後 / 597,000件 / TRAIN 2020-2024 → TEST 2025 の1回ホールドアウト / 各1試行）。
  検証条件・前回との比較・注意点は memo.md「2026-09-13: ダミーデータ生成の不具合修正を採用し、
  アブレーション・税収分解の考察を改訂」を参照。

  要点:
    - 03の生成特徴量全体で重要度シェア32.3%。全て外すと MAE +143円(+11.6%)・WMAPE +0.062pt 悪化。
    - 効果が明確なのは 比率特徴量 > 差引所得控除合計 > 総所得金額等 の順。
    - 所得種別フラグ5個（分離所得有無以外）は重要度0.00で未使用、前年特徴量5個は
      外しても精度が落ちない（当年の所得・控除列と情報が重複）。
    - ただしダミーデータは tax_reform.py の計算式で税額を決定的に生成しており R2=0.9997 と
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
    RAW_FEATURE_COLS, GENERATED_FEATURE_COLS,
)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["仮ID", "年度"]).reset_index(drop=True)
    before_cols = set(df.columns)   # 2026-09-13追加: 作った列を最後に確認するため、処理前の列を記録

    # ── リーク列の除外 ──────────────────────────────────────────────────────────
    # 均等割・所得割の合計 = 年税額（目的変数）のため説明変数に使用しない
    # 実データCSVにこれらの列が含まれている場合もここでdropする
    LEAK_COLS = ["市町村_均等割", "市町村_所得割", "都道府県_均等割", "都道府県_所得割", "控除不足額"]
    df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])

    # ── 合計所得（ALL_INCOME_COLS に含まれる列を合計。存在しない列は0扱い）──────
    # 意図: 非課税判定（地方税法295条）も各種控除の判定も「合計所得金額」が基準。個別所得列だけ
    #       与えると木モデルは加算を閾値分割の組合せでしか近似できないため、合計を明示的に渡して
    #       非課税ライン付近の分割を1回で済ませる。
    # 効果: 除外すると MAE +27円 / WMAPE +0.012pt 悪化（重要度3.72%）。意図どおり寄与している。
    inc_cols = [c for c in ALL_INCOME_COLS if c in df.columns]
    df["総所得金額等"] = df[inc_cols].sum(axis=1)

    # ── 所得種別フラグ ──────────────────────────────────────────────────────────
    # 各所得が0より大きければ1フラグをたてる
    # なお、給与所得は「給与収入」列があればそれを優先し、なければ「給与所得」を使用する
    # 意図: 給与所得者／年金受給者／事業所得者という納税者類型ごとに税額構造が違うという仮説。
    # 効果: 分離所得有無 以外の5個は重要度0.00＝一度も分割に使われていない。6個まとめて除外しても
    #       WMAPE ±0.000pt / MAE ±0円 と変化がなく、`給与収入 > 0` のような分割を木が自力で
    #       作れるため冗長だった。複数列の合計を要する 分離所得有無 だけは例外的に0.58%使用。
    #       → 5個は削除候補だが、実データで同じ結果になるかは未確認のため現状は残している。
    salary_base = (
        df["給与収入"]
        if "給与収入" in df.columns
        else df["給与所得"]
    )
    df["給与所得有無"]   = (salary_base > 0).astype(int)
    df["事業所得有無"] = (df.get("事業所得_営業等", 0) > 0).astype(int)
    df["年金所得有無"]  = (df.get("雑所得_公的年金等",  0) > 0).astype(int)
    df["不動産所得有無"] = (df.get("不動産所得", 0) > 0).astype(int)
    df["配当所得有無"] = (df.get("配当所得", 0) > 0).astype(int)
    # 分離課税所得のいずれかを保有するフラグ
    sep_cols = [c for c in ALL_INCOME_COLS if c.startswith("分離_") and c in df.columns]
    df["分離所得有無"] = (df[sep_cols].sum(axis=1) > 0).astype(int) if sep_cols else 0

    # ── 基礎控除（列がなければ自動計算）────────────────────────────────────────
    if "基礎控除" not in df.columns:
        df["基礎控除"] = (
            compute_basic_deduction(df["総所得金額等"].values)
            .round(0).astype(int)
        )

    # ── 所得控除の合計 ──────────────────────────────────────────────────────────
    # 意図: 「課税標準額＝合計所得−所得控除合計」という税額計算の骨格そのもの。14種類の控除列の
    #       和を明示的に渡す（理由は 総所得金額等 と同じく木モデルが加算を苦手とするため）。
    # 効果: 除外すると MAE +42円 / WMAPE +0.018pt 悪化。重要度7.16%で03生成特徴量の中で最大。
    deduct_cols = [
        "社会保険料控除", "小規模企業共済等掛金控除", "生命保険料控除",
        "地震保険料控除", "雑損控除", "医療費控除",
        "障害者控除", "寡婦控除", "配偶者控除", "配偶者特別控除",
        "扶養控除", "基礎控除", "勤労学生控除",
        "寄附金控除",
    ]
    df["差引所得控除合計"] = sum(
        df[c] if c in df.columns else pd.Series(0, index=df.index)
        for c in deduct_cols
    )

    # ── 税額控除（列がなければ概算補完）────────────────────────────────────────
    if "寄附金税額控除" not in df.columns:
        df["寄附金税額控除"] = estimate_furusato_resident_deduction(
            df["課税標準額"].values,
            donation_rate=FURUSATO_PARAMS["donation_rate"],
            one_stop_ratio=FURUSATO_PARAMS["one_stop_ratio"],
        ).astype(int)
    if "住宅借入金特別控除" not in df.columns:
        df["住宅借入金特別控除"] = 0

    # ── 比率特徴量 ──────────────────────────────────────────────────────────────
    # 意図: 控除率・課税所得率という正規化指標。所得水準が違っても「控除の厚い人／薄い人」を同一
    #       スケールで比較でき、絶対額のみの場合に必要な所得帯ごとの閾値分割を節約できる。
    # 効果: 2個まとめて除外すると MAE +79円 / WMAPE +0.034pt 悪化し、03生成特徴量の中で最も
    #       効果が大きい（所得控除率 4.25% / 課税標準率 3.52%）。設計意図が検証で裏付けられた。
    safe_total = df["総所得金額等"].replace(0, np.nan)
    df["所得控除率"]  = (df["差引所得控除合計"]    / safe_total).fillna(0).clip(0, 1)
    df["課税標準率"] = (df["課税標準額"]  / safe_total).fillna(0).clip(0, 1)

    # ── 年齢 → 数値区分 ─────────────────────────────────────────────────────────
    # age列（実年齢）があれば直接区分化、なければ 年齢区分 ラベルからマッピング
    # 意図: 年齢区分 は文字列カテゴリでLightGBMに直接渡せないため、順序性（若年→高齢）を保った
    #       数値に変換する。年金受給開始年齢（65歳）や扶養控除の年齢要件との関係を捉える狙い。
    # 効果: 除外しても MAE +1円 / WMAPE ±0.000pt と寄与はほぼない（重要度1.73%）。ダミーデータでは
    #       年齢の影響が所得列（年金収入等）に既に織り込まれているためと考えられる。
    if "年齢" in df.columns:
        # 2026-09-12変更: 旧「80以上」を 80-84/85-89/90over に分割したため13区分→15区分
        df["年齢区分番号"] = pd.cut(
            df["年齢"],
            bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 200],
            labels=list(range(1, 16)),
            right=False,
        ).astype(int)
    else:
        df["年齢区分番号"] = df["年齢区分"].map(AGE_MAP).fillna(7).astype(int)

    # ── 前年データとの時系列結合 ────────────────────────────────────────────────
    # 意図: 住民税は前年所得課税であり、実務上「前年の税額」が翌年予測の最強の予測子になるという
    #       想定。転入・新規課税者（前年データなし）は 継続者フラグ で区別する。
    # 効果: 重要度は高い（総所得金額等_前年差 4.43% / 前年_年税額 2.88%）が、5個まとめて除外
    #       しても WMAPE -0.001pt / MAE -2円 と精度が落ちない（前年_年税額だけ外すと MAE -15円 と
    #       逆にわずかに改善）。当年の所得・控除列が完全に揃っているため情報が重複しており、
    #       木が「使いやすいので使っているだけ」の状態と考えられる。
    #       ※実データでは申告遅れ・未申告で当年データが不完全なケースがあり前年値の価値が変わる
    #         可能性が高いため、削除せず残している。実データでの再検証が必要。
    prev = df[df["年度"] < df["年度"].max()][
        ["仮ID", "年度", "総所得金額等", "年税額", "課税標準額"]
    ].copy()
    prev["年度"] = prev["年度"] + 1
    prev.columns = [
        "仮ID", "年度",
        "前年_総所得金額等", "前年_年税額", "前年_課税標準額",
    ]
    df = df.merge(prev, on=["仮ID", "年度"], how="left")

    df["総所得金額等_前年差"] = (df["総所得金額等"] - df["前年_総所得金額等"]).fillna(0)
    df["継続者フラグ"]     = df["前年_総所得金額等"].notna().astype(int)

    # ── 徴収区分フラグ ──────────────────────────────────────────────────────────
    # 意図: 給与天引き（特別徴収）か自主納付（普通徴収）かで納税者層（給与所得者 vs 年金・事業
    #       所得者）が分かれるという想定。
    # 効果: 除外しても MAE -5円 / WMAPE -0.002pt と精度が落ちず、重要度0.25%とほぼ未使用。
    #       元の 徴収区分 列と情報が重複している。
    df["特別徴収フラグ"] = (df["徴収区分"] == 1).astype(int)

    # ── config.py の GENERATED_FEATURE_COLS と、実際に作った列が一致するか確認（2026-09-13追加）──
    # ずれたまま進むと、04は「列なし（スキップ）」と表示するだけで学習を続け、特徴量が黙って欠けるため、ここで止める。
    # RAW_FEATURE_COLS も差し引くのは、実データ向けに「列がなければ補う」基礎控除等を一覧外の新規列と誤判定しないため。
    not_created = [c for c in GENERATED_FEATURE_COLS if c not in df.columns]
    unlisted = sorted(set(df.columns) - before_cols - set(GENERATED_FEATURE_COLS) - set(RAW_FEATURE_COLS))
    if not_created:
        raise ValueError(f"config.py の GENERATED_FEATURE_COLS にあるのに03で作っていない列: {not_created}")
    if unlisted:
        raise ValueError(f"03で作ったのに config.py の GENERATED_FEATURE_COLS に書いていない列: {unlisted}")

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
    print(f"  {len(df_raw):,} 件 / {df_raw['年度'].nunique()} 年分\n")

    print("── 年度別レコード数（前処理前） ──")
    print(df_raw.groupby("年度")["年税額"].agg(
        件数="count",
        税額合計_億円=lambda x: round(x.sum() / 1e8, 2),
        一人あたり平均_万円=lambda x: round(x.mean() / 1e4, 1),
    ).to_string())

    print("\n前処理・特徴量生成中...")
    df_prep = preprocess(df_raw)
    print(f"  完了: {len(df_prep.columns)} 列")

    df_prep.to_csv(PREPARED_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"\n→ {PREPARED_DATA_PATH} に保存 ({len(df_prep):,} 件)")

    summary = df_raw.groupby("年度")["年税額"].sum().reset_index()
    summary.columns = ["年度", "tax_total"]
    summary["tax_total_oku"] = (summary["tax_total"] / 1e8).round(2)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"→ {SUMMARY_PATH} に年度別集計を保存")
    print("\n次: python 04_model_train.py")


if __name__ == "__main__":
    main()
