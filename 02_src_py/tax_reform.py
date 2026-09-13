"""
tax_reform.py
=============
税制改正補正モジュール（04_model_train / 05_predict_2026 / 06_trend_correction から呼び出される共通モジュール）

tax_reform_config.csv から補正ルールを読み込み、
学習ラベル補正（label_correction）および予測特徴量補正（feature_correction）を適用する。

【使い方】
  from tax_reform import load_reforms, apply_reforms, print_reform_summary

  # 04_model_train.py: 学習ラベル補正（全訓練期間を対象に読み込む）
  reforms = load_reforms(path, target_year=2025, reform_type="label_correction")
  df_corrected = apply_reforms(df.copy(), reforms)

  # 05_predict_2026.py: 翌年特徴量補正
  reforms = load_reforms(path, target_year=2026, reform_type="feature_correction")
  df_reformed  = apply_reforms(df.copy(), reforms)

【CSV列の説明】
  reform_name    : 補正の識別子。REFORM_REGISTRY のキーと対応
  effective_year : 施行年度
  one_time       : True=その年のみ / False=以降継続（補正関数内で制御）
  active         : False なら読み込みから除外（未実装・データ不足など）
  reform_type    : label_correction または feature_correction
  param_key      : パラメータ名
  param_value    : パラメータ値（int/float を自動判定）
  memo           : 人間向け備考（コード上は不使用）

【補正追加の手順】
  1. tax_reform_config.csv に行を追加（reform_name, params）
  2. 計算式が新規なら _apply_<name> 関数を実装
  3. REFORM_REGISTRY にエントリを追加
"""

import numpy as np
import pandas as pd
from config import (
    NON_TAXABLE_PER_PERSON,
    NON_TAXABLE_FLAT,
    NON_TAXABLE_FAMILY_ADD,
)

TARGET_COL         = "年税額"
REFORM_CONFIG_PATH = "data/tax_reform_config.csv"


# ─────────────────────────────────────────────
# 給与所得控除ブラケット計算（住民税・所得税共通）
# ─────────────────────────────────────────────
def _calc_salary_deduction(income: np.ndarray, floor: float) -> np.ndarray:
    """
    給与所得控除額を 5段階ブラケット式で計算し、最低額 floor を適用する（令和2年分以降）。

    NTA の区分表記と max(式, floor) の関係:
      NTA は「〜1,625,000円: 一律 floor」と書くが、
      max(収入×40%−100,000, floor) と数学的に等価
      （1,625,000×40%−100,000 = floor=550,000 が境界）

    区分（令和2年分〜。令和7年分以降は floor のみ変更）:
      〜1,800,000円 : max(収入×40%−100,000,  floor)
      〜3,600,000円 : max(収入×30%+80,000,   floor)
      〜6,600,000円 : max(収入×20%+440,000,  floor)
      〜8,500,000円 : max(収入×10%+1,100,000, floor)
      8,500,000円超 : 1,950,000 円（上限）

    floor の影響範囲:
      令和2〜6年分 floor=55万: 給与収入 〜約162.5万円（第1区分のみ）
      令和7年分〜  floor=65万: 給与収入 〜約190万円（第1〜2区分にまたがる）
    """
    d = np.where(
        income <= 1_800_000,
        np.maximum(income * 0.40 - 100_000, floor),
        np.where(
            income <= 3_600_000,
            np.maximum(income * 0.30 + 80_000,  floor),
            np.where(
                income <= 6_600_000,
                np.maximum(income * 0.20 + 440_000, floor),
                np.where(
                    income <= 8_500_000,
                    np.maximum(income * 0.10 + 1_100_000, floor),
                    np.maximum(1_950_000.0, floor),
                ),
            ),
        ),
    )
    return d


# ─────────────────────────────────────────────
# 給与所得控除・給与所得の計算（公開 API）
# ─────────────────────────────────────────────
def compute_salary_deduction(income_gross: np.ndarray, year: int) -> np.ndarray:
    """
    給与収入から給与所得控除額を計算する（年度対応）。

    2026年以降は最低額が 55万 → 65万 に引き上げられる。
    """
    floor = 650_000 if year >= 2026 else 550_000
    return _calc_salary_deduction(np.asarray(income_gross, dtype=float), float(floor))


def compute_salary_income(income_gross: np.ndarray, year: int) -> np.ndarray:
    """
    給与収入から給与所得を計算する（給与収入 − 給与所得控除）。

    例:
      給与収入 350万（2025年）→ 30%×350万 + 8万 = 113万控除 → 給与所得 237万
      給与収入 350万（2026年）→ 同じ計算（最低65万は超えているため変化なし）
      給与収入 120万（2025年）→ 最低55万適用 → 給与所得 65万
      給与収入 120万（2026年）→ 最低65万適用 → 給与所得 55万（=10万円減）
    """
    gross     = np.asarray(income_gross, dtype=float)
    deduction = compute_salary_deduction(gross, year)
    return np.maximum(gross - deduction, 0.0)


def compute_pension_income(gross: np.ndarray, age: np.ndarray) -> np.ndarray:
    """
    公的年金等控除後の雑所得（雑所得_公的年金等）を計算する（令和2年分以降）。

    65歳未満: 収入 ≤ 60万 → 所得0 / 〜130万 → 収入-60万 / 〜410万 → ×0.75-27.5万 / ...
    65歳以上: 収入 ≤ 110万 → 所得0 / 〜330万 → 収入-110万 / 〜410万 → ×0.75-27.5万 / ...
    """
    gross  = np.asarray(gross, dtype=float)
    age    = np.asarray(age,   dtype=float)
    income = np.zeros_like(gross)

    # 65歳未満
    m = age < 65
    g = gross[m]
    income[m] = np.maximum(np.where(
        g <= 600_000, 0,
        np.where(g <= 1_300_000, g - 600_000,
        np.where(g <= 4_100_000, g * 0.75 - 275_000,
        np.where(g <= 7_700_000, g * 0.85 - 685_000,
        np.where(g <= 10_000_000, g * 0.95 - 1_455_000, g - 1_955_000))))), 0)

    # 65歳以上
    m = ~m
    g = gross[m]
    income[m] = np.maximum(np.where(
        g <= 1_100_000, 0,
        np.where(g <= 3_300_000, g - 1_100_000,
        np.where(g <= 4_100_000, g * 0.75 - 275_000,
        np.where(g <= 7_700_000, g * 0.85 - 685_000,
        np.where(g <= 10_000_000, g * 0.95 - 1_455_000, g - 1_955_000))))), 0)

    return income


def compute_income_tax_rate(taxable_income_resident: np.ndarray) -> np.ndarray:
    """
    国税（所得税）の限界税率を返す（復興特別所得税 2.1% 含む）。

    住民税課税所得を国税課税所得の近似値として使用する。
    （国税は基礎控除・配偶者控除等が住民税より大きいため厳密には異なるが、
      ふるさと納税控除の推計用近似として許容する）

    ふるさと納税住民税控除（確定申告ルート）の適用:
      住民税控除額 = (寄付額-2,000) × (1 − 本関数の戻り値)
      → 所得税率が高い人ほど住民税からの控除は少なくなる

    税率ブラケット（令和2年分以降）:
      〜195万円  : 5%    → 実効 5.105%
      〜330万円  : 10%   → 実効 10.21%
      〜695万円  : 20%   → 実効 20.42%
      〜900万円  : 23%   → 実効 23.483%
      〜1,800万円: 33%   → 実効 33.693%
      〜4,000万円: 40%   → 実効 40.84%
      4,000万超  : 45%   → 実効 45.945%
    """
    income = np.asarray(taxable_income_resident, dtype=float)
    rate = np.where(
        income <= 1_950_000, 0.05,
        np.where(income <= 3_300_000, 0.10,
        np.where(income <= 6_950_000, 0.20,
        np.where(income <= 9_000_000, 0.23,
        np.where(income <= 18_000_000, 0.33,
        np.where(income <= 40_000_000, 0.40,
                 0.45))))))
    return rate * 1.021  # 復興特別所得税（2.1%）含む


def estimate_furusato_resident_deduction(
    taxable_income: np.ndarray,
    donation_rate: float,
    one_stop_ratio: float,
) -> np.ndarray:
    """
    ふるさと納税による住民税からの控除額（概算）を返す。

    寄付額 = taxable_income × donation_rate で推計。

    【ワンストップ特例 (one_stop_ratio の割合)】
      住民税控除 = min(寄付額 − 2,000, 上限)
      確定申告不要の給与所得者向け。所得税控除なし、全額住民税から控除。

    【確定申告 (1 − one_stop_ratio の割合)】
      住民税控除 = min((寄付額 − 2,000) × (1 − 所得税率 × 1.021), 上限)
      所得税率が高いほど住民税控除は少なくなる（高所得者の確定申告は控除効率が下がる）。

    【上限（両者共通）】
      住民税所得割額 × 20% = taxable_income × 10% × 20%

    Args:
        taxable_income  : 住民税課税所得（配列）
        donation_rate   : 課税所得に対する平均寄付率（例: 0.015 = 1.5%）
        one_stop_ratio  : ワンストップ特例利用者の割合（0〜1）

    Returns:
        住民税からの推計控除額（円）
    """
    income    = np.asarray(taxable_income, dtype=float)
    tax_rate  = compute_income_tax_rate(income)

    donation  = income * donation_rate
    net       = np.maximum(donation - 2_000, 0.0)
    upper     = income * 0.10 * 0.20           # 住民税所得割額 × 20%

    deduct_one_stop = np.minimum(net, upper)
    deduct_final    = np.minimum(net * (1.0 - tax_rate), upper)

    deduct = one_stop_ratio * deduct_one_stop + (1.0 - one_stop_ratio) * deduct_final
    return deduct.round(0)


def compute_non_taxable_flag(
    income_total: np.ndarray,
    n_dependents: np.ndarray,
    has_spouse:   np.ndarray,
) -> np.ndarray:
    """
    住民税非課税（均等割・所得割とも非課税）に該当するかを判定する。

    非課税条件（地方税法第295条）:
      扶養なし: 合計所得 ≤ 45万円  (35万+10万)
      扶養あり: 合計所得 ≤ 35万 × (1+N) + 31万  (N = 控除対象配偶者 + 扶養親族数)

    Args:
        income_total : 合計所得金額（配列）
        n_dependents : 扶養親族数（整数配列）
        has_spouse   : 配偶者控除（deduct_spouse > 0）がある場合 1、ない場合 0
                      ※配偶者特別控除は控除対象配偶者に該当しないため含まない

    Returns:
        True なら非課税 → tax_amount = 0 とすること
    """
    n_family  = np.asarray(has_spouse, dtype=int) + np.asarray(n_dependents, dtype=int)
    threshold = NON_TAXABLE_PER_PERSON * (1 + n_family) + NON_TAXABLE_FLAT
    threshold = np.where(n_family > 0, threshold + NON_TAXABLE_FAMILY_ADD, threshold)
    return np.asarray(income_total, dtype=float) <= threshold


def compute_basic_deduction(income_total: np.ndarray) -> np.ndarray:
    """
    個人住民税の基礎控除を計算する（令和2年分以降）。

    合計所得金額（income_total）に応じたブラケット:
      ≤ 2,400万円 : 43万円
      ≤ 2,450万円 : 29万円
      ≤ 2,500万円 : 15万円
      > 2,500万円 : 0円
    """
    income = np.asarray(income_total, dtype=float)
    return np.where(
        income <= 24_000_000, 430_000,
        np.where(
            income <= 24_500_000, 290_000,
            np.where(
                income <= 25_000_000, 150_000,
                0.0,
            ),
        ),
    )


# ─────────────────────────────────────────────
# CSV 読み込み・ユーティリティ
# ─────────────────────────────────────────────
def _parse_value(v: str):
    """文字列を int → float → str の順で変換する。"""
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        pass
    return str(v)


def load_reforms(config_path: str, target_year: int, reform_type: str = None) -> list:
    """
    対象年度（target_year）以前に施行された補正ルールを CSV から読み込む。

    effective_year <= target_year かつ active=True の行が対象。
    補正関数が year 列で年次フィルタを行うため、
    一括読み込みした後に apply_reforms に渡せばよい。

    Returns:
        [{"name": str, "params": dict}, ...]  適用順リスト
    """
    df = pd.read_csv(config_path, encoding="utf-8-sig", dtype=str)
    df["active"]         = df["active"].str.strip().str.lower() == "true"
    df["effective_year"] = df["effective_year"].astype(int)

    df = df[df["active"] & (df["effective_year"] <= target_year)]

    if reform_type:
        df = df[df["reform_type"].str.strip() == reform_type]

    if df.empty:
        return []

    reforms = []
    for (name, eff_year), group in df.groupby(["reform_name", "effective_year"], sort=True):
        params = {row["param_key"]: _parse_value(row["param_value"])
                  for _, row in group.iterrows()}
        params["_effective_year"] = int(eff_year)
        params["_one_time"]       = group["one_time"].iloc[0].strip().lower() == "true"
        reforms.append({"name": name, "params": params})

    return reforms


def apply_reforms(df: pd.DataFrame, reforms: list) -> pd.DataFrame:
    """
    load_reforms の返り値を順番に適用する。

    Returns:
        補正済み DataFrame（元 df は変更しない）
    """
    for r in reforms:
        name = r["name"]
        if name in REFORM_REGISTRY:
            df = REFORM_REGISTRY[name](df, r["params"])
        else:
            print(f"  ⚠ 未実装の改正: {name}（REFORM_REGISTRY に追加してください）")
    return df


def print_reform_summary(reforms: list, label: str = ""):
    """適用する補正の一覧を表示する。"""
    tag = f"[{label}] " if label else ""
    if not reforms:
        print(f"  {tag}適用する補正なし")
        return
    print(f"  {tag}適用補正: {len(reforms)}件")
    for r in reforms:
        eff  = r["params"]["_effective_year"]
        once = "単年" if r["params"]["_one_time"] else "継続"
        print(f"    - {r['name']} (施行{eff}年・{once})")


# ─────────────────────────────────────────────
# 各補正関数の実装
# ─────────────────────────────────────────────
def _apply_teigaku_reduction(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    定額減税（単年）: 対象年の tax_amount に本人分 1万円を加算して学習ラベルを補正。

    2024年は「本人 + 扶養親族1人あたり住民税1万円」の定額減税が適用され
    tax_amount が恒久的でない水準に低下している。加算補正により
    モデルが異常値を学習するバイアスを防ぐ。
    扶養分（1人1万円）は個人データから正確に算出困難なため本人分のみ補正。

    【現在の運用方針】
    tax_reform_config.csv で active=False に設定済み（この関数は呼ばれない）。
    実データ投入時に 2024年の税額を「定額減税前の水準 (+1万円)」に加工することで
    モデルパイプライン外で対応する方針に変更。
    ダミーデータには定額減税効果が未実装のため、active=True にすると 2024年ラベルが
    架空に膨らみ全年度で系統的な過大予測が生じる（実験で確認済み）。
    """
    amount   = int(params["amount_per_person"])
    eff_year = int(params["_effective_year"])

    if "年度" not in df.columns:
        return df

    mask = df["年度"] == eff_year
    n    = int(mask.sum())
    if n == 0:
        return df

    df = df.copy()
    df.loc[mask, TARGET_COL] = df.loc[mask, TARGET_COL] + amount
    print(f"    定額減税補正 ({eff_year}年): {n:,}人 × +{amount/1e4:.0f}万円 = +{amount*n/1e8:.2f}億円")
    return df


def _apply_salary_deduction_floor(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    給与所得控除最低額引き上げ（2026年〜）: 55万円 → 65万円。

    income_salary_gross（給与収入）が存在する場合はそれを基に正確な控除差分を算出し、
    income_salary（給与所得）・income_total・taxable_income を更新する。

    income_salary_gross が存在しない場合（旧形式データ）は taxable_income を直接削減。

    影響範囲: 給与収入 〜約190万円（第1区分全体 + 第2区分の低所得側）。
    190万超 : ブラケット計算値が既に65万超 → 変化なし。

    【新スキーマでの用途】
    estimate_next_year() は target_year の式を使って income_salary を既に計算するため
    この関数は「補正前後の税額比較を明示したい場合」に使う。
    """
    old_floor = float(params["old_floor"])  # 550_000
    new_floor = float(params["new_floor"])  # 650_000

    # 給与収入（給与収入）があればそれを使用、なければ 給与所得 を収入とみなす
    if "給与収入" in df.columns:
        gross   = df["給与収入"].values
        has_sal = df.get("給与所得有無", (df["給与収入"] > 0).astype(int))
    else:
        gross   = df["給与所得"].values
        has_sal = df.get("給与所得有無", (df["給与所得"] > 0).astype(int))

    old_deduction = _calc_salary_deduction(gross, old_floor)
    new_deduction = _calc_salary_deduction(gross, new_floor)

    deduction_increase = pd.Series(
        np.maximum(new_deduction - old_deduction, 0.0), index=df.index
    )

    mask = (has_sal == 1) & (deduction_increase > 0)
    n    = int(mask.sum())

    if n == 0:
        print("    給与所得控除引き上げ: 対象者なし")
        return df

    df = df.copy()

    if "給与収入" in df.columns:
        # 新スキーマ: 給与所得（所得）を削減 → 総所得金額等 を再計算
        df.loc[mask, "給与所得"] = (
            (df.loc[mask, "給与所得"] - deduction_increase.loc[mask])
            .clip(lower=0).round(0).astype(df["給与所得"].dtype)
        )
        from config import ALL_INCOME_COLS
        inc_cols = [c for c in ALL_INCOME_COLS if c in df.columns]
        df["総所得金額等"] = df[inc_cols].sum(axis=1).clip(lower=0)

    # 課税標準額 を削減
    df.loc[mask, "課税標準額"] = (
        (df.loc[mask, "課税標準額"] - deduction_increase.loc[mask])
        .clip(lower=0).round(0).astype(df["課税標準額"].dtype)
    )

    if "差引所得控除合計" in df.columns:
        df.loc[mask, "差引所得控除合計"] = (
            (df.loc[mask, "差引所得控除合計"] + deduction_increase.loc[mask])
            .round(0).astype(df["差引所得控除合計"].dtype)
        )
    if "所得控除率" in df.columns and "総所得金額等" in df.columns:
        df["所得控除率"] = (
            df["差引所得控除合計"] / df["総所得金額等"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)
    if "課税標準率" in df.columns and "総所得金額等" in df.columns:
        df["課税標準率"] = (
            df["課税標準額"] / df["総所得金額等"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

    max_inc  = deduction_increase[mask].max()
    total_red = (deduction_increase[mask] * 0.10).sum() / 1e8
    median_gross = (df.loc[mask, "給与収入"].median() if "給与収入" in df.columns
                    else df.loc[mask, "給与所得"].median()) / 1e4
    print(f"    給与所得控除引き上げ: {n:,}人対象 / 対象者給与収入中央値 {median_gross:.0f}万円 / "
          f"最大控除増 {max_inc/1e4:.1f}万円 / 税額減少概算 -{total_red:.2f}億円")
    return df


def _apply_dependent_income_limit(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    扶養要件引き上げ（2026年〜）: 扶養可能所得上限 48万→58万円。

    個人データに家族関係情報がないため個人レベル補正は不可。
    06_trend_correction.py の dependent_income_limit で集計レベル補正を実施。
    """
    print(f"    扶養要件引き上げ ({params['_effective_year']}年〜): スキップ → 43 で集計補正")
    return df


def _apply_special_dependent_allowance(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    特定親族特別控除（2026年〜）: 19-22歳扶養親族への段階的控除。

    個人データに扶養親族の年齢情報がないため個人レベル補正は不可。
    06_trend_correction.py の special_dependent_allowance で集計レベル補正を実施。
    """
    age_from = int(params.get("age_from", 19))
    age_to   = int(params.get("age_to",   22))
    print(f"    特定親族特別控除 ({age_from}-{age_to}歳, {params['_effective_year']}年〜): スキップ → 43 で集計補正")
    return df


# ─────────────────────────────────────────────
# 補正レジストリ（名前 → 関数）
# ─────────────────────────────────────────────
REFORM_REGISTRY: dict = {
    "定額減税額"          : _apply_teigaku_reduction,
    "salary_deduction_floor"     : _apply_salary_deduction_floor,
    "dependent_income_limit"     : _apply_dependent_income_limit,
    "special_dependent_allowance": _apply_special_dependent_allowance,
}
