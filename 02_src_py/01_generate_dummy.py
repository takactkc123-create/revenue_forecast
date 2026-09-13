"""
01_generate_dummy.py
====================
個人住民税予測モデル - Step1: ダミーデータ生成

確定申告書第1表の項目を基準に、現実に近い個人住民税レコードを生成する。
実データがある場合は 00_CsvCreate_SQL の手順でcsvを作成し、03_feature_eng.py から開始する。

【実行方法】
  python 01_generate_dummy.py
  python 01_generate_dummy.py --n 50000    # 1年あたり5万件に変更したい場合
  
【import】
  config.py（人口統計・所得分布・控除上限等のパラメータ）
  tax_reform.py（所得控除・課税所得計算関数）

【export】
  data/01out_individual_raw.csv  ← 03_feature_eng.py にて import
"""
import argparse
import os
import numpy as np
import pandas as pd

from tax_reform import (
    compute_salary_income,
    compute_pension_income,
    compute_basic_deduction,
    estimate_furusato_resident_deduction,
    compute_non_taxable_flag,
)
from config import (
    RANDOM_SEED, N_PER_YEAR, TURNOVER_RATE, POPULATION_GROWTH_RATES, GENDER_RATIO,
    AGE_GROUPS, AGE_WEIGHTS, AGE_RANGE_BY_GROUP,
    HOUSING_PARAMS, HOUSING_PROB_BY_AGE,
    FURUSATO_PARAMS, RAW_DATA_PATH,
    TRAIN_YEARS, TEST_YEAR, ALL_INCOME_COLS,
    APPLY_WAGE_GROWTH, SALARY_GROWTH_RATES, PENSION_GROWTH_RATES,
)

OUTPUT_PATH = RAW_DATA_PATH

# ─── 住民税の所得控除上限（所得税より低い） ──────────────────────────────────
_LIFE_INS_UPPER       = 70_000   # 生命保険料控除（住民税）
_EARTHQUAKE_INS_UPPER = 25_000   # 地震保険料控除（住民税）
_DISABILITY_AMOUNT    = 260_000  # 障害者控除（普通）
_SPECIAL_DISABILITY   = 300_000  # 障害者控除（特別）
_WIDOW_AMOUNT         = 260_000  # 寡婦控除
_SINGLE_PARENT_AMOUNT = 300_000  # ひとり親控除


def _generate_ages(rng: np.random.Generator, n: int) -> np.ndarray:
    """AGE_GROUPS/AGE_WEIGHTS の分布に従って実年齢（整数）を生成する。"""
    group_idx = rng.choice(len(AGE_GROUPS), size=n, p=AGE_WEIGHTS)
    ages = np.zeros(n, dtype=int)
    for gi, g in enumerate(AGE_GROUPS):
        lo, hi = AGE_RANGE_BY_GROUP[g]
        mask = group_idx == gi
        if mask.any():
            ages[mask] = rng.integers(lo, hi + 1, size=mask.sum())
    return ages


def _age_to_group(ages: np.ndarray) -> np.ndarray:
    """実年齢の配列から年齢区分ラベル（'20-24' 等）の配列を返す。"""
    ages = np.asarray(ages)
    # 2026-09-12変更: 旧「80以上」を 80-84 / 85-89 / 90over の3区分に分割
    conditions = [
        ages < 25, ages < 30, ages < 35, ages < 40, ages < 45,
        ages < 50, ages < 55, ages < 60, ages < 65, ages < 70,
        ages < 75, ages < 80, ages < 85, ages < 90,
    ]
    choices = [
        "20-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69",
        "70-74", "75-79", "80-84", "85-89",
    ]
    return np.select(conditions, choices, default="90over")


def _spouse_special_deduction(income_total: np.ndarray, has_spouse_special: np.ndarray) -> np.ndarray:
    """
    配偶者特別控除（住民税）を段階計算する。
    配偶者の合計所得 48万〜133万円の場合に適用。
    ここではダミーとして has_spouse_special=1 の人に一律 33万円を付与する。
    実データでは配偶者所得に応じた段階額を使用すること。
    """
    return np.where(has_spouse_special, 330_000, 0).astype(int)


def _gen_rare(rng, n, prob, mean, std):
    """低確率で発生するベクトル化所得生成の共通処理。非負に丸める。"""
    return np.where(
        rng.random(n) < prob,
        np.abs(rng.normal(mean, std, n)).round(0),
        0,
    ).astype(int)


def _cumulative_factor(rates: dict, year: int) -> float:
    """基準年(rates辞書の最小年)から year までの累積成長率を返す。
    APPLY_WAGE_GROWTH=False のときは常に 1.0。"""
    if not APPLY_WAGE_GROWTH:
        return 1.0
    base = min(rates.keys())
    f = 1.0
    for y in range(base + 1, year + 1):
        f *= rates.get(y, 1.0)
    return f


def _gen_non_salary_incomes(rng, n, age_arr):
    """給与以外の所得を生成して dict で返す（初年度・新規加入者で共通。分布・確率は従来の初年度と同じ）。

    2026-09-13 関数化: 従来は初年度のみこの分布で生成し、新規加入者は全て0円固定だった。
    """
    inc = {}
    # 事業・不動産・配当・雑所得（年齢非依存、国税庁R4参考）
    # 2026-09-13修正: rng.normal に size を指定。未指定だと乱数が1つしか引かれず、保有者全員が同額になっていた
    inc["income_business"] = np.where(rng.random(n) < 0.12, rng.normal(4_728_000, 3_000_000, n), 0.0)
    inc["income_property"] = np.where(rng.random(n) < 0.06, rng.normal(5_425_000, 3_500_000, n), 0.0)
    # 事業所得・不動産所得は正規分布のためマイナス（赤字＝損失）の人も生成されるが、損失を表すものとして許容する（2026-09-13 判断）
    inc["income_dividend"] = _gen_rare(rng, n, 0.10, 200_000, 250_000).astype(float)
    inc["income_other"]    = _gen_rare(rng, n, 0.05, 100_000, 200_000).astype(float)

    # ── 公的年金等収入（厚生労働省R5概況: 国民年金平均＋厚生年金平均の合計。1万円単位）
    # 保険会社の集計結果より設定
    _pension_mu = np.select(
        [(age_arr >= 60) & (age_arr < 65),
         (age_arr >= 65) & (age_arr < 70),
         (age_arr >= 70) & (age_arr < 75),
         (age_arr >= 75) & (age_arr < 80),
         age_arr >= 80],
        [1_450_000, 2_480_000, 2_440_000, 2_470_000, 2_590_000],
        default=0,
    ).astype(float)
    inc["income_pension_gross"] = np.where(
        age_arr < 60, 0.0,
        np.maximum(0.0, rng.normal(_pension_mu, np.maximum(_pension_mu * 0.30, 1.0))),
    )
    # 年金所得を収入から計算（65歳境界で控除額の計算式が変わる）
    inc["income_pension"] = compute_pension_income(inc["income_pension_gross"], age_arr).round(0).astype(int)

    # 農業・利子・業務雑所得・総合譲渡・一時所得（低頻度。ベクトル化生成）
    inc["income_farming"]       = _gen_rare(rng, n, 0.015, 500_000,   400_000)
    inc["income_interest"]      = _gen_rare(rng, n, 0.060, 30_000,    50_000)
    inc["income_misc_business"] = _gen_rare(rng, n, 0.040, 200_000,   300_000)
    inc["income_stcg"]          = _gen_rare(rng, n, 0.010, 500_000,   800_000)
    inc["income_ltcg"]          = _gen_rare(rng, n, 0.020, 800_000,   1_000_000)
    inc["income_occasional"]    = _gen_rare(rng, n, 0.010, 200_000,   300_000)

    # 分離課税所得（繰越控除後のため非負。高所得層中心で低確率）
    inc["sep_stcg_general"]    = _gen_rare(rng, n, 0.003, 1_000_000, 2_000_000)
    inc["sep_stcg_reduced"]    = _gen_rare(rng, n, 0.002, 800_000,   1_500_000)
    inc["sep_ltcg_general"]    = _gen_rare(rng, n, 0.015, 2_000_000, 3_000_000)
    inc["sep_ltcg_specific"]   = _gen_rare(rng, n, 0.008, 3_000_000, 5_000_000)
    inc["sep_ltcg_reduced"]    = _gen_rare(rng, n, 0.003, 1_500_000, 2_000_000)
    inc["sep_stock_general"]   = _gen_rare(rng, n, 0.015, 500_000,   800_000)
    inc["sep_stock_listed"]    = _gen_rare(rng, n, 0.060, 800_000,   1_500_000)
    inc["sep_dividend_listed"] = _gen_rare(rng, n, 0.040, 200_000,   400_000)
    inc["sep_futures"]         = _gen_rare(rng, n, 0.003, 300_000,   500_000)
    inc["sep_forestry"]        = _gen_rare(rng, n, 0.001, 2_000_000, 3_000_000)
    return inc


def generate_dummy(n_per_year: int = N_PER_YEAR, years: list = None) -> pd.DataFrame:
    """実データに近い個人住民税レコードを確定申告書第1表の列構成で生成する。"""
    if years is None:
        years = TRAIN_YEARS + [TEST_YEAR]

    rng = np.random.default_rng(RANDOM_SEED)
    all_dfs = []

    # 実年齢（整数）を生成し、年齢区分ラベルを派生させる
    age_arr       = _generate_ages(rng, n_per_year)
    age_group_arr = _age_to_group(age_arr)
    gender        = rng.choice([0, 1], size=n_per_year, p=GENDER_RATIO)  # 0=男性, 1=女性

    # ── 収入の生成（全ベクトル化）────────────────────────────────────────────────

    # 給与収入（国税庁R6 第14図: 年齢階層別・男女別平均給与、万円→円）
    _age_sal_conds = [
        age_arr < 25,
        (age_arr >= 25) & (age_arr < 30),
        (age_arr >= 30) & (age_arr < 35),
        (age_arr >= 35) & (age_arr < 40),
        (age_arr >= 40) & (age_arr < 45),
        (age_arr >= 45) & (age_arr < 50),
        (age_arr >= 50) & (age_arr < 55),
        (age_arr >= 55) & (age_arr < 60),
        (age_arr >= 60) & (age_arr < 65),
        (age_arr >= 65) & (age_arr < 70),
        age_arr >= 70,
    ]
    _sal_mu = np.where(
        gender == 0,
        np.select(_age_sal_conds, [295, 438, 512, 574, 630, 663, 709, 735, 604, 472, 587], default=438) * 10_000,
        np.select(_age_sal_conds, [258, 370, 362, 351, 359, 369, 363, 356, 294, 240, 209], default=370) * 10_000,
    ).astype(float)
    
    # 年齢割合は国税庁：民間給与の実態調査結果の「年齢階層別の給与所得者数」/　年齢階層別総人口　で分布
    _sal_prob = np.select(
        [age_arr < 25, age_arr < 60, age_arr < 65, age_arr < 75],
        [0.40, 0.70, 0.60, 0.36],
        default=0.20,
    )
    income_salary_gross = np.where(
        rng.random(n_per_year) < _sal_prob,
        np.maximum(0, rng.normal(_sal_mu, _sal_mu * 0.40)),
        0,
    )

    # 給与以外の所得（2026-09-13: 新規加入者と共通の関数で生成。分布・確率は従来どおり）
    _inc = _gen_non_salary_incomes(rng, n_per_year, age_arr)
    income_business      = _inc["income_business"]
    income_farming       = _inc["income_farming"]
    income_property      = _inc["income_property"]
    income_interest      = _inc["income_interest"]
    income_dividend      = _inc["income_dividend"]
    income_pension       = _inc["income_pension"]
    income_misc_business = _inc["income_misc_business"]
    income_other         = _inc["income_other"]
    income_stcg          = _inc["income_stcg"]
    income_ltcg          = _inc["income_ltcg"]
    income_occasional    = _inc["income_occasional"]
    sep_stcg_general     = _inc["sep_stcg_general"]
    sep_stcg_reduced     = _inc["sep_stcg_reduced"]
    sep_ltcg_general     = _inc["sep_ltcg_general"]
    sep_ltcg_specific    = _inc["sep_ltcg_specific"]
    sep_ltcg_reduced     = _inc["sep_ltcg_reduced"]
    sep_stock_general    = _inc["sep_stock_general"]
    sep_stock_listed     = _inc["sep_stock_listed"]
    sep_dividend_listed  = _inc["sep_dividend_listed"]
    sep_futures          = _inc["sep_futures"]
    sep_forestry         = _inc["sep_forestry"]
    income_pension_gross = _inc["income_pension_gross"]

    # 給与所得を収入から計算
    income_salary  = compute_salary_income(income_salary_gross, year=years[0])

    n = n_per_year

    income_total = (
        income_salary + income_business + income_farming + income_property
        + income_interest + income_dividend + income_pension + income_misc_business
        + income_other + income_stcg + income_ltcg + income_occasional
        + sep_stcg_general + sep_stcg_reduced + sep_ltcg_general + sep_ltcg_specific
        + sep_ltcg_reduced + sep_stock_general + sep_stock_listed + sep_dividend_listed
        + sep_futures + sep_forestry
    )

    # ── 所得控除の生成 ──────────────────────────────────────────────────────────
    has_spouse         = rng.random(n) < 0.20
    has_spouse_special = (~has_spouse) & (rng.random(n) < 0.04)
    n_dependent        = rng.integers(0, 4, size=n)

    deduct_spouse         = np.where(has_spouse, 330_000, 0).astype(int)
    deduct_spouse_special = _spouse_special_deduction(income_total, has_spouse_special)
    deduct_dependent      = (n_dependent * 330_000).astype(int)

    # 社会保険料控除: 給与収入の17%（実務近似）
    # 財務省国民負担率（2026年度）より設定
    deduct_social_ins = (income_salary_gross * 0.176).round(0).astype(int)

    # 小規模企業共済等掛金控除（国税庁R4標本調査: 申告者の13%が適用）
    deduct_small_biz_ins = np.where(
        rng.random(n) < 0.13,
        rng.integers(100_000, 840_001, size=n), 0,
    ).astype(int)

    # 生命保険料控除（国税庁R4標本調査: 申告者の79%が適用）
    deduct_life_ins = np.where(
        rng.random(n) < 0.79,
        rng.integers(10_000, _LIFE_INS_UPPER + 1, size=n), 0,
    ).astype(int)

    # 地震保険料控除（国税庁R4標本調査: 申告者の38%が適用）
    deduct_earthquake = np.where(
        rng.random(n) < 0.38,
        rng.integers(5_000, _EARTHQUAKE_INS_UPPER + 1, size=n), 0,
    ).astype(int)

    # 雑損控除（災害・盗難等。非常に稀）
    deduct_casualty = np.where(
        rng.random(n) < 0.001,
        rng.integers(100_000, 2_000_001, size=n), 0,
    ).astype(int)

    # 医療費控除（家族分も申告可能。厚生労働省: 年齢別医療費支出割合 × 全体適用率0.29（国税庁）より逆算）
    # 支出割合 44以下:17.9% / 45-64:22.0% / 65以上:60.1% を人口ウェイトで全体0.29に整合
    medical_probs = np.select(
        [age_arr < 45, age_arr < 65],
        [0.15, 0.19],
        default=0.52, # 65歳以上
    )
    deduct_medical = np.where(
        rng.random(n) < medical_probs,
        rng.integers(50_000, 500_000, size=n), 0,
    ).astype(int)

    # 障害者控除
    deduct_disability = np.where(rng.random(n) < 0.05, _DISABILITY_AMOUNT, 0).astype(int)

    # 寡婦・ひとり親控除（合計0.04）
    deduct_widow = np.where(rng.random(n) < 0.04, _WIDOW_AMOUNT, 0).astype(int)

    # 勤労学生控除（25歳未満の学生。26万円固定）
    deduct_working_student = np.where(
        (age_arr < 25) & (rng.random(n) < 0.05), 260_000, 0
    ).astype(int)

    # 寄附金控除（所得控除分）: 確定申告経由のふるさと納税等。ワンストップ特例者は対象外
    # 全ドナー率 = 確定申告率(NTA R4=0.20) / (1 - one_stop_ratio) ≈ 0.47
    # 確定申告分 = 全ドナー率 × (1 - one_stop_ratio) → 0.20 に戻る（NTA観測値と一致）
    _one_stop = FURUSATO_PARAMS["one_stop_ratio"]
    _donation_rate = 0.20 / (1 - _one_stop) * (1 - _one_stop)  # = 0.20
    deduct_donation_income = np.where(
        rng.random(n) < _donation_rate,
        rng.integers(10_000, 200_001, size=n), 0,
    ).astype(int)

    # 基礎控除
    deduct_basic = compute_basic_deduction(income_total).round(0).astype(int)

    deduct_total = (
        deduct_social_ins + deduct_small_biz_ins + deduct_life_ins + deduct_earthquake
        + deduct_casualty + deduct_medical + deduct_disability + deduct_widow
        + deduct_spouse + deduct_spouse_special + deduct_dependent + deduct_basic
        + deduct_working_student + deduct_donation_income
    )
    taxable_inc = (income_total - deduct_total).clip(0)

    # ── 税額控除の生成 ──────────────────────────────────────────────────────────
    housing_probs      = np.array([HOUSING_PROB_BY_AGE.get(ag, 0.0) for ag in age_group_arr])
    deduct_tax_housing = np.where(
        rng.random(n) < housing_probs,
        rng.integers(50_000, HOUSING_PARAMS["upper_limit"] + 1, size=n), 0,
    ).astype(int)

    deduct_tax_furusato = estimate_furusato_resident_deduction(
        taxable_inc,
        donation_rate=FURUSATO_PARAMS["donation_rate"],
        one_stop_ratio=FURUSATO_PARAMS["one_stop_ratio"],
    ).astype(int)

    # 調整控除（住民税・所得税の人的控除差を補正。簡略化: 課税者一律2,500円）
    taxcredit_adjustment = np.where(taxable_inc > 0, 2_500, 0).astype(int)

    # 配当控除（総合課税の配当所得に対して2.8%）
    taxcredit_dividend = (income_dividend * 0.028).round(0).astype(int)

    # 外国税額控除（ダミーでは0）
    taxcredit_foreign = np.zeros(n, dtype=int)

    # 配当割額・株式等譲渡所得割額の控除（源泉徴収済み5%相当）
    taxcredit_dividend_split = ((sep_stock_listed + sep_dividend_listed) * 0.05).round(0).astype(int)

    # 税額 = max(課税所得×10% − 税額控除合計, 0) + 均等割（非課税者は0）
    tax_amount = (
        np.maximum(
            taxable_inc * 0.10
            - deduct_tax_housing 
            - deduct_tax_furusato
            - taxcredit_adjustment 
            - taxcredit_dividend 
            - taxcredit_dividend_split,
            0,
        ) + 5_300
    ).round(0).astype(int)
    is_non_taxable = compute_non_taxable_flag(
        income_total, n_dependent, has_spouse.astype(int)
    )
    tax_amount = np.where(is_non_taxable, 0, tax_amount)

    # 年税額を市町村・都道府県の均等割・所得割に分解（EDA用。03でdrop）
    # 市町村: 均等割3,500円 + 所得割6%、都道府県: 均等割1,800円 + 所得割4% ※1,800円については大阪府で設定
    _itp = np.maximum(tax_amount - 5_300, 0).astype(float)
    _nz  = tax_amount > 0

    collection = rng.choice([1, 2], size=n, p=[0.80, 0.20])
    person_ids = np.arange(1, n + 1)

    base_df = pd.DataFrame({
        "仮ID"               : person_ids,
        "年度"                    : years[0],
        "性別"                  : gender,
        "年齢"                     : age_arr,
        "年齢区分"               : age_group_arr,
        # 収入
        "給与収入"     : income_salary_gross.round(0).astype(int),
        "雑収入_公的年金等"    : income_pension_gross.round(0).astype(int),
        # 所得
        "給与所得"           : income_salary.round(0).astype(int),
        "事業所得_営業等"         : income_business.round(0).astype(int),
        "事業所得_農業"          : income_farming.round(0).astype(int),
        "不動産所得"         : income_property.round(0).astype(int),
        "利子所得"         : income_interest.round(0).astype(int),
        "配当所得"         : income_dividend.round(0).astype(int),
        "雑所得_公的年金等"          : income_pension.round(0).astype(int),
        "雑所得_業務"    : income_misc_business.round(0).astype(int),
        "雑所得_その他"            : income_other.round(0).astype(int),
        "総合短期譲渡所得"             : income_stcg.round(0).astype(int),
        "総合長期譲渡所得"             : income_ltcg.round(0).astype(int),
        "一時所得"       : income_occasional.round(0).astype(int),
        # 分離課税所得
        "分離_短期譲渡一般"        : sep_stcg_general.round(0).astype(int),
        "分離_短期譲渡軽減"        : sep_stcg_reduced.round(0).astype(int),
        "分離_長期譲渡一般"        : sep_ltcg_general.round(0).astype(int),
        "分離_長期譲渡特定"       : sep_ltcg_specific.round(0).astype(int),
        "分離_長期譲渡軽減"        : sep_ltcg_reduced.round(0).astype(int),
        "分離_一般株式譲渡"       : sep_stock_general.round(0).astype(int),
        "分離_特定株式譲渡"        : sep_stock_listed.round(0).astype(int),
        "分離_上場株式配当"     : sep_dividend_listed.round(0).astype(int),
        "分離_先物取引"             : sep_futures.round(0).astype(int),
        "分離_山林"            : sep_forestry.round(0).astype(int),
        # 所得控除
        "社会保険料控除"       : deduct_social_ins.round(0).astype(int),
        "小規模企業共済等掛金控除"    : deduct_small_biz_ins.round(0).astype(int),
        "生命保険料控除"         : deduct_life_ins.round(0).astype(int),
        "地震保険料控除"   : deduct_earthquake.round(0).astype(int),
        "雑損控除"         : deduct_casualty.round(0).astype(int),
        "医療費控除"          : deduct_medical.round(0).astype(int),
        "障害者控除"       : deduct_disability.round(0).astype(int),
        "寡婦控除"            : deduct_widow.round(0).astype(int),
        "配偶者控除"           : deduct_spouse.round(0).astype(int),
        "配偶者特別控除"   : deduct_spouse_special.round(0).astype(int),
        "扶養控除"        : deduct_dependent.round(0).astype(int),
        "基礎控除"            : deduct_basic.round(0).astype(int),
        "勤労学生控除"  : deduct_working_student.round(0).astype(int),
        "寄附金控除"  : deduct_donation_income.round(0).astype(int),
        "扶養人数"             : n_dependent,
        # 課税所得
        "課税標準額"          : taxable_inc.round(0).astype(int),
        # 税額控除
        "住宅借入金特別控除"      : deduct_tax_housing.round(0).astype(int),
        "寄附金税額控除"     : deduct_tax_furusato.round(0).astype(int),
        "調整控除"    : taxcredit_adjustment.round(0).astype(int),
        "配当控除"      : taxcredit_dividend.round(0).astype(int),
        "外国税額控除"       : taxcredit_foreign.round(0).astype(int),
        "配当割額・株式等譲渡所得割額の控除": taxcredit_dividend_split.round(0).astype(int),
        # 税額
        "年税額"              : tax_amount.round(0).astype(int),
        "徴収区分"         : collection,
        # EDA用内訳（モデルでは03でdrop）
        "市町村_均等割"          : np.where(_nz, 3_500, 0).astype(int),
        "市町村_所得割"           : (_itp * 0.6).round(0).astype(int),
        "都道府県_均等割"          : np.where(_nz, 1_800, 0).astype(int),
        "都道府県_所得割"           : (_itp * 0.4).round(0).astype(int),
        "控除不足額"               : 0,
    })
    all_dfs.append(base_df)

    current_df = base_df.copy()

    for yr in years[1:]:
        n_leave   = int(n_per_year * TURNOVER_RATE)
        leave_idx = rng.choice(current_df.index, size=n_leave, replace=False)
        stay_df   = current_df.drop(leave_idx).copy()
        stay_df["年度"]      = yr
        stay_df["年齢"]       = stay_df["年齢"] + 1
        stay_df["年齢区分"] = _age_to_group(stay_df["年齢"].values)

        # 前年比成長率: APPLY_WAGE_GROWTH=True なら年度別、False なら旧挙動（+0.5%固定）
        sal_trend   = SALARY_GROWTH_RATES.get(yr, 1.005) if APPLY_WAGE_GROWTH else 1.005
        pen_trend   = PENSION_GROWTH_RATES.get(yr, 1.005) if APPLY_WAGE_GROWTH else 1.005
        other_trend = 1.005
        noise = rng.normal(1.0, 0.03, size=len(stay_df))

        # 給与収入に年度別賃上げ率を適用し、給与所得を再算出
        stay_df["給与収入"] = (
            stay_df["給与収入"] * sal_trend * noise
        ).clip(0).round(0).astype(int)
        stay_df["給与所得"] = (
            compute_salary_income(stay_df["給与収入"].values, year=yr)
        ).round(0).astype(int)

        # 公的年金収入に年度別改定率を適用（農業・事業・不動産は旧トレンド固定）
        stay_df["雑収入_公的年金等"] = (
            stay_df["雑収入_公的年金等"] * pen_trend * noise
        ).clip(0).round(0).astype(int)
        for col in ["事業所得_農業"]:
            stay_df[col] = (stay_df[col] * other_trend * noise).clip(0).round(0).astype(int)
        for col in ["事業所得_営業等", "不動産所得"]:
            stay_df[col] = (stay_df[col] * other_trend * noise).round(0).astype(int)

        # 年金所得を収入から再計算（年齢に応じた控除を適用）
        stay_df["雑所得_公的年金等"] = compute_pension_income(
            stay_df["雑収入_公的年金等"].values, stay_df["年齢"].values
        ).round(0).astype(int)

        # 扶養変動（8%が毎年変化）
        change_dep = rng.random(len(stay_df)) < 0.08
        stay_df.loc[change_dep, "扶養控除"] = (
            rng.integers(0, 4, size=change_dep.sum()) * 330_000
        )
        stay_df.loc[change_dep, "扶養人数"] = (
            stay_df.loc[change_dep, "扶養控除"] // 330_000
        )

        # 合計所得を全所得列で再計算
        inc_cols = [c for c in ALL_INCOME_COLS if c in stay_df.columns]
        total_inc_s = stay_df[inc_cols].sum(axis=1)

        stay_df["社会保険料控除"] = (
            stay_df["給与収入"] * 0.14
        ).round(0).astype(int)
        stay_df["基礎控除"] = (
            compute_basic_deduction(total_inc_s.values)
        ).round(0).astype(int)

        total_dec_s = (
            stay_df["社会保険料控除"] + stay_df["小規模企業共済等掛金控除"]
            + stay_df["生命保険料控除"] + stay_df["地震保険料控除"]
            + stay_df["雑損控除"] + stay_df["医療費控除"]
            + stay_df["障害者控除"] + stay_df["寡婦控除"]
            + stay_df["配偶者控除"] + stay_df["配偶者特別控除"]
            + stay_df["扶養控除"] + stay_df["基礎控除"]
            + stay_df["勤労学生控除"] + stay_df["寄附金控除"]
        )
        stay_df["課税標準額"] = (total_inc_s - total_dec_s).clip(0).round(0).astype(int)

        # 住宅ローン控除: 年間消滅率で一部が0に
        keep_housing = rng.random(len(stay_df)) >= HOUSING_PARAMS["annual_exit_rate"]
        stay_df["住宅借入金特別控除"] = np.where(
            keep_housing, stay_df["住宅借入金特別控除"].values, 0
        ).astype(int)

        stay_df["寄附金税額控除"] = estimate_furusato_resident_deduction(
            stay_df["課税標準額"].values,
            donation_rate=FURUSATO_PARAMS["donation_rate"],
            one_stop_ratio=FURUSATO_PARAMS["one_stop_ratio"],
        ).astype(int)

        # 税額控除を更新
        stay_taxable = stay_df["課税標準額"].values
        stay_df["調整控除"]    = np.where(stay_taxable > 0, 2_500, 0).astype(int)
        stay_df["配当控除"]      = (stay_df["配当所得"].values * 0.028).round(0).astype(int)
        stay_df["配当割額・株式等譲渡所得割額の控除"] = (
            (stay_df["分離_特定株式譲渡"].values + stay_df["分離_上場株式配当"].values) * 0.05
        ).round(0).astype(int)

        stay_tax = (
            np.maximum(
                stay_taxable * 0.10
                - stay_df["住宅借入金特別控除"].values
                - stay_df["寄附金税額控除"].values
                - stay_df["調整控除"].values
                - stay_df["配当控除"].values
                - stay_df["配当割額・株式等譲渡所得割額の控除"].values,
                0,
            ) + 5_300
        ).round(0).astype(int)
        stay_is_non_taxable = compute_non_taxable_flag(
            total_inc_s.values,
            stay_df["扶養人数"].values,
            (stay_df["配偶者控除"].values > 0).astype(int),
        )
        stay_df["年税額"] = np.where(stay_is_non_taxable, 0, stay_tax).astype(int)

        # 均等割・所得割を更新（EDA用）
        _s_itp = np.maximum(stay_df["年税額"].values - 5_300, 0).astype(float)
        _s_nz  = stay_df["年税額"].values > 0
        stay_df["市町村_均等割"] = np.where(_s_nz, 3_500, 0).astype(int)
        stay_df["市町村_所得割"]  = (_s_itp * 0.6).round(0).astype(int)
        stay_df["都道府県_均等割"] = np.where(_s_nz, 1_800, 0).astype(int)
        stay_df["都道府県_所得割"]  = (_s_itp * 0.4).round(0).astype(int)
        stay_df["控除不足額"]      = 0

        # 人口成長率から目標人数を計算し、流入者数を調整
        target_n  = round(n_per_year * POPULATION_GROWTH_RATES.get(yr, 1.0))
        n_new     = max(0, target_n - len(stay_df))

        # ── 新規流入者（給与は独自水準。給与以外の所得は2026-09-13から初年度と同じ分布で生成）──
        new_ids        = np.arange(current_df["仮ID"].max() + 1,
                                   current_df["仮ID"].max() + n_new + 1)
        new_age_arr    = _generate_ages(rng, n_new)
        new_age_groups = _age_to_group(new_age_arr)
        # 新規流入者の給与は基準年水準に累積成長率を乗じて当該年度の水準に調整
        _sal_base = 3_200_000 * _cumulative_factor(SALARY_GROWTH_RATES, yr)
        _sal_std  = 1_300_000 * _cumulative_factor(SALARY_GROWTH_RATES, yr)
        new_sal_gross  = np.maximum(0, rng.normal(_sal_base, _sal_std, n_new))
        new_sal_net    = compute_salary_income(new_sal_gross, year=yr)
        # 2026-09-13変更: 新規加入者にも給与以外の所得を初年度と同じ確率・分布で持たせる
        # （従来は全て0円固定だったため、年を追うごとに事業所得・不動産所得等の保有者が減っていた）
        new_inc   = _gen_non_salary_incomes(rng, n_new, new_age_arr)
        new_total = (
            new_sal_net
            + new_inc["income_business"]
            + new_inc["income_farming"]
            + new_inc["income_property"]
            + new_inc["income_interest"]
            + new_inc["income_dividend"]
            + new_inc["income_pension"]
            + new_inc["income_misc_business"]
            + new_inc["income_other"]
            + new_inc["income_stcg"]
            + new_inc["income_ltcg"]
            + new_inc["income_occasional"]
            + new_inc["sep_stcg_general"]
            + new_inc["sep_stcg_reduced"]
            + new_inc["sep_ltcg_general"]
            + new_inc["sep_ltcg_specific"]
            + new_inc["sep_ltcg_reduced"]
            + new_inc["sep_stock_general"]
            + new_inc["sep_stock_listed"]
            + new_inc["sep_dividend_listed"]
            + new_inc["sep_futures"]
            + new_inc["sep_forestry"]
        )
        new_dep_n      = rng.integers(0, 3, n_new)
        new_dep        = (new_dep_n * 330_000).astype(int)
        new_spo        = np.where(rng.random(n_new) < 0.20, 330_000, 0).astype(int)
        new_si         = (new_sal_gross * 0.14).round(0).astype(int)
        new_li         = np.where(rng.random(n_new) < 0.79,
                                  rng.integers(10_000, _LIFE_INS_UPPER + 1, size=n_new), 0).astype(int)
        new_eq         = np.where(rng.random(n_new) < 0.38,
                                  rng.integers(5_000, _EARTHQUAKE_INS_UPPER + 1, size=n_new), 0).astype(int)
        new_sbiz       = np.where(rng.random(n_new) < 0.13,
                                  rng.integers(100_000, 840_001, size=n_new), 0).astype(int)
        new_basic      = compute_basic_deduction(new_total).round(0).astype(int)
        new_taxable    = (
            new_total - new_si - new_li - new_eq - new_dep - new_spo - new_basic
        ).clip(0).round(0).astype(int)

        new_donation = np.where(rng.random(n_new) < _donation_rate,
                                rng.integers(10_000, 200_001, size=n_new), 0).astype(int)
        new_h_probs  = np.array([HOUSING_PROB_BY_AGE.get(ag, 0.0) for ag in new_age_groups])
        new_housing  = np.where(
            rng.random(n_new) < new_h_probs,
            rng.integers(50_000, HOUSING_PARAMS["upper_limit"] + 1, size=n_new), 0,
        ).astype(int)
        new_furusato = estimate_furusato_resident_deduction(
            new_taxable,
            donation_rate=FURUSATO_PARAMS["donation_rate"],
            one_stop_ratio=FURUSATO_PARAMS["one_stop_ratio"],
        ).astype(int)
        new_adj   = np.where(new_taxable > 0, 2_500, 0).astype(int)
        new_div_credit = (new_inc["income_dividend"] * 0.028).round(0).astype(int)
        new_div_split  = ((new_inc["sep_stock_listed"] + new_inc["sep_dividend_listed"]) * 0.05).round(0).astype(int)
        new_tax   = (
            np.maximum(new_taxable * 0.10 - new_housing - new_furusato - new_adj
                       - new_div_credit - new_div_split, 0) + 5_300
        ).round(0).astype(int)
        new_is_non_taxable = compute_non_taxable_flag(
            new_total, new_dep_n, (new_spo > 0).astype(int)
        )
        new_tax = np.where(new_is_non_taxable, 0, new_tax).astype(int)

        new_df = pd.DataFrame({
            "仮ID"              : new_ids,
            "年度"                   : yr,
            "性別"                 : rng.choice([0, 1], size=n_new, p=GENDER_RATIO),
            "年齢"                    : new_age_arr,
            "年齢区分"              : new_age_groups,
            # 収入
            "給与収入"    : new_sal_gross.round(0).astype(int),
            "雑収入_公的年金等"   : new_inc["income_pension_gross"].round(0).astype(int),
            # 所得
            "給与所得"          : new_sal_net.round(0).astype(int),
            "事業所得_営業等"        : new_inc["income_business"].round(0).astype(int),
            "事業所得_農業"         : new_inc["income_farming"].round(0).astype(int),
            "不動産所得"        : new_inc["income_property"].round(0).astype(int),
            "利子所得"        : new_inc["income_interest"].round(0).astype(int),
            "配当所得"        : new_inc["income_dividend"].round(0).astype(int),
            "雑所得_公的年金等"         : new_inc["income_pension"].round(0).astype(int),
            "雑所得_業務"   : new_inc["income_misc_business"].round(0).astype(int),
            "雑所得_その他"           : new_inc["income_other"].round(0).astype(int),
            "総合短期譲渡所得"            : new_inc["income_stcg"].round(0).astype(int),
            "総合長期譲渡所得"            : new_inc["income_ltcg"].round(0).astype(int),
            "一時所得"      : new_inc["income_occasional"].round(0).astype(int),
            # 分離課税所得
            "分離_短期譲渡一般"       : new_inc["sep_stcg_general"].round(0).astype(int),
            "分離_短期譲渡軽減"       : new_inc["sep_stcg_reduced"].round(0).astype(int),
            "分離_長期譲渡一般"       : new_inc["sep_ltcg_general"].round(0).astype(int),
            "分離_長期譲渡特定"      : new_inc["sep_ltcg_specific"].round(0).astype(int),
            "分離_長期譲渡軽減"       : new_inc["sep_ltcg_reduced"].round(0).astype(int),
            "分離_一般株式譲渡"      : new_inc["sep_stock_general"].round(0).astype(int),
            "分離_特定株式譲渡"       : new_inc["sep_stock_listed"].round(0).astype(int),
            "分離_上場株式配当"    : new_inc["sep_dividend_listed"].round(0).astype(int),
            "分離_先物取引"            : new_inc["sep_futures"].round(0).astype(int),
            "分離_山林"           : new_inc["sep_forestry"].round(0).astype(int),
            # 所得控除
            "社会保険料控除"      : new_si,
            "小規模企業共済等掛金控除"   : new_sbiz,
            "生命保険料控除"        : new_li,
            "地震保険料控除"  : new_eq,
            "雑損控除"        : 0,
            "医療費控除"         : 0,
            "障害者控除"      : 0,
            "寡婦控除"           : 0,
            "配偶者控除"          : new_spo,
            "配偶者特別控除"  : 0,
            "扶養控除"       : new_dep,
            "基礎控除"           : new_basic,
            "勤労学生控除" : 0,
            "寄附金控除" : new_donation,
            "扶養人数"            : new_dep_n,
            # 課税所得
            "課税標準額"         : new_taxable,
            # 税額控除
            "住宅借入金特別控除"     : new_housing,
            "寄附金税額控除"    : new_furusato,
            "調整控除"   : new_adj,
            "配当控除"     : new_div_credit,
            "外国税額控除"      : 0,
            "配当割額・株式等譲渡所得割額の控除": new_div_split,
            # 税額
            "年税額"             : new_tax,
            "徴収区分"        : rng.choice([1, 2], size=n_new, p=[0.80, 0.20]),
            # EDA用内訳
            "市町村_均等割"         : np.where(new_tax > 0, 3_500, 0).astype(int),
            "市町村_所得割"          : (np.maximum(new_tax - 5_300, 0) * 0.6).round(0).astype(int),
            "都道府県_均等割"         : np.where(new_tax > 0, 1_800, 0).astype(int),
            "都道府県_所得割"          : (np.maximum(new_tax - 5_300, 0) * 0.4).round(0).astype(int),
            "控除不足額"              : 0,
        })

        year_df = pd.concat([stay_df, new_df], ignore_index=True)
        all_dfs.append(year_df)
        current_df = year_df.copy()
        print(f"  {yr}年: {len(year_df):,} 件生成")

    return pd.concat(all_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PER_YEAR,
                        help=f"1年あたりのレコード数（デフォルト: {N_PER_YEAR:,}）")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    all_years = TRAIN_YEARS + [TEST_YEAR]
    print(f"=== 01: ダミーデータ生成（{all_years[0]}〜{all_years[-1]}年） ===\n")
    print(f"  1年あたり {args.n:,} 件 × {len(all_years)} 年分\n")

    print(f"{all_years[0]}年: 基準年生成中...")
    df = generate_dummy(n_per_year=args.n, years=all_years)

    print(f"\n生成完了: 合計 {len(df):,} 件")
    print("\n── 年度別サマリー ──")
    print(df.groupby("年度")["年税額"].agg(
        件数="count",
        税額合計_億円=lambda x: round(x.sum() / 1e8, 2),
        一人あたり平均_万円=lambda x: round(x.mean() / 1e4, 1),
    ).to_string())

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n→ {OUTPUT_PATH} に保存")
    print("\n次: python 03_feature_eng.py")


if __name__ == "__main__":
    main()
