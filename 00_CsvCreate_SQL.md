# 個人住民税 課税データ抽出SQL

## 概要
Microsoft Accessを用いて基幹システムから課税データを抽出するSQL
個人IDは仮名化処理済みのCSVをPythonにインポートして分析する

## 使用環境
- Microsoft Access 2019
- 対象年度：2025年度
- 賦課期日：2025年1月1日

## 処理内容
- 更正日が2025年6月30日以前かつ最新のレコードを抽出
- 同一更正日の場合は履歴番号が最大のレコードを採用
- 個人IDを四則演算により仮名化
- 賦課期日時点の年齢を算出

## 注意事項
- テーブル名・フィールド名は実務のものをサンプル用に置き換え済み
- 変換式の実際の数値は非公開

## SQL

```sql
SELECT
    FORMAT(CLng(A.個人ID) * [乗数] + [加算値], "0000000000") AS 仮ID,
    A.所得フィールド1,
    A.所得フィールド2,
    A2.控除フィールド1,
    AT.性別,
    INT((CLng(20250101) - CLng(AT.生年月日)) / 10000) AS 年齢
FROM 課税テーブル AS A
INNER JOIN 控除テーブル AS A2
    ON  A.個人ID   = A2.個人ID
    AND A.年度     = A2.年度
    AND A.更正日   = A2.更正日
    AND A.履歴番号 = A2.履歴番号
INNER JOIN 宛名テーブル AS AT
    ON A.個人ID = AT.個人ID
WHERE A.年度 = 2025
  AND A.更正日 = (
      SELECT MAX(B.更正日)
      FROM 課税テーブル AS B
      WHERE B.個人ID = A.個人ID
        AND B.年度 = 2025
        AND B.更正日 <= 20250630
  )
  AND A.履歴番号 = (
      SELECT MAX(C.履歴番号)
      FROM 課税テーブル AS C
      WHERE C.個人ID = A.個人ID
        AND C.年度 = 2025
        AND C.更正日 = A.更正日
  )
...
```

## 逆変換クエリ（Access内管理用）

```sql
SELECT
    FORMAT((CLng(仮名ID) - [加算値]) / [乗数], "0000000000") AS 元ID
FROM 変換後テーブル
```