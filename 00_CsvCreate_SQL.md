# 個人住民税 課税データ抽出SQL

## 概要
Microsoft Access上で基幹システムからRDBMSを用いて課税データテーブルを抽出するSQL
個人IDは仮名化処理をしたのち、出力したCSVをPythonにインポートして分析する

## 使用環境
- Office : Microsoft Access (Microsoft Office 365)
- RDBMS  : Oracre Data Base 
- SQL    : SQL Server

## 処理内容

- 対象年度　　| 2020 年 ~ 2025 年　（各年度ごとに出力）
- 更正日　　　| 各年6月30日時点の最新データ
- レコード　　| 1個人1レコード。更正日が同じレコードが複数ある場合は履歴番号の最大のみ抽出  
- 個人データ　| 個人を特定するIDを未公開の四則演算で仮変換。csv 出力後、pythonにて各年度の個人IDで結合
- 年齢　　　　| 賦課期日（各年1月1日）時点の年齢を生年月日から計算し出力

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