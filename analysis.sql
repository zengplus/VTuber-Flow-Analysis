-- =========================================================
-- 0. 建库
-- duckdb mydb.duckdb < analysis.sql
-- =========================================================

-- 0.1 主播维度表（仅真实主播 + -3=YLG）
CREATE OR REPLACE TABLE dim_liver (
    liver BIGINT PRIMARY KEY,
    name  VARCHAR
);
INSERT OR REPLACE INTO dim_liver VALUES
(672328094,'嘉然'),(672346917,'向晚'),(672353429,'贝拉'),(672342685,'乃琳'),(351609538,'珈乐'),
(3537115310721181,'心宜'),(3537115310721781,'思诺'),(440738032,'安可'),(698438232,'扇宝'),
(401315430,'星瞳'),(1660392980,'恬豆'),(1217754423,'又一'),(1878154667,'沐霂'),
(1900141897,'梨安'),(1778026586,'米诺'),(1875044092,'莞儿'),(1811071010,'虞莫'),
(1669777785,'露早'),(1795147802,'柚恩'),(7706705,'阿梓'),(434334701,'七海'),
(666726799,'悠亚'),(480680646,'阿萨'),(14387072,'小可'),(477317922,'弥希'),
(1265680561,'塔菲'),(2132180406,'奶绿'),(1265605287,'麻尤米'),(1908273021,'露娜'),
(1855519979,'dodo'),(2000609327,'露米'),(1377212676,'永恒娘'),(686201061,'古堡龙姬'),
(3461582034045213,'宣小纸'),(1011797664,'卡缇娅'),(3461578781362279,'叶河黎'),
(1383815813,'吉诺儿'),(1219196749,'唐九夏'),(1734978373,'小柔'),(1501380958,'艾露露'),
(3493139945884106,'雪糕'),(51030552,'星汐'),(15641218,'笙歌'),(3821157,'东爱璃'),
(-3,'YLG');

-- 1. 原始事件（按月分区）
CREATE OR REPLACE TABLE events AS
SELECT uid, liver, ts, DATE_TRUNC('month', ts) AS month
FROM 'data/fans_events.parquet';

-- 2. 用户-主播首次互动
CREATE OR REPLACE TABLE liver_new AS
SELECT liver, DATE_TRUNC('month', MIN(ts)) AS month, uid
FROM events
GROUP BY uid, liver;

-- 3. 用户-主播互动特征
CREATE OR REPLACE TABLE user_liver_stats AS
WITH mc AS (
    SELECT uid, liver,
           COUNT(DISTINCT DATE_TRUNC('month', ts)) AS month_cnt,
           MAX(ts) AS last_ts
    FROM events
    GROUP BY uid, liver
), ref AS (SELECT MAX(ts) AS ref_date FROM events)
SELECT uid, liver, month_cnt, last_ts,
       DATE_DIFF('day', last_ts, ref_date) AS recency_days
FROM mc, ref;

-- 4. 主播阈值
CREATE OR REPLACE TABLE liver_threshold AS
SELECT liver,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY month_cnt) AS month_med,
       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY recency_days) AS recency_75
FROM user_liver_stats
GROUP BY liver;

-- 5. 用户-主播标签
CREATE OR REPLACE TABLE user_liver_type AS
SELECT s.uid, s.liver,
       CASE WHEN s.month_cnt >= t.month_med AND s.recency_days <= t.recency_75
            THEN -2 ELSE -1 END AS liver_label
FROM user_liver_stats s
JOIN liver_threshold t USING (liver);

-- 6. 全局流动观众
CREATE OR REPLACE TABLE user_global_type AS
SELECT uid,
       CASE MIN(liver_label) WHEN -1 THEN -3 ELSE MIN(liver_label) END AS global_label
FROM user_liver_type
GROUP BY uid;
INSERT INTO user_liver_type
SELECT uid, -3 AS liver, -3 AS liver_label
FROM user_global_type WHERE global_label = -3;

-- =========================================================
-- 7. 来源主播（替换段）
-- =========================================================
CREATE OR REPLACE TABLE new_source AS
WITH tmp AS (
    SELECT n.uid,
           n.liver            AS target_liver,      -- 起别名
           n.month,
           (SELECT e.liver
            FROM events e
            WHERE e.uid = n.uid
              AND e.liver <> n.liver
              AND e.ts < n.month
            GROUP BY e.liver
            ORDER BY COUNT(*) DESC
            LIMIT 1) AS source_liver_raw
    FROM liver_new n
)
SELECT uid,
       target_liver,
       DATE_TRUNC('month', month) AS month,
       COALESCE(source_liver_raw, -3) AS source_liver
FROM tmp;

-- =========================================================
-- 8. 去向主播（替换段）
-- =========================================================
CREATE OR REPLACE TABLE new_target AS
WITH tmp AS (
    SELECT n.uid,
           n.liver            AS source_liver,      -- 起别名
           n.month,
           (SELECT e.liver
            FROM events e
            WHERE e.uid = n.uid
              AND e.liver <> n.liver
              AND e.ts >= n.month + INTERVAL 1 MONTH
            GROUP BY e.liver
            ORDER BY COUNT(*) DESC
            LIMIT 1) AS target_liver_raw
    FROM liver_new n
)
SELECT uid,
       source_liver,
       DATE_TRUNC('month', month) AS month,
       COALESCE(target_liver_raw, -3) AS target_liver
FROM tmp;

-- 9. 月度流入矩阵
CREATE OR REPLACE TABLE monthly_matrix_in AS
SELECT target_liver, DATE_TRUNC('month', month) AS month, source_liver, COUNT(*) AS cnt
FROM new_source
GROUP BY ALL;

-- 10. 月度流出矩阵
CREATE OR REPLACE TABLE monthly_matrix_out AS
SELECT source_liver, DATE_TRUNC('month', month) AS month, target_liver, COUNT(*) AS cnt
FROM new_target
GROUP BY ALL;

-- 11. AARRR 漏斗
CREATE OR REPLACE TABLE aarr_metrics AS
WITH acq AS (
    SELECT liver, DATE_TRUNC('month', month) AS month, COUNT(*) AS acq
    FROM liver_new GROUP BY liver, month
), activ AS (
    SELECT ln.liver, DATE_TRUNC('month', ln.month) AS month,
           COUNT(DISTINCT ln.uid) AS activ
    FROM liver_new ln
    WHERE EXISTS (
        SELECT 1 FROM events e
        WHERE e.uid = ln.uid AND e.liver = ln.liver
          AND e.ts BETWEEN ln.month AND ln.month + INTERVAL 7 DAY)
    GROUP BY ln.liver, ln.month
), reten AS (
    SELECT ln.liver, DATE_TRUNC('month', ln.month) AS month,
           COUNT(DISTINCT ln.uid) AS reten
    FROM liver_new ln
    WHERE EXISTS (
        SELECT 1 FROM events e
        WHERE e.uid = ln.uid AND e.liver = ln.liver
          AND e.ts BETWEEN ln.month + INTERVAL 27 DAY AND ln.month + INTERVAL 33 DAY)
    GROUP BY ln.liver, ln.month
), refer AS (
    SELECT source_liver AS liver, DATE_TRUNC('month', month) AS month,
           COUNT(DISTINCT uid) AS refer
    FROM new_source WHERE source_liver <> -3
    GROUP BY source_liver, month
), rev AS (
    SELECT ln.liver, DATE_TRUNC('month', ln.month) AS month,
           SUM(c.cnt) AS revenue
    FROM liver_new ln
    JOIN (
        SELECT uid, liver, COUNT(*) AS cnt
        FROM events
        WHERE ts BETWEEN ln.month AND ln.month + INTERVAL 30 DAY
        GROUP BY uid, liver
    ) c ON c.uid = ln.uid AND c.liver = ln.liver
    GROUP BY ln.liver, ln.month
)
SELECT a.liver, a.month, a.acq,
       COALESCE(ac.activ,0) AS activ,
       COALESCE(r.reten,0)  AS reten,
       COALESCE(ref.refer,0) AS refer,
       COALESCE(rv.revenue,0) AS revenue,
       COALESCE(ac.activ,0)*1.0 / NULLIF(a.acq,0) AS activation_rate,
       COALESCE(r.reten,0)*1.0 / NULLIF(a.acq,0)  AS retention_rate,
       COALESCE(ref.refer,0)*1.0 / NULLIF(a.acq,0) AS referral_rate
FROM acq a
LEFT JOIN activ ac USING (liver, month)
LEFT JOIN reten r  USING (liver, month)
LEFT JOIN refer ref USING (liver, month)
LEFT JOIN rev rv USING (liver, month);

-- 12. RFM 分层
CREATE OR REPLACE TABLE rfm_user AS
WITH base AS (
    SELECT uid, liver,
           DATE_DIFF('day', MAX(ts), (SELECT MAX(ts) FROM events)) AS recent_days,
           COUNT(*) AS freq,
           COUNT(*) AS monetary
    FROM events
    WHERE ts >= (SELECT MAX(ts) - INTERVAL 90 DAY FROM events)
    GROUP BY uid, liver
), qt AS (
    SELECT liver,
           PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY recent_days) AS r20,
           PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY recent_days) AS r80,
           PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY freq) AS f20,
           PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY freq) AS f80
    FROM base
    GROUP BY liver
), score AS (
    SELECT b.*,
           CASE WHEN recent_days <= r80 THEN 5
                WHEN recent_days <= r20 THEN 3 ELSE 1 END AS r,
           CASE WHEN freq >= f80 THEN 5
                WHEN freq >= f20 THEN 3 ELSE 1 END AS f,
           CASE WHEN monetary >= f80 THEN 5
                WHEN monetary >= f20 THEN 3 ELSE 1 END AS m
    FROM base b
    JOIN qt q USING (liver)
)
SELECT *, r||f||m AS rfm_code,
       CASE WHEN r>=4 AND f>=4 AND m>=4 THEN '高价值忠诚'
            WHEN r>=3 AND f>=3 THEN '潜力用户'
            WHEN r<=2 THEN '流失风险'
            ELSE '一般用户' END AS rfm_tag
FROM score;

-- 13. 观众类型维表
CREATE OR REPLACE TABLE dim_audience_type (liver BIGINT PRIMARY KEY, name VARCHAR);
INSERT OR REPLACE INTO dim_audience_type VALUES
(-3,'YLG（全站流动）'),(-2,'固定观众'),(-1,'流动观众');

-- 14. 每日事件按类型（全站）
CREATE OR REPLACE TABLE daily_events_by_type AS
SELECT DATE_TRUNC('day', ts) AS day,
       SUM(CASE WHEN liver = -3 THEN 1 ELSE 0 END) AS weak,
       SUM(CASE WHEN liver <> -3 THEN 1 ELSE 0 END) AS strong,
       COUNT(*) AS total
FROM events
GROUP BY 1;

-- 15. 固定观众 cohort 30d 留存
CREATE OR REPLACE TABLE cohort_retention_30d AS
WITH cohort AS (
    SELECT DATE_TRUNC('month', n.month) AS cohort_month, n.uid
    FROM liver_new n
    JOIN user_liver_type l
      ON n.uid = l.uid AND n.liver = l.liver AND l.liver_label = -2
), activity AS (
    SELECT DISTINCT DATE_TRUNC('month', ts) AS month, uid FROM events
)
SELECT c.cohort_month,
       DATE_DIFF('month', c.cohort_month, a.month) AS month_age,
       COUNT(DISTINCT a.uid) AS retained
FROM cohort c
JOIN activity a ON a.uid = c.uid
GROUP BY 1, 2;

-- 16. RFM 每日事件贡献
CREATE OR REPLACE TABLE rfm_daily_contrib AS
SELECT DATE_TRUNC('day', e.ts) AS day,
       r.rfm_tag,
       e.liver,
       COUNT(DISTINCT e.uid) AS users,
       COUNT(*) AS events,
       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY DATE_TRUNC('day', e.ts), e.liver) AS evt_ratio
FROM events e
JOIN rfm_user r ON e.uid = r.uid AND e.liver = r.liver
GROUP BY 1, 2, 3;

-- 17. 用户全局标签（复用）
CREATE OR REPLACE TABLE uid_global AS
SELECT uid, MODE(liver_label) AS global_label
FROM user_liver_type
GROUP BY uid;

-- 18. 用户-主播-自然月活跃视图
CREATE OR REPLACE VIEW v_user_liver_month_active AS
SELECT DISTINCT DATE_TRUNC('month', ts) AS month, liver, uid
FROM events
WHERE liver > 0;

-- 19. 全站 MAU 分层
CREATE OR REPLACE TABLE monthly_mau_layer AS
SELECT a.month,
       COUNT(DISTINCT a.uid) AS mau,
       COUNT(DISTINCT CASE WHEN g.global_label = -2 THEN a.uid END) AS fixed_mau,
       COUNT(DISTINCT CASE WHEN g.global_label = -1 THEN a.uid END) AS flowing_mau,
       COUNT(DISTINCT CASE WHEN g.global_label = -3 THEN a.uid END) AS ylg_mau
FROM (SELECT DISTINCT month, uid FROM v_user_liver_month_active) a
JOIN uid_global g ON g.uid = a.uid
GROUP BY a.month;

-- 20. 主播 MAU 分层
CREATE OR REPLACE TABLE liver_monthly_mau_layer AS
SELECT a.month, a.liver,
       COUNT(DISTINCT a.uid) AS mau,
       COUNT(DISTINCT CASE WHEN g.global_label = -2 THEN a.uid END) AS fixed_mau,
       COUNT(DISTINCT CASE WHEN g.global_label = -1 THEN a.uid END) AS flowing_mau,
       COUNT(DISTINCT CASE WHEN g.global_label = -3 THEN a.uid END) AS ylg_mau
FROM v_user_liver_month_active a
JOIN uid_global g ON g.uid = a.uid
GROUP BY a.month, a.liver;

CREATE INDEX IF NOT EXISTS idx_liver_month ON liver_monthly_mau_layer (liver, month);

-- 21. 行业渗透率堆叠
CREATE OR REPLACE VIEW v_penetration_stacked AS
WITH industry AS (
    SELECT month, COUNT(DISTINCT uid) AS industry_mau
    FROM v_user_liver_month_active
    GROUP BY month
), anchor AS (
    SELECT month, liver, COUNT(DISTINCT uid) AS mau
    FROM v_user_liver_month_active
    GROUP BY month, liver
)
SELECT a.month, a.liver, a.mau, i.industry_mau,
       a.mau * 1.0 / i.industry_mau AS penetration,
       (a.mau * 1.0 / i.industry_mau)
         - LAG(a.mau * 1.0 / i.industry_mau)
           OVER (PARTITION BY a.liver ORDER BY a.month) AS pct_change
FROM anchor a
JOIN industry i USING (month);

-- 22. 流动层净流失（全站）
CREATE OR REPLACE VIEW v_flowing_net_churn AS
WITH in_flow AS (
    SELECT DATE_TRUNC('month', month) AS month,
           SUM(CASE WHEN source_liver = -3 THEN cnt ELSE 0 END) AS in_ylg
    FROM monthly_matrix_in
    GROUP BY month
), out_flow AS (
    SELECT DATE_TRUNC('month', month) AS month,
           SUM(CASE WHEN target_liver = -3 THEN cnt ELSE 0 END) AS out_ylg
    FROM monthly_matrix_out
    GROUP BY month
), flow AS (
    SELECT COALESCE(i.month, o.month) AS month,
           COALESCE(in_ylg, 0)  AS in_ylg,
           COALESCE(out_ylg, 0) AS out_ylg
    FROM in_flow i
    FULL JOIN out_flow o USING (month)
), pop AS (
    SELECT month, flowing_mau
    FROM monthly_mau_layer
)
SELECT f.month,
       (in_ylg - out_ylg) AS net_flow,
       (in_ylg - out_ylg) * 1.0 / NULLIF(p.flowing_mau, 0) AS net_rate,
       - (in_ylg - out_ylg) * 1.0 / NULLIF(p.flowing_mau, 0) AS churn_rate
FROM flow f
JOIN pop p USING (month);

-- 23. S 曲线数据
CREATE OR REPLACE VIEW v_scurve_data AS
SELECT liver, month, mau,
       ROW_NUMBER() OVER (PARTITION BY liver ORDER BY month) - 1 AS seq
FROM liver_monthly_mau_layer;

CREATE OR REPLACE TABLE liver_logistic_params AS
WITH raw AS (
    SELECT liver, seq, mau
    FROM v_scurve_data
    WHERE seq IS NOT NULL AND mau > 0
), param AS (
    SELECT liver,
           MAX(mau) * 1.2  AS K_init,
           0.35            AS r_init,
           COUNT(*) * 0.5  AS t0_init,
           COUNT(*)        AS cnt
    FROM raw
    GROUP BY liver
    HAVING COUNT(*) >= 5
)
SELECT liver, K_init AS K, r_init AS r, t0_init AS t0
FROM param;

CREATE OR REPLACE VIEW v_five_stage_all AS
WITH logi AS (
    SELECT s.liver, s.month, s.mau, s.seq,
           p.K, p.r, p.t0,
           p.K * p.r * EXP(-p.r * (s.seq - p.t0))
             / POWER(1 + EXP(-p.r * (s.seq - p.t0)), 2) AS deriv
    FROM v_scurve_data s
    JOIN liver_logistic_params p ON p.liver = s.liver
)
SELECT *,
       CASE
         WHEN seq < t0 - 2                  THEN '起飞期'
         WHEN seq BETWEEN t0 - 2 AND t0 + 2 THEN '加速期'
         WHEN seq BETWEEN t0 + 2 AND t0 + 6 THEN '峰值冲刺期'
         WHEN seq BETWEEN t0 + 6 AND t0 + 10 THEN '增速放缓期'
         WHEN seq BETWEEN t0 + 10 AND t0 + 14 THEN '回落预警期'
         ELSE '衰退期'
       END AS stage
FROM logi;

-- 24. 三阶段 cohort 漏斗
CREATE OR REPLACE VIEW v_layer_funnel AS
WITH new_cohort AS (
    SELECT liver,
           DATE_TRUNC('month', month) AS cohort_month,
           uid
    FROM liver_new
),
fixed_now AS (
    SELECT uid, liver
    FROM user_liver_type
    WHERE liver_label = -2
),
base AS (
    SELECT liver, cohort_month, COUNT(DISTINCT uid) AS users
    FROM new_cohort
    GROUP BY liver, cohort_month
),
s2 AS (
    SELECT liver, cohort_month, COUNT(DISTINCT n.uid) AS users
    FROM new_cohort n
    WHERE EXISTS (
        SELECT 1
        FROM fixed_now f
        WHERE f.uid = n.uid
          AND EXISTS (
              SELECT 1
              FROM new_cohort n2
              WHERE n2.uid = n.uid
                AND n2.liver = n.liver
                AND n2.cohort_month = n.cohort_month + INTERVAL 1 MONTH))
    GROUP BY liver, cohort_month
),
s3 AS (
    SELECT liver, cohort_month, COUNT(DISTINCT n.uid) AS users
    FROM new_cohort n
    WHERE EXISTS (
        SELECT 1
        FROM fixed_now f
        WHERE f.uid = n.uid
          AND EXISTS (
              SELECT 1
              FROM new_cohort n3
              WHERE n3.uid = n.uid
                AND n3.liver = n.liver
                AND n3.cohort_month = n.cohort_month + INTERVAL 2 MONTH))
    GROUP BY liver, cohort_month
)
SELECT liver, cohort_month, '新增' AS stage, users, 1.0 AS pct FROM base
UNION ALL
SELECT liver, cohort_month, '次月固定', s2.users, s2.users * 1.0 / NULLIF(b.users, 0)
FROM base b
JOIN s2 USING (liver, cohort_month)
UNION ALL
SELECT liver, cohort_month, '第三月固定', s3.users, s3.users * 1.0 / NULLIF(b.users, 0)
FROM base b
JOIN s3 USING (liver, cohort_month);

-- 2. 落盘成表（Streamlit 里用表或视图都行）
CREATE OR REPLACE TABLE layer_funnel_liver AS
SELECT * FROM v_layer_funnel;

-- 25. 预处理汇总（一次性补齐，不再报错）
CREATE OR REPLACE TABLE pre_mau_layer AS
SELECT month, liver, mau, fixed_mau, flowing_mau, ylg_mau,
       SUM(mau)        OVER(PARTITION BY month) AS industry_mau,
       SUM(fixed_mau)  OVER(PARTITION BY month) AS industry_fixed,
       SUM(flowing_mau)OVER(PARTITION BY month) AS industry_flowing,
       SUM(ylg_mau)    OVER(PARTITION BY month) AS industry_ylg
FROM liver_monthly_mau_layer;

-- 26. 主播×日事件类型（弱/强）
CREATE OR REPLACE TABLE daily_events_by_liver AS
SELECT DATE_TRUNC('day', ts) AS day,
       liver,
       SUM(CASE WHEN liver = -3 THEN 1 ELSE 0 END) AS weak,
       SUM(CASE WHEN liver <> -3 THEN 1 ELSE 0 END) AS strong,
       COUNT(*) AS total
FROM events
WHERE liver > 0
GROUP BY 1, 2;

CREATE OR REPLACE TABLE pre_daily_type AS
SELECT day, liver, weak, strong, total
FROM daily_events_by_liver;

-- 27. 主播×月 AARRR
CREATE OR REPLACE TABLE pre_aarr AS
SELECT liver, month, acq, activ, reten, refer, revenue,
       activation_rate, retention_rate, referral_rate
FROM aarr_metrics;

-- 28. 主播×月流动层净流失
CREATE OR REPLACE TABLE flowing_net_churn_liver AS
WITH in_flow AS (
    SELECT target_liver AS liver, DATE_TRUNC('month', month) AS month,
           SUM(CASE WHEN source_liver = -3 THEN cnt ELSE 0 END) AS in_ylg
    FROM monthly_matrix_in
    WHERE target_liver > 0
    GROUP BY 1, 2
), out_flow AS (
    SELECT source_liver AS liver, DATE_TRUNC('month', month) AS month,
           SUM(CASE WHEN target_liver = -3 THEN cnt ELSE 0 END) AS out_ylg
    FROM monthly_matrix_out
    WHERE source_liver > 0
    GROUP BY 1, 2
), flow AS (
    SELECT COALESCE(i.liver, o.liver) AS liver,
           COALESCE(i.month, o.month) AS month,
           COALESCE(in_ylg, 0)  AS in_ylg,
           COALESCE(out_ylg, 0) AS out_ylg
    FROM in_flow i
    FULL JOIN out_flow o USING (liver, month)
), pop AS (
    SELECT liver, month, flowing_mau
    FROM liver_monthly_mau_layer
)
SELECT f.liver, f.month,
       (in_ylg - out_ylg) AS net_flow,
       - (in_ylg - out_ylg) * 1.0 / NULLIF(p.flowing_mau, 0) AS churn_rate
FROM flow f
JOIN pop p USING (liver, month);

CREATE OR REPLACE TABLE pre_churn AS
SELECT liver, month, net_flow, churn_rate
FROM flowing_net_churn_liver;

-- 29. 渗透率
CREATE OR REPLACE TABLE pre_penetration AS
SELECT month, liver, mau, industry_mau, penetration, pct_change
FROM v_penetration_stacked;