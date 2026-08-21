{{ config(
    materialized='external',
    location='s3://LESSONED/comtrade-bucket/marts_timeseries.parquet'
) }}


WITH silver as (
    SELECT * 
    FROM {{ref('stg_comtrade')}}
),

base_aggregation AS (
    -- 1. Macro Grain & STRPTIME Conversion
    -- Convert period (e.g., 202001) to a valid Date (2020-01-01) and sum the value
    SELECT 
        CAST(STRPTIME(CAST(period AS VARCHAR) || '01', '%Y%m%d') AS DATE) AS trade_date,
        reporter_name,
        partner_name,
        SUM(primaryValue) AS total_value_usd
    FROM silver
    WHERE partnerCode != 0
    AND motCode != 0
    GROUP BY 1, 2, 3
),

corridor_combinations AS (
    -- 2. Identify all unique trade corridors
    SELECT DISTINCT 
        reporter_name, 
        partner_name
    FROM base_aggregation
),

date_bounds AS (
    -- 3. Find the absolute start and end months of the entire dataset
    SELECT 
        MIN(trade_date) AS min_date,
        MAX(trade_date) AS max_date
    FROM base_aggregation
),

date_spine AS (
    -- 4. The Date Spine
    -- DuckDB uses UNNEST(GENERATE_SERIES) to create an unbroken monthly calendar
    SELECT 
        UNNEST(GENERATE_SERIES(
            (SELECT min_date FROM date_bounds), 
            (SELECT max_date FROM date_bounds), 
            INTERVAL 1 MONTH
        )) AS spine_date
),

dense_grid AS (
    -- 5. Cross join the calendar with the corridors to create the perfect grid
    -- This guarantees a row exists for every month for every country pair
    SELECT 
        d.spine_date AS trade_date,
        c.reporter_name,
        c.partner_name
    FROM date_spine d
    CROSS JOIN corridor_combinations c
),

stitched_data AS (
    -- 6. Join the actual trade data onto the perfect grid
    -- Any missing months are coalesced to 0 (no trade occurred)
    SELECT 
        g.trade_date,
        g.reporter_name,
        g.partner_name,
        COALESCE(b.total_value_usd, 0) AS total_value_usd
    FROM dense_grid g
    LEFT JOIN base_aggregation b
        ON g.trade_date = b.trade_date
        AND g.reporter_name = b.reporter_name
        AND g.partner_name = b.partner_name
),

ml_features AS (
    -- 7. Time-Series Feature Engineering (Window Functions)
    SELECT 
        trade_date,
        reporter_name,
        partner_name,
        total_value_usd,
        
        -- Lag 1: The previous month's trade value
        LAG(total_value_usd, 1) OVER (
            PARTITION BY reporter_name, partner_name 
            ORDER BY trade_date
        ) AS lag_1_value,

        LAG(total_value_usd, 12) OVER (
            PARTITION BY reporter_name, partner_name
            ORDER BY trade_date
        ) AS lag_12_value,

        -- For Power BI / Tableau Reporting
AVG(total_value_usd) OVER (
    PARTITION BY reporter_name, partner_name 
    ORDER BY trade_date 
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
) AS reporting_3mo_avg,

-- For Predictive Modeling / Regression Algorithms
AVG(total_value_usd) OVER (
    PARTITION BY reporter_name, partner_name 
    ORDER BY trade_date 
    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
) AS predictive_3mo_lag_avg,

AVG(total_value_usd) OVER (
            PARTITION BY reporter_name, partner_name
            ORDER BY trade_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS predictive_6mo_lag_avg,

STDDEV(total_value_usd) OVER (
            PARTITION BY reporter_name, partner_name
            ORDER BY trade_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rolling_6mo

 FROM stitched_data)


-- Final Output: Filter out the initial rows that lack sufficient history for the window functions
SELECT * 
FROM ml_features
WHERE predictive_3mo_lag_avg IS NOT NULL
AND reporting_3mo_avg IS NOT NULL
AND rolling_6mo IS NOT NULL
AND predictive_6mo_lag_avg IS NOT NULL 
ORDER BY reporter_name, partner_name, trade_date
