{{ config(
    materialized='external',
    location='s3://LESSONED/comtrade-bucket/marts_regression.parquet'
) }}


WITH silver as (
    SELECT * 
    FROM {{ref('stg_comtrade')}}
),

filtered_silver AS (
    -- Step 1: Base filters for transactional purity
    SELECT 
        period,
        commodity,
        reporter_name,
        partner_name,
        qty,
        net_weight_kg,
        im_exp,
        primaryValue,
        shippment
    FROM silver
    WHERE partnerCode != 0 
      AND motCode != 0
      AND primaryValue > 0 -- Required to prevent LN(0) or negative errors
)

SELECT 
    -- CATEGORICAL FEATURES (Hand off to Python for One-Hot Encoding)
    CAST(RIGHT(period, 2) AS INTEGER) AS month,
    reporter_name,
    partner_name,
    commodity,
    shippment,
    im_exp,

    -- NUMERICAL FEATURES
    net_weight_kg,
    qty,

    -- Add any other numerical features you want to regress against here
    -- TARGET VARIABLES (The Y)
    primaryValue AS raw_target_value_usd,
    
    -- The Log-Transformed Target (Using +1 to prevent mathematical zero-errors)
    LN(primaryValue + 1) AS log_target_value,

    SUM(CASE WHEN shippment ILIKE '%air%' THEN primaryValue ELSE 0 END) 
            / NULLIF(SUM(primaryValue), 0) AS air_freight_pct
    
FROM filtered_silver
GROUP BY ALL
ORDER BY month ASC