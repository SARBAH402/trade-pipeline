{{ config(
    materialized='external',
    location='s3://LESSONED/comtrade-bucket/marts_clustering.parquet'
) }}


WITH silver as (
    SELECT * 
    FROM {{ref('stg_comtrade')}}
),

filtered_silver AS (
    -- Step 1: Clean the input to ensure mathematical purity
    SELECT 
        reporter_name,
        partner_name,
        period,
        commodity,
        shippment,
        primaryValue,
        net_weight_kg
    FROM silver
    WHERE partnerCode != 0       -- Drop the 'World' aggregate
      AND motCode != 0           -- Drop the 'All modes' aggregate
      AND primaryValue > 0       -- Prevent log(0) or negative value errors later
),

corridor_profiles AS (
    -- Step 2: Build the row-wise relationship resumes
    SELECT 
        reporter_name,
        partner_name,
        
        -- 1. Scale (Log-Transformed)
        -- Adding 1 is a standard data science trick to completely eliminate 
        -- any risk of a mathematical error if a value ever rounds to 0.
        LN(SUM(primaryValue) + 1) AS log_total_trade_value,
        
        -- 2. Economic Density (Price-per-kilo)
        -- NULLIF(..., 0) catches corridors that reported $ value but forgot weight, 
        -- forcing DuckDB to output NULL instead of crashing the pipeline.
        COALESCE(SUM(primaryValue) / NULLIF(SUM(net_weight_kg), 0), 0) AS price_per_kg,
        
        -- 3. Frequency (The Loyalty Score)
        COUNT(DISTINCT period) AS active_trading_months,
        
        -- 4. Air Freight Dependency (The Urgency Factor)
        -- Using ILIKE catches 'Air', 'air', or 'Air transport'. 
        -- Adjust the string if your API uses a different exact term.
        SUM(CASE WHEN shippment ILIKE '%air%' THEN primaryValue ELSE 0 END) 
            / NULLIF(SUM(primaryValue), 0) AS air_freight_pct,
          
        -- 5. Product Diversity (Complexity)
        COUNT(DISTINCT commodity) AS unique_commodities_traded
        
    FROM filtered_silver
    GROUP BY 
        reporter_name, 
        partner_name
)

SELECT * FROM corridor_profiles
