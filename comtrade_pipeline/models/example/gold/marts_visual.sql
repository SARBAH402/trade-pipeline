{{ config(
    materialized='external',
    location='s3://LESSONED/comtrade-bucket/marts_dashboard.parquet'
) }}


WITH silver as (
    SELECT * 
    FROM {{ref('stg_comtrade')}}
),

filtered_silver AS (
    -- Step 1: Same strict foundation as the ML models
    SELECT *
    FROM silver
    WHERE partnerCode != 0 
      AND motCode != 0
),

dashboard_obt AS (
    -- Step 2: Build the One Big Table for BI slicing and dicing
    SELECT 
        -- TIME DIMENSIONS
        -- BI tools love explicit date parts for their drill-down hierarchies
        period,
        LEFT(CAST(period AS VARCHAR), 4) AS trade_year,
        RIGHT(CAST(period AS VARCHAR), 2) AS trade_month,
        
        -- GEOGRAPHY DIMENSIONS
        reporter_name AS reporting_country,
        partner_name AS partner_country,
        
        -- LOGISTICS & PRODUCT DIMENSIONS
          CASE 
        WHEN shippment IN ('Air', 'Postal consignments, mail or courier shipment') THEN 'Air Freight'
        WHEN shippment IN ('Road', 'Railway', 'Land') THEN 'Land Freight'
        WHEN shippment IN ('Sea', 'Water') THEN 'Ocean Freight'
        WHEN shippment IN ('Pipelines and cables', 'Pipelines', 'Self propelled goods', 'Other', 'Not elsewhere classified') THEN 'Special & Other'
        ELSE shippment
    END AS shippment_grouped,
        shippment,
        commodity,
        CASE 
        WHEN commodity LIKE '%Petroleum oils%' THEN 'Energy & Fuels'
        WHEN commodity LIKE '%Gold%' 
          OR commodity LIKE '%Aluminium ores%' 
          OR commodity LIKE '%Manganese ores%' THEN 'Minerals & Metals'
        WHEN commodity LIKE '%Cocoa beans%' 
          OR commodity LIKE '%Nuts, edible%' 
          OR commodity LIKE '%Coffee%' THEN 'Cash Crops'
        WHEN commodity LIKE '%Wheat%' 
          OR commodity LIKE '%Rice%' THEN 'Staple Cereals'
        ELSE 'Other'
    END AS commodity_sector,
        im_exp AS trade_flow,
        
        -- FACTS (Metrics)
        -- No Log-transforms here. Stakeholders want real dollars and kilos.
        primaryValue AS total_value_usd,
        net_weight_kg AS total_weight_kg,
        qty,
        
        -- PRE-CALCULATED BI METRICS
        -- Calculating this in dbt saves the BI tool from doing row-level math
        CASE 
            WHEN net_weight_kg > 0 THEN primaryValue / net_weight_kg 
            ELSE NULL 
        END AS price_per_kg,

        CASE 
            WHEN qty > 0 THEN primaryValue / qty
            ELSE NULL 
        END AS price_per_unit

    FROM filtered_silver
)

SELECT * FROM dashboard_obt
