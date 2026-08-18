 {{config(
    materialized='table'
    )}}
 
 
 WITH trade_data AS (
    SELECT * 
    FROM 'hf://datasets/LESSONED/comtrade-bronze/data/train-00000-of-00001.parquet'
),

-- Deduplicate reference tables so each ID matches exactly ONCE
ref_reporter AS (
    SELECT DISTINCT id, text AS reporter_name 
    FROM 'hf://datasets/LESSONED/comtrade-reference/data/reporter-*.parquet'
),
ref_partner AS (
    SELECT DISTINCT id, text AS partner_name 
    FROM 'hf://datasets/LESSONED/comtrade-reference/data/partner-*.parquet'
),
ref_hs AS (
    SELECT id, text AS commodity_desc 
    FROM 'hf://datasets/LESSONED/comtrade-reference/data/hs-*.parquet'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY length(text) DESC) = 1
),
ref_qtyunit AS (
    SELECT DISTINCT id, text AS unit_desc 
    FROM 'hf://datasets/LESSONED/comtrade-reference/data/qtyunit-*.parquet'
),
ref_flow AS (
    SELECT DISTINCT id, text AS im_exp 
    FROM 'hf://datasets/LESSONED/comtrade-reference/data/flowCode-*.parquet'
),
ref_mot AS (
    SELECT DISTINCT id, text AS shippment 
    FROM 'hf://datasets/LESSONED/comtrade-reference/data/motCode-*.parquet'
),

cte AS (
    SELECT 
        t.period,
        t.cmdCode,
        hs.commodity_desc,
        t.reporterCode,
        r.reporter_name,
        t.partnerCode,
        p.partner_name,
        t.flowCode,
        f.im_exp,
        t.qty,
        t.qtyUnitCode,
        u.unit_desc,
        t.netWgt,
        t.primaryValue,
        t.cifvalue,
        t.fobvalue,
        t.motCode,
        m.shippment,
        t.classificationCode
    FROM trade_data t
    LEFT JOIN ref_reporter r ON CAST(t.reporterCode AS VARCHAR) = r.id
    LEFT JOIN ref_partner p  ON CAST(t.partnerCode AS VARCHAR) = p.id
    LEFT JOIN ref_hs hs      ON CAST(t.cmdCode AS VARCHAR) = hs.id
    LEFT JOIN ref_qtyunit u  ON CAST(t.qtyUnitCode AS VARCHAR) = u.id
    LEFT JOIN ref_flow f     ON CAST(t.flowCode AS VARCHAR) = f.id
    LEFT JOIN ref_mot m      ON CAST(t.motCode AS VARCHAR) = m.id
)

SELECT 
    period,
    cmdCode,
    regexp_replace(commodity_desc, '^[0-9A-Za-z]+ -', '') AS commodity,
    reporterCode,
    reporter_name,
    partnerCode,
    partner_name,
    flowCode,
    im_exp,
    NULLIF(qty, 0) AS qty,
    NULLIF(qtyUnitCode, -1) AS qtyUnitCode,
    NULLIF(unit_desc, 'N/A') AS unit_desc,
    nullif(netWgt, 0) AS net_weight_kg,
    primaryValue,
    NULLIF(cifvalue, 0) AS cifvalue,
    NULLIF(fobvalue, 0) AS fobvalue,
    motCode,
    shippment,
    classificationCode
FROM cte
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY period, cmdCode, reporterCode, partnerCode, flowCode, motCode, classificationCode 
    ORDER BY primaryValue DESC
) = 1
ORDER BY period ASC, primaryValue DESC