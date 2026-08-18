import time
import comtradeapicall
import pandas as pd
from dotenv import load_dotenv
import os
from datasets import Dataset

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
COMTRADE_KEY = os.getenv("COMTRADE_KEY")

target_commodities = '2709,7108,2606,2602,1801,0801,1001,1006,0901'
reporter_batches = [
    '288,566,710,384,170',  # Batch 1: Africa & S. America
    '156,356,392,704,360',  # Batch 2: Asia
    '276,528,756,380,840',  # Batch 3: Europe & N. America
    '076,124,784,036,682'   # Batch 4: Global Bulk Exporters
]

# 1. Scope adjusted to 72-month full extraction (6 Years)
# Using 2019-01-01 as a start date to ensure 6 years of dense, published historical data.
date_range = pd.date_range(start='2018-01-01', periods=72, freq='ME')
months = date_range.strftime('%Y%m').tolist()

all_data = []
total_records_so_far = 0

print(f"Starting ingestion for {len(months)} periods (72 months) across 4 global batches...")
print(f"Estimated API calls: {len(months) * len(reporter_batches)}. This will take some time.")

for month in months:
    for batch_idx, batch in enumerate(reporter_batches, 1):
        print(f"Extracting data for {month} | Batch {batch_idx}/4")

        # Robust API Retry Logic (prevents silent data drops on a 6-year run)
        max_api_retries = 3
        for attempt in range(max_api_retries):
            try:
                df_chunck = comtradeapicall.getFinalData(
                    COMTRADE_KEY,
                    typeCode='C', freqCode='M', clCode='HS',
                    period=month, reporterCode=batch,
                    cmdCode=target_commodities, 
                    flowCode='M,X', # Includes Imports (M) and Exports (X)
                    partnerCode=None,
                    partner2Code=None, customsCode=None,
                    motCode=None,
                    format_output='JSON'
                )

                if df_chunck is not None and not df_chunck.empty:
                    all_data.append(df_chunck)
                    total_records_so_far += len(df_chunck)
                    print(f"   -> Success: {len(df_chunck)} rows. (Total so far: {total_records_so_far})")
                else:
                    print("   -> Success: 0 rows returned for this batch.")
                
                break # Break out of retry loop on success

            except Exception as e:
                print(f"   -> API attempt {attempt + 1} failed: {e}")
                if attempt < max_api_retries - 1:
                    wait_time = 15 * (attempt + 1) # Exponential backoff: 15s, 30s
                    print(f"   -> Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("   -> CRITICAL: Max retries reached for this batch. Moving to next.")

        # Rate Limiting: 4 seconds is safer than 3 for prolonged 288-call sessions
        time.sleep(4)

# 2. Concatenation (In-Memory)
if not all_data:
    print("\nCRITICAL: No data was extracted. Exiting script to prevent crash.")
    exit()

print("\nCompiling Zero-Disk Bronze Dataset...")
df_bronze = pd.concat(all_data, ignore_index=True)

# 3. The 13 required variables (including M49 codes, qty, motCode)
required_columns = [
    'period', 'reporterCode', 'partnerCode', 'cmdCode', 
    'classificationCode', 'flowCode', 'primaryValue', 'cifvalue', 
    'fobvalue', 'netWgt', 'qty', 'qtyUnitCode', 'motCode'
]

# 4. Enforce strict schema: Add missing columns if API omitted them in this batch
for col in required_columns:
    if col not in df_bronze.columns:
        df_bronze[col] = None

# Slice to exact schema
df_lean = df_bronze[required_columns].copy()

# 5. Direct-to-Cloud Push (Zero-Disk)
hf_dataset = Dataset.from_pandas(df_lean)

print(f"\nPushing {len(df_lean)} records to Hugging Face...")
max_hf_retries = 3
for attempt in range(max_hf_retries):
    try:
        # Pushing straight to your private repo
        hf_dataset.push_to_hub("LESSONED/comtrade-bronze", token=HF_TOKEN, private=True)
        print(f"\nSuccess! {len(df_lean)} records securely loaded to the cloud.")
        break
        
    except Exception as e:
        print(f"\nUpload attempt {attempt + 1} failed due to network timeout: {e}")
        if attempt < max_hf_retries - 1:
            print("Retrying in 15 seconds...")
            time.sleep(15)
        else:
            print(f"\nCRITICAL: All upload attempts failed.")

