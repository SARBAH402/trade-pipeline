import comtradeapicall
import pandas as pd
import io
import os
from huggingface_hub import HfApi

# 1. Credentials
comtrade_key = os.getenv("COMTRADE_API_KEY")  # Ensure this is in your env variables
hf_token = os.getenv("HF_TOKEN")              # Ensure this is in your env variables

# 2. Execute Request via Official Package (Strict 3-Month Window)
# Using getFinalData with freqCode='M' for monthly data
df_raw = comtradeapicall.getFinalData(
    comtrade_key, 
    typeCode='C',                   # Commodities
    freqCode='M',                   # Monthly frequency
    clCode='HS',                    # HS Classification
    period='202001,202002,202003',  # Constrained 3-month sample
    reporterCode='all', 
    cmdCode=['2709,7108,2606,2602,1801,0801,1001,1006,0901'], # e.g., '1001,1005'
    flowCode='M,X',                 # Imports and Exports
    partnerCode=None, 
    motCode=None, 
    customsCode=None, 
    partner2Code=None
)

# 3. The Sealed Variable Roster
required_columns = [
    'period', 'reporterCode', 'partnerCode', 'cmdCode', 
    'classificationCode', 'flowCode', 'primaryValue', 'cifvalue', 
    'fobvalue', 'netWgt', 'qty', 'qtyUnitCode', 'motCode'
]

# 4. In-Memory Processing & Enforcement
# Safety catch: If the API completely omits a column for this specific quarter, 
# this ensures the DataFrame structure remains intact for dbt downstream.
for col in required_columns:
    if col not in df_raw.columns:
        df_raw[col] = None

# Slice the dataframe to strictly our 12 variables
df_lean = df_raw[required_columns]

# 5. Zero-Disk Buffer for Hugging Face
parquet_buffer = io.BytesIO()
df_lean.to_parquet(parquet_buffer, index=False, engine='pyarrow')

print(f"Extraction complete. Shape: {df_lean.shape}")

# 6. Push directly to comtrade-bronze
parquet_buffer.seek(0) # Reset buffer pointer
api = HfApi(token=hf_token)

api.upload_file(
    path_or_fileobj=parquet_buffer,
    path_in_repo="raw/2020_Q1_validation.parquet", 
    repo_id="LESSONED/comtrade-bronze",    
    repo_type="dataset"
)

print("Validation sample successfully pushed to comtrade-bronze.")