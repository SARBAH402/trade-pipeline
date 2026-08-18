# %%
import os
import pandas as pd
from dotenv import load_dotenv
import duckdb

load_dotenv()
KEY = os.getenv("HF_S3_KEY")
SECRET = os.getenv("HF_S3_SECRET")
HF_BUCKET = os.getenv("HF_BUCKET")
MD_TOKEN = os.getenv("MD_TOKEN")
s3_path = "s3://LESSONED/comtrade-bucket/marts_dashboard.parquet"

con = duckdb.connect(f'md:?motherduck_token = {MD_TOKEN}')
print("[OK] Connected to MotherDuck Cloud.")

con.sql('INSTALL httpfs; LOAD httpfs;')

con.sql(f"""
    CREATE OR REPLACE SECRET md_secret(
        TYPE S3,
        KEY_ID '{KEY}',
        SECRET '{SECRET}',
        REGION 'us-east-1',
        ENDPOINT 's3.hf.co',
        URL_STYLE 'path'
    );
""")

df_db = con.sql(f"""CREATE OR REPLACE TABLE marts_dashboard 
                     AS SELECT * FROM '{s3_path}';
""")

print("\n[OK] Data successfully materialized inside MotherDuck!")


# %%
