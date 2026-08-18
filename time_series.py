# %%
import os
import pandas as pd
from dotenv import load_dotenv
import duckdb

load_dotenv()
KEY = os.getenv("HF_S3_KEY")
SECRET = os.getenv("HF_S3_SECRET")
HF_BUCKET = os.getenv("HF_BUCKET")
s3_path = "s3://LESSONED/comtrade-bucket/marts_timeseries.parquet"

con = duckdb.connect(database=":memory:")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

con.execute(
    f"""
    CREATE SECRET hf_s3(
    TYPE S3,
    KEY_ID '{KEY}',
    SECRET '{SECRET}',
    REGION 'us-east-1',
    URL_STYLE 'path',
    ENDPOINT 's3.hf.co'
    );
    """
)

df_ts = con.execute(f"SELECT * FROM '{s3_path}'").df()

print("DATA LOADED SUCCESSFULLY")
print(df_ts.head())

# %%
print(df_ts.info())
print(df_ts.describe())
# %%
print("--- PHASE 1: TEMPORAL DIAGNOSTICS ---\n")

# 1. Global Timeline Check
# Group by date to see if there are any missing months in the global dataset
global_timeline = df_ts.groupby('trade_date')['total_value_usd'].sum()
expected_months = pd.date_range(start=global_timeline.index.min(), end=global_timeline.index.max(), freq='MS')

missing_months = expected_months.difference(global_timeline.index)
if len(missing_months) == 0:
    print("[OK] Global timeline is perfectly continuous (No missing months).")
else:
    print(f"[WARNING] Missing global months detected: {missing_months}")

# 2. Find the Heavyweight Corridors (Zero Sparsity)
# We want routes that have exactly 71 months of data (Feb 2018 - Dec 2023) and massive volume
route_stats = df_ts.groupby(['reporter_name', 'partner_name']).agg(
    total_volume=('total_value_usd', 'sum'),
    month_count=('trade_date', 'count'),
    zero_count=('total_value_usd', lambda x: (x == 0).sum())
)

# Filter for routes that have data for every single month, and absolutely NO zeroes
perfect_routes = route_stats[(route_stats['month_count'] == len(expected_months)) & (route_stats['zero_count'] == 0)]

# Sort by sheer trade volume to find the biggest continuous supply chains
top_corridors = perfect_routes.sort_values(by='total_volume', ascending=False).head(3)

print("\n--- TOP 3 HEAVYWEIGHT CORRIDORS FOR FORECASTING ---")
print(top_corridors)
# %%
print("--- ISOLATING THE FORECASTING ROUTE ---\n")

# 1. Filter for only the Canada-USA corridor
df_ts = df_ts[(df_ts['reporter_name'] == 'Canada') & (df_ts['partner_name'] == 'USA')].copy()

# 2. Sort chronologically (just in case)
df_ts = df_ts.sort_values('trade_date')

# 3. Set the trade_date as the index so time-series algorithms can read it
df_ts = df_ts.set_index('trade_date')

print(f"Dataset reduced from 112,393 rows to exactly {len(df_ts)} rows.")
print("The data is now a perfect, single continuous time series.")
print("\nFirst 3 months of our isolated forecasting data:")
print(df_ts['total_value_usd'].head(3))
# %%
print("--- ISOLATING & SPLITTING THE FORECASTING ROUTE ---\n")

# 1. Filter for only the Canada-USA corridor
df_ts = df_ts[(df_ts['reporter_name'] == 'Canada') & (df_ts['partner_name'] == 'USA')].copy()

# 2. Foolproof Index Setting
if 'trade_date' in df_ts.columns:
    df_ts = df_ts.sort_values('trade_date')
    df_ts = df_ts.set_index('trade_date')
else:
    df_ts = df_ts.sort_index() # It's already the index, just make sure it's sorted

print(f"[OK] Canada-USA isolated. Total rows: {len(df_ts)}")
# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 3. Chronological Train/Test Split
train_df = df_ts[df_ts.index < '2023-01-01'].copy()
test_df = df_ts[df_ts.index >= '2023-01-01'].copy()
print(f"[OK] Training Set: {len(train_df)} months | Testing Set: {len(test_df)} months\n")

print("--- PHASE 2: ALGORITHM SHOWDOWN ---\n")
results = {}
# %%
import pmdarima as pm

# --- A. SARIMAX ---
print("Training SARIMAX (Auto-ARIMA)...")
y_train, y_test = train_df['total_value_usd'], test_df['total_value_usd']
sarimax_model = pm.auto_arima(y_train, seasonal=True, m=12, suppress_warnings=True, stepwise=True)
forecast_sari = sarimax_model.predict(n_periods=len(y_test))
mape_sari = np.mean(np.abs(y_test - forecast_sari) / y_test) * 100
results['SARIMAX'] = {'model': sarimax_model, 'mape': mape_sari}
print(f"SARIMAX MAPE:  {mape_sari:.2f}%")
# %%
from prophet import Prophet

# --- B. PROPHET ---
print("Training Prophet...")
prophet_train = train_df.reset_index()[['trade_date', 'total_value_usd']].rename(columns={'trade_date': 'ds', 'total_value_usd': 'y'})
prophet_test = test_df.reset_index()[['trade_date', 'total_value_usd']].rename(columns={'trade_date': 'ds', 'total_value_usd': 'y'})

model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model_prophet.fit(prophet_train)
forecast_prophet = model_prophet.predict(prophet_test[['ds']])
mape_prophet = np.mean(np.abs(prophet_test['y'].values - forecast_prophet['yhat'].values) / prophet_test['y'].values) * 100
results['Prophet'] = {'model': model_prophet, 'mape': mape_prophet}
print(f"Prophet MAPE:  {mape_prophet:.2f}%")

# %%
import xgboost as xgb
import lightgbm as lgb

# --- C. XGBOOST & LIGHTGBM ---
print("Training XGBoost & LightGBM...")
features = ['lag_1_value', 'predictive_3mo_lag_avg']
X_train, X_test = train_df[features], test_df[features]

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
mape_xgb = np.mean(np.abs(y_test - xgb_model.predict(X_test)) / y_test) * 100
results['XGBoost'] = {'model': xgb_model, 'mape': mape_xgb}
print(f"XGBoost MAPE:  {mape_xgb:.2f}%")

lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
mape_lgb = np.mean(np.abs(y_test - lgb_model.predict(X_test)) / y_test) * 100
results['LightGBM'] = {'model': lgb_model, 'mape': mape_lgb}
print(f"LightGBM MAPE: {mape_lgb:.2f}%\n")
# %%
import io
import joblib
from huggingface_hub import HfApi
import json
from prophet.serialize import model_to_json

# --- PHASE 3: CROWN THE CHAMPION & ZERO-DISK UPLOAD ---
print("--- INITIATING CLOUD DEPLOYMENT ---")

best_model_name = min(results, key=lambda k: results[k]['mape'])
best_model = results[best_model_name]['model']
print(f"🏆 Champion Model: {best_model_name} (MAPE: {results[best_model_name]['mape']:.2f}%)")

# Initialize Hugging Face API
HF_REPO_ID = "YourUsername/Your-Repo-Name"
api = HfApi()

# 1. Dynamic Model Serialization
print(f"Streaming {best_model_name} Model to Hugging Face...")
if best_model_name == 'Prophet':
    # Native JSON serialization for Prophet
    model_json = model_to_json(best_model)
    model_buffer = io.BytesIO(model_json.encode('utf-8'))
    filename = "trade_forecast_model.json"
else:
    # Joblib serialization for SARIMAX, XGBoost, or LightGBM
    model_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)
    filename = "trade_forecast_model.joblib"

model_buffer.seek(0)
api.upload_file(
    path_or_fileobj=model_buffer,
    path_in_repo=filename,
    repo_id=HF_REPO_ID,
    repo_type="model"
)

# 2. Export the Time-Series Dataframe 
print("Streaming Time-Series dataset to Hugging Face...")
parquet_buffer = io.BytesIO()
# Reset index to guarantee the trade_date survives Parquet compression
export_df = df_ts.reset_index()
export_df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
parquet_buffer.seek(0)

api.upload_file(
    path_or_fileobj=parquet_buffer,
    path_in_repo="trade_timeseries.parquet",
    repo_id=HF_REPO_ID,
    repo_type="model"
)

print("✅ Time-Series artifacts successfully deployed to the Hub.")