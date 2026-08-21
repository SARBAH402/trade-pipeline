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
con.sql("INSTALL httpfs;")
con.sql("LOAD httpfs;")

con.sql(
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

df_ts = con.sql(f"SELECT * FROM '{s3_path}'").df()

print("DATA LOADED SUCCESSFULLY")
print(df_ts.head())

# %%
print(df_ts.info())
print(df_ts.describe())
# %%
import numpy as np
import pandas as pd

# =====================================================================
# --- PHASE 1: TEMPORAL DIAGNOSTICS & CORRIDOR ISOLATION ---
# =====================================================================
print("--- PHASE 1: TEMPORAL DIAGNOSTICS ---\n")

# 1. Global Timeline Check
global_timeline = df_ts.groupby('trade_date')['total_value_usd'].sum()
expected_months = pd.date_range(start=global_timeline.index.min(), end=global_timeline.index.max(), freq='MS')

missing_months = expected_months.difference(global_timeline.index)
if len(missing_months) == 0:
    print("[OK] Global timeline is perfectly continuous (No missing months).")
else:
    print(f"[WARNING] Missing global months detected: {missing_months}")

# 2. Find the Heavyweight Corridors (Zero Sparsity)
route_stats = df_ts.groupby(['reporter_name', 'partner_name']).agg(
    total_volume=('total_value_usd', 'sum'),
    month_count=('trade_date', 'count'),
    zero_count=('total_value_usd', lambda x: (x == 0).sum())
)

perfect_routes = route_stats[(route_stats['month_count'] == len(expected_months)) & (route_stats['zero_count'] == 0)]
top_corridors = perfect_routes.sort_values(by='total_volume', ascending=False).head(3)

print("\n--- TOP 3 HEAVYWEIGHT CORRIDORS FOR FORECASTING ---")
print(top_corridors)
# %%
# 3. Isolate the Forecasting Route (Canada-USA)
print("\n--- ISOLATING THE FORECASTING ROUTE ---")
df_ts = df_ts[(df_ts['reporter_name'] == 'Canada') & (df_ts['partner_name'] == 'USA')].copy()

# Foolproof Index Setting
if 'trade_date' in df_ts.columns:
    df_ts = df_ts.sort_values('trade_date').set_index('trade_date')
else:
    df_ts = df_ts.sort_index()

print(f"[OK] Canada-USA isolated. Total rows: {len(df_ts)}")
print(f"First 3 months of forecasting data:\n{df_ts['total_value_usd'].head(3)}\n")
# %%
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================================
# --- PHASE 2: ALGORITHM SHOWDOWN (TWO-TRACK ARCHITECTURE) ---
# =====================================================================
print("--- PHASE 2: ALGORITHM SHOWDOWN ---\n")

# Chronological Train/Test Split
train_df = df_ts[df_ts.index < '2023-01-01'].copy()
test_df = df_ts[df_ts.index >= '2023-01-01'].copy()
print(f"[OK] Training Set: {len(train_df)} months | Testing Set: {len(test_df)} months\n")

results = {}
y_train, y_test = train_df['total_value_usd'], test_df['total_value_usd']

# --- A. SARIMAX (Classical Track) ---
print("Training SARIMAX (Auto-ARIMA)...")
sarimax_model = pm.auto_arima(y_train, seasonal=True, m=12, suppress_warnings=True, stepwise=True)
forecast_sari = sarimax_model.predict(n_periods=len(y_test))
mape_sari = np.mean(np.abs(y_test - forecast_sari) / y_test) * 100

results['SARIMAX'] = {'model': sarimax_model, 'mape': mape_sari}
print(f"SARIMAX MAPE:  {mape_sari:.2f}%")
# %%
from prophet import Prophet

# --- B. PROPHET (Classical Track) ---
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

# --- C. XGBOOST & LIGHTGBM (Machine Learning Track) ---
print("Training XGBoost & LightGBM...")
features = ['lag_1_value', 'lag_12_value', 'predictive_3mo_lag_avg', 'predictive_6mo_lag_avg', 'rolling_6mo']
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
# =====================================================================
# --- PHASE 3: CROWN THE CHAMPION, RETRAIN, & ZERO-DISK UPLOAD ---
# =====================================================================
print("--- INITIATING PRODUCTION RETRAINING & CLOUD DEPLOYMENT ---")

# Dynamic Champion Selection
best_model_name = min(results, key=lambda k: results[k]['mape'])
best_model = results[best_model_name]['model']
print(f"🏆 Champion Model: {best_model_name} (Test Set MAPE: {results[best_model_name]['mape']:.2f}%)")

# 1. Retrain the Champion on the ENTIRE Dataset
print(f"Retraining {best_model_name} on the entire historical dataset for future forecasting...")
y_full = df_ts['total_value_usd']

if best_model_name == 'SARIMAX':
    best_model.fit(y_full)

elif best_model_name == 'Prophet':
    prophet_full = df_ts.reset_index()[['trade_date', 'total_value_usd']].rename(
        columns={'trade_date': 'ds', 'total_value_usd': 'y'}
    )
    best_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    best_model.fit(prophet_full)

elif best_model_name in ['XGBoost', 'LightGBM']:
    X_full = df_ts[features]
    best_model.fit(X_full, y_full)

print("[OK] Production retraining complete.")
# %%
import io
import json
import joblib
from prophet.serialize import model_to_json
from huggingface_hub import HfApi

# 2. Initialize Hugging Face API
HF_REPO_ID = "LESSONED/comtrade-bucket"
api = HfApi()

# 3. Dynamic Model Serialization
print(f"Streaming {best_model_name} Model to Hugging Face...")
if best_model_name == 'Prophet':
    model_json = model_to_json(best_model)
    model_buffer = io.BytesIO(model_json.encode('utf-8'))
    filename = "trade_forecast_model.json"
else:
    model_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)
    filename = f"trade_forecast_model_{best_model_name}.joblib"

model_buffer.seek(0)
api.upload_file(
    path_or_fileobj=model_buffer,
    path_in_repo=filename,
    repo_id=HF_REPO_ID,
    repo_type="model"
)

# 4. Export the Time-Series Dataframe 
print("Streaming Time-Series dataset to Hugging Face...")
parquet_buffer = io.BytesIO()
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
