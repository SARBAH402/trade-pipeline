# %%
import duckdb
import pandas as pd 
from dotenv import load_dotenv
import os

load_dotenv()
HF_BUCKET = os.getenv("HF_BUCKET")
HF_S3_SECRET = os.getenv("HF_S3_SECRET")
HF_S3_KEY = os.getenv("HF_S3_KEY")

s3_path = "s3://LESSONED/comtrade-bucket/marts_regression.parquet"

con = duckdb.connect(':memory:')
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

con.execute(f"""
            CREATE SECRET s3_hf(
                TYPE S3,
                KEY_ID '{HF_S3_KEY}',
                SECRET '{HF_S3_SECRET}',
                ENDPOINT 's3.hf.co',
                REGION 'us-east-1',
                URL_STYLE 'path');

""")

df_reg = con.execute(f"SELECT * FROM '{s3_path}'").df()

print("Data Loaded Successfully")
print(df_reg.head())
print(df_reg.describe())
print(df_reg.info())
print(df_reg.nunique())
# %%
print(df_reg['shippment'].unique())

# Define the categorical buckets
land_modes = ['Road', 'Railway', 'Land']

special_modes = ['Pipelines and cables', 
'Pipelines', 'Self propelled goods', 
'Other', 'Not elsewhere classified']

air_modes = ['Air', 'Postal consignments, mail or courier shipment']

# Extract numerical indicators
df_reg['is_land_freight'] = df_reg['shippment'].isin(land_modes).astype(int)
df_reg['is_special_freight'] = df_reg['shippment'].isin(special_modes).astype(int)
df_reg['is_air_freight'] = df_reg['shippment'].isin(air_modes).astype(int) 

# Drop the raw text column
df_reg = df_reg.drop(columns=['shippment'])
# %%
import numpy as np

df_reg = df_reg.drop(columns=['air_freight_pct'])

df_reg['log_wght_kg'] = np.log1p(df_reg['net_weight_kg'])
df_reg['log_qty'] = np.log1p(df_reg['qty'])
# %%
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, FunctionTransformer

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(2 * np.pi * x / period),
                               feature_names_out='one-to-one')

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(2 * np.pi * x / period),
                               feature_names_out='one-to-one')

numeric_features = ['log_wght_kg', 'log_qty', 'is_land_freight', 'is_special_freight', 'is_air_freight']
ohe_features = ['im_exp']
target_enc_features = ['reporter_name', 'partner_name', 'commodity']
cyclical_feature = ['month']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ohe_transformer = Pipeline([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False))
])

target_transformer = Pipeline([
    ('target_enc', TargetEncoder(target_type='continuous'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ohe', ohe_transformer, ohe_features),
        ('target', target_transformer, target_enc_features),
        ('month_sin', sin_transformer(12), cyclical_feature),
        ('month_cos', cos_transformer(12), cyclical_feature)
    ],
    remainder='drop'
)
# %%
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

X = df_reg.drop(columns=['raw_target_value_usd', 'log_target_value'])
y = df_reg['log_target_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv_fold = KFold(n_splits=5, shuffle=True, random_state=42)

lr = LinearRegression()
lr_params = {}

ridge = Ridge()
ridge_params = {
    'model__alpha':[0.1, 1.0, 10.0, 100.0]}

rf = RandomForestRegressor(random_state=42)
rf_params ={
    'model__n_estimators': [100],
    'model__max_depth': [10, 20],
    'model__min_samples_split': [5, 10],
    'model__max_features': ['sqrt']
}

xgb = XGBRegressor(random_state = 42)
xgb_params = {
    'model__n_estimators': [100],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [5, 8],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8]
}

models = {
    'model__Linear Regression': (lr, lr_params),
    'model__Ridge Regression': (ridge, ridge_params),
    'model__Random Forest': (rf, rf_params),
    'model__XGBoost': (xgb, xgb_params)
}

best_models = {}

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid= params,
        cv= cv_fold,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_
    print(f"Best Score (R2): {grid.best_score_:.4f}")
    print(f"Best Parameters: {grid.best_params_}")

# %%
import io
import joblib
from huggingface_hub import HfApi

# --- EVALUATION BLOCK ---
print("\n--- Final Test Set Evaluation ---")
test_results = {}

for name, model_pipe in best_models.items():
    y_pred = model_pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    test_results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
    print(f"{name} -> R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# =====================================================================
# --- PHASE 3: CROWN THE CHAMPION, RETRAIN, & ZERO-DISK UPLOAD ---
# =====================================================================
print("\n--- INITIATING PRODUCTION RETRAINING & CLOUD DEPLOYMENT ---")

# Dynamically find the best model based on the highest R2 score
best_model_name = max(test_results, key=lambda k: test_results[k]['R2'])
best_model_pipeline = best_models[best_model_name]

print(f"🏆 Champion Model: {best_model_name} (Test Set R2: {test_results[best_model_name]['R2']:.4f})")
print(f"Retraining {best_model_name} on the complete dataset for production...")

# Combine the splits back into full datasets
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

# Fit the entire pipeline on 100% of the data
best_model_pipeline.fit(X_full, y_full)

print("[OK] Production retraining complete.")

# 2. Initialize Hugging Face API
HF_REPO_ID = "YourUsername/Your-Repo-Name"
api = HfApi()

# 3. Dynamic Model Serialization
print(f"Streaming {best_model_name} Model to Hugging Face...")
model_buffer = io.BytesIO()
joblib.dump(best_model, model_buffer)
model_buffer.seek(0)

api.upload_file(
    path_or_fileobj=model_buffer,
    path_in_repo="trade_pricing_model.joblib",
    repo_id=HF_REPO_ID,
    repo_type="model"
)

# 4. (Optional) Export the combined feature set if needed for future dashboarding
print("Streaming feature dataset to Hugging Face...")
parquet_buffer = io.BytesIO()
X_full.to_parquet(parquet_buffer, index=False, engine='pyarrow')
parquet_buffer.seek(0)

api.upload_file(
    path_or_fileobj=parquet_buffer,
    path_in_repo="trade_pricing_features.parquet",
    repo_id=HF_REPO_ID,
    repo_type="model"
)

print("✅ Regression artifacts successfully deployed to the Hub.")