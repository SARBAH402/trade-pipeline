# %%
import os
import duckdb
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
HF_BUCKET = os.getenv("HF_BUCKET")
KEY = os.getenv("HF_S3_KEY")
SECRET = os.getenv("HF_S3_SECRET")
s3_path = "s3://LESSONED/comtrade-bucket/marts_clustering.parquet"

con = duckdb.connect(database=':memory:')
con.sql('INSTALL httpfs;')
con.sql('LOAD httpfs;')

con.sql(f"""
    CREATE SECRET hf_s3(
    TYPE S3,
    KEY_ID '{KEY}',
    SECRET '{SECRET}',
    ENDPOINT 's3.hf.co',
    URL_STYLE 'path',
    REGION 'us-east-1'
    )
""")
df_clust = con.sql(f"SELECT * FROM '{s3_path}'").df()
print(df_clust.head())
# %%
print(df_clust.describe())
print(df_clust.info())

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("--- PHASE 1: DATA PREPARATION ---\n")

# 1. Protect the identifiers (Move them to the index)
df_clust = df_clust.set_index(['reporter_name', 'partner_name'])

# 2. Fix the geometry (Log the price)
df_clust['log_price_per_kg'] = np.log1p(df_clust['price_per_kg'])

# 3. Prevent double-weighting (Drop the raw price)
pure_features = df_clust.drop(columns=['price_per_kg'])

print("Final 5 Features for Clustering:")
print(pure_features.columns.tolist())
print("\n")

# 4. Standardize the scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pure_features)

# Wrap it in a DataFrame just to keep it clean
X_scaled_df = pd.DataFrame(X_scaled, columns=pure_features.columns, index=pure_features.index)
print("[OK] Data successfully indexed, logged, and scaled.")
# %%
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

K = 4
print("--- PHASE 2: ALGORITHM SHOWDOWN ---\n")

# 1. K-MEANS (The Geometric Baseline)
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
print(f"K-Means Score:       {kmeans_score:.4f}")

# 2. AGGLOMERATIVE (The Hierarchical Challenger)
agglo = AgglomerativeClustering(n_clusters=K)
agglo_labels = agglo.fit_predict(X_scaled)
agglo_score = silhouette_score(X_scaled, agglo_labels)
print(f"Agglomerative Score: {agglo_score:.4f}")

# 3. GMM (The Probabilistic Challenger)
gmm = GaussianMixture(n_components=K, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_score = silhouette_score(X_scaled, gmm_labels)
print(f"GMM Score:           {gmm_score:.4f}")

# 4. HDBSCAN (The Density Challenger)
# HDBSCAN finds its own clusters, so we don't pass K=4. 
# We set a minimum cluster size (e.g., 15 routes to form a viable trade corridor).
hdb = HDBSCAN(min_cluster_size=15)
hdb_labels = hdb.fit_predict(X_scaled)

# To score HDBSCAN fairly, we must exclude the "Noise" points (-1) from the silhouette calculation
valid_hdb_mask = hdb_labels != -1
if len(set(hdb_labels[valid_hdb_mask])) > 1: # Ensure it found at least 2 valid clusters
    hdb_score = silhouette_score(X_scaled[valid_hdb_mask], hdb_labels[valid_hdb_mask])
    noise_count = list(hdb_labels).count(-1)
    print(f"HDBSCAN Score:       {hdb_score:.4f} (Note: Classified {noise_count} routes as noise)")
else:
    print("HDBSCAN Score:       N/A (Failed to find multiple distinct dense clusters)")
# %%
from sklearn.pipeline import Pipeline


print("--- BUILDING THE CHAMPION MODEL ---\n")

# Wrap the scaler and model into a single pipeline for seamless deployment
champion_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10))
])

# Fit and predict directly on the pure unscaled features (Pipeline handles scaling)
df_clust['final_cluster'] = champion_pipe.fit_predict(pure_features)

final_profiles = df_clust.groupby('final_cluster').mean()
final_profiles['route_count'] = df_clust['final_cluster'].value_counts()
display_profiles = final_profiles.drop(columns=['log_price_per_kg'], errors='ignore')

print("--- FINAL MACROECONOMIC PROFILES ---")
print(display_profiles.round(2).T)
print("\n")

# --- PREMIUM ROUTE EXTRACTION ---
premium_cluster_id = df_clust.groupby('final_cluster')['price_per_kg'].mean().idxmax()
premium_routes = df_clust[df_clust['final_cluster'] == premium_cluster_id].copy()
premium_routes_sorted = premium_routes.sort_values(by='price_per_kg', ascending=False)
# %%
import io
import joblib
from huggingface_hub import HfApi

# --- ZERO-DISK HUGGING FACE UPLOAD BLOCK ---
print("\n--- INITIATING CLOUD DEPLOYMENT ---")

# TODO: Add your repo details here
HF_REPO_ID = "LESSONED/comtrade-bucket"
api = HfApi()

# 1. Export the Pipeline (Model + Scaler)
print("Streaming Clustering Pipeline to Hugging Face...")
cluster_buffer = io.BytesIO()
joblib.dump(champion_pipe, cluster_buffer)
cluster_buffer.seek(0)

api.upload_file(
    path_or_fileobj=cluster_buffer,
    path_in_repo="trade_clustering_pipeline.joblib",
    repo_id=HF_REPO_ID,
    repo_type="model"
)

# 2. Export the Dataframe (Resetting index so Parquet stays flat)
print("Streaming clustered dataset to Hugging Face...")
parquet_buffer = io.BytesIO()
df_export = df_clust.reset_index()
df_export.to_parquet(parquet_buffer, index=False, engine='pyarrow')
parquet_buffer.seek(0)

api.upload_file(
    path_or_fileobj=parquet_buffer,
    path_in_repo="trade_clusters.parquet",
    repo_id=HF_REPO_ID,
    repo_type="model"
)

print("✅ Clustering artifacts successfully deployed to the Hub.")