"""
train_model.py
--------------
Trains a RandomForest risk-score model using synthetic Chicago crime data.
In production, replace generate_synthetic_crime_data() with your real
Kaggle dataset (Latitude, Longitude, hour, day, month, risk_score).

Usage:
    python train_model.py
Output:
    crime_model.pkl
    crime_data.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Generate synthetic Chicago crime data ──────────────────────────────────

def generate_synthetic_crime_data(n=10000):
    np.random.seed(42)

    # Chicago bounding box
    lat_min, lat_max = 41.644, 42.023
    lon_min, lon_max = -87.940, -87.524

    # High-crime clusters (South/West side Chicago)
    cluster_centers = [
        (41.748, -87.700, 0.85),   # Englewood
        (41.776, -87.660, 0.80),   # Greater Grand Crossing
        (41.856, -87.758, 0.78),   # Austin
        (41.796, -87.625, 0.72),   # Woodlawn
        (41.840, -87.706, 0.75),   # North Lawndale
        (41.893, -87.634, 0.55),   # Near West Side
        (41.950, -87.655, 0.45),   # Humboldt Park
        (41.870, -87.627, 0.40),   # Loop fringe
    ]

    rows = []
    per_cluster = n // (len(cluster_centers) + 1)

    for clat, clon, base_risk in cluster_centers:
        for _ in range(per_cluster):
            lat   = np.random.normal(clat, 0.018)
            lon   = np.random.normal(clon, 0.018)
            hour  = np.random.randint(0, 24)
            day   = np.random.randint(1, 8)
            month = np.random.randint(1, 13)
            time_factor    = 0.15 if hour in range(21,24) or hour in range(0,5) else 0.0
            weekend_factor = 0.05 if day in (6,7) else 0.0
            risk = min(1.0, max(0.0, base_risk + time_factor + weekend_factor
                                + np.random.normal(0, 0.06)))
            rows.append([lat, lon, hour, day, month, risk])

    remaining = n - len(rows)
    for _ in range(remaining):
        lat   = np.random.uniform(lat_min, lat_max)
        lon   = np.random.uniform(lon_min, lon_max)
        hour  = np.random.randint(0, 24)
        day   = np.random.randint(1, 8)
        month = np.random.randint(1, 13)
        risk  = np.random.uniform(0.05, 0.35)
        rows.append([lat, lon, hour, day, month, risk])

    return pd.DataFrame(rows, columns=["Latitude","Longitude","hour","day","month","risk_score"])


# ── 2. Train ──────────────────────────────────────────────────────────────────

def train(df):
    X = df[["Latitude","Longitude","hour","day","month"]].values
    y = df["risk_score"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=120, max_depth=10, min_samples_leaf=4,
        n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print(f"  MAE : {mean_absolute_error(y_te, preds):.4f}")
    print(f"  R²  : {r2_score(y_te, preds):.4f}")
    return model


# ── 3. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SafeRoute AI — Chicago Crime Model Trainer")
    print("=" * 55)

    print("\nGenerating synthetic Chicago crime data …")
    df = generate_synthetic_crime_data(10000)
    print(f"  Dataset shape : {df.shape}")

    print("\nTraining RandomForest …")
    model = train(df)

    joblib.dump(model, "crime_model.pkl")
    print("Saved → crime_model.pkl")

    heatmap = df[["Latitude","Longitude","risk_score"]].sample(
        n=3000, random_state=42
    ).values.tolist()
    joblib.dump(heatmap, "crime_data.pkl")
    print(f"Saved → crime_data.pkl  ({len(heatmap)} heatmap points)")

    print("\nSanity predictions:")
    tests = [
        (41.748, -87.700, 22, 6, 7,  "Englewood – Saturday night"),
        (41.878, -87.629, 14,  2, 4, "Loop area – Tuesday afternoon"),
        (41.950, -87.655, 10,  3, 5, "Humboldt – Wednesday morning"),
    ]
    for lat, lon, h, d, m, label in tests:
        risk = model.predict(np.array([[lat, lon, h, d, m]]))[0]
        print(f"  {label:40s} → risk = {risk:.3f}")

    print("\nDone! Run:  uvicorn main:app --reload")
