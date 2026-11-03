import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\cpi-predictor')
df = pd.read_parquet(base / 'data' / 'processed' / 'cpi_features.parquet')
df = df.sort_values('date').reset_index(drop=True)

# Ensure this matches exactly with the feature list in script 03
feature_cols = [
    'cpi_yoy_past', 'ppi_mom', 'ppi_yoy', 
    'gas_mom', 'oil_mom', 'wage_yoy', 
    'm2_yoy', 'usd_mom', 'treasury_10y',
    'consensus_forecast', 'fed_funds'
]

X = df[feature_cols]
y_dir = df['direction']
y_surprise = df['surprise']

# --- 80/20 Time-Series Split ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_dir_train, y_dir_test = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
y_surp_train, y_surp_test = y_surprise.iloc[:split_idx], y_surprise.iloc[split_idx:]

print("=" * 70)
print("CPI DUAL-MODEL TRAINING (No Consensus Gap)")
print("=" * 70)
print(f"Train size: {len(X_train)} ({df['date'].iloc[0].date()} to {df['date'].iloc[split_idx-1].date()})")
print(f"Test size:  {len(X_test)} ({df['date'].iloc[split_idx].date()} to {df['date'].iloc[-1].date()})")

# --- MODEL A: Classification (Direction) ---
print("\n" + "-" * 70)
print("MODEL A: Classifier (Predicting HOT vs COOL)")
print("-" * 70)
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=4)
clf.fit(X_train, y_dir_train)
pred_dir = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_dir_test, pred_dir):.2%}")
print("\nClassification Report (0=Cool, 1=Hot):")
print(classification_report(y_dir_test, pred_dir, zero_division=0))

# --- MODEL B: Regression (The Number) ---
print("-" * 70)
print("MODEL B: Regressor (Predicting the Surprise Amount)")
print("-" * 70)
reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=4)
reg.fit(X_train, y_surp_train)
pred_surp = reg.predict(X_test)

mae = mean_absolute_error(y_surp_test, pred_surp)
print(f"Mean Absolute Error (MAE): {mae:.3f}%")

# --- SHOW PREDICTIONS ---
print("\n" + "=" * 70)
print("TEST SET PREDICTIONS (Recent History)")
print("=" * 70)

results = pd.DataFrame({
    'Date': df['date'].iloc[split_idx:].dt.strftime('%Y-%m-%d'),
    'Actual Surprise': y_surp_test.values,
    'Pred Surprise': np.round(pred_surp, 2),
    'Actual Dir': np.where(y_dir_test == 1, 'Hot', 'Cool'),
    'Pred Dir': np.where(pred_dir == 1, 'Hot', 'Cool'),
    'Correct?': np.where(pred_dir == y_dir_test, '✅', '❌')
})
print(results.tail(15).to_string(index=False))

# --- FEATURE IMPORTANCE ---
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "=" * 70)
print("TOP FEATURES")
print("=" * 70)
print(importance.to_string(index=False))