import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\cpi-predictor')
df = pd.read_parquet(base / 'data' / 'processed' / 'cpi_dataset.parquet')

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

print(f"Original shape: {df.shape}")

# 1. CPI Index Change (YoY - Year over Year) 
df['cpi_yoy_past'] = df['cpi_index'].pct_change(periods=12) * 100
df['cpi_yoy_past'] = df['cpi_yoy_past'].shift(1)

# 2. PPI Change
df['ppi_mom'] = df['ppi_index'].pct_change(periods=1) * 100
df['ppi_yoy'] = df['ppi_index'].pct_change(periods=12) * 100
df['ppi_mom'] = df['ppi_mom'].shift(1)
df['ppi_yoy'] = df['ppi_yoy'].shift(1)

# 3. Gas Price Change (Weekly data, 4 periods = 1 month)
df['gas_mom'] = df['gas_price'].pct_change(periods=4) * 100 
df['gas_mom'] = df['gas_mom'].shift(1)

# 4. Oil Price Change (WTI Crude, 20 trading days ~ 1 month)
df['oil_mom'] = df['oil_price'].pct_change(periods=20) * 100
df['oil_mom'] = df['oil_mom'].shift(1)

# 5. Wage Growth
df['wage_yoy'] = df['wages'].pct_change(periods=12) * 100
df['wage_yoy'] = df['wage_yoy'].shift(1)

# 6. M2 Money Supply
df['m2_yoy'] = df['m2_money'].pct_change(periods=12) * 100
df['m2_yoy'] = df['m2_yoy'].shift(1)

# 7. Dollar Index
df['usd_mom'] = df['usd_index'].pct_change(periods=20) * 100
df['usd_mom'] = df['usd_mom'].shift(1)

# Drop rows with NaN (the first few years will have NaN because of shifts)
df = df.dropna().reset_index(drop=True)

# Define features to use for the model
# NOTE: consensus_gap is REMOVED
feature_cols = [
    'cpi_yoy_past', 'ppi_mom', 'ppi_yoy', 
    'gas_mom', 'oil_mom', 'wage_yoy', 
    'm2_yoy', 'usd_mom', 'treasury_10y',
    'consensus_forecast', 'fed_funds'
]

print("\n" + "="*50)
print("FEATURE ENGINEERING COMPLETE")
print("="*50)
print(f"New shape: {df.shape}")
print(f"Total features: {len(feature_cols)}")

print("\nCorrelation with Surprise:")
print(df[feature_cols + ['surprise']].corr()['surprise'].sort_values(ascending=False).to_string())

# Save
output_path = base / 'data' / 'processed' / 'cpi_features.parquet'
df.to_parquet(output_path)
print(f"\nSaved to: {output_path}")