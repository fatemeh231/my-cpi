import pandas as pd
import re
from pathlib import Path

# Paths
base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\cpi-predictor')
raw_path = base / 'data' / 'raw' / 'cpi_raw.txt'
output_path = base / 'data' / 'raw' / 'cpi_dates.csv'

# Read the raw text
with open(raw_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = []

for line in lines:
    if 'Release date' in line or not line.strip():
        continue

    parts = line.strip().split('\t')
    
    if len(parts) >= 4:
        date_str = parts[0].strip()
        actual_str = parts[2].strip()
        forecast_str = parts[3].strip()

        # Clean Date: Extract just the "Oct 14, 2026" part and convert to YYYY-MM-DD
        date_match = re.search(r'([A-Z][a-z]{2} \d{1,2}, \d{4})', date_str)
        if date_match:
            clean_date = pd.to_datetime(date_match.group(1)).strftime('%Y-%m-%d')
        else:
            continue

        # Clean Numbers
        def clean_float(val):
            val = val.replace('%', '').strip()
            return float(val) if val else None

        actual = clean_float(actual_str)
        forecast = clean_float(forecast_str)

        # ONLY add if we have BOTH actual and forecast
        if actual is not None and forecast is not None:
            data.append({
                'date': clean_date,
                'actual_cpi': actual,
                'consensus_forecast': forecast
            })

df = pd.DataFrame(data)

# Calculate Surprise and Direction
df['surprise'] = df['actual_cpi'] - df['consensus_forecast']
df['direction'] = (df['surprise'] > 0).astype(int)

# Sort chronologically
df = df.sort_values('date').reset_index(drop=True)

# Save
df.to_csv(output_path, index=False)

print("=" * 50)
print("CPI DATA CLEANED SUCCESSFULLY")
print("=" * 50)
print(f"Total rows: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
print("\nLast 3 rows:")
print(df.tail(3).to_string(index=False))
print(f"\nSaved to: {output_path}")