import pandas as pd
from pathlib import Path

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\cpi-predictor')
cpi_dates = pd.read_csv(base / 'data' / 'raw' / 'cpi_dates.csv', parse_dates=['date'])
macro = pd.read_parquet(base / 'data' / 'raw' / 'macro_data_cpi.parquet')

macro.index = pd.to_datetime(macro.index)
macro = macro.sort_index().ffill()
cpi_dates = cpi_dates.sort_values('date')

merged = pd.merge_asof(cpi_dates, macro, left_on='date', right_index=True, direction='backward')

output_path = base / 'data' / 'processed' / 'cpi_dataset.parquet'
merged.to_parquet(output_path)
print(f"Merged shape: {merged.shape}")
print(merged[['date', 'fed_funds']].tail(3))