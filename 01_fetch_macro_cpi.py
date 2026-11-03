import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\cpi-predictor')
load_dotenv(base / '.env')
fred = Fred(api_key=os.getenv('FRED_API_KEY'))

series = {
    'cpi_index': 'CPIAUCSL',
    'ppi_index': 'PPIACO',
    'gas_price': 'GASREGW',
    'wages': 'CES0500000003',
    'm2_money': 'M2SL',
    'treasury_10y': 'DGS10',
    'usd_index': 'DTWEXBGS',
    'oil_price': 'DCOILWTICO',
    'fed_funds': 'FEDFUNDS'  # <--- ADDED THIS
}

data = {}
print("Pulling macro data from FRED...")
for name, code in series.items():
    print(f"  - Pulling {name}...")
    data[name] = fred.get_series(code)

df = pd.DataFrame(data).sort_index()
output_path = base / 'data' / 'raw' / 'macro_data_cpi.parquet'
df.to_parquet(output_path)
print(f"Saved {df.shape} to {output_path}")