import pandas as pd
df = pd.read_parquet('final_dataset.parquet')
print(df.head())
print(f"Liczba wczytanych aut: {len(df)}")
print(df.head())