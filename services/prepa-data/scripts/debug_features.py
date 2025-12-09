import pandas as pd
from pathlib import Path

FE_DIR = Path("datasets/oulad/fe")

df = pd.read_csv(FE_DIR / "train.csv")

print("\n===== Colonnes contenant encore des valeurs non-numériques =====")
for col in df.columns:
    bad = df[col].apply(lambda x: isinstance(x, str)).sum()
    if bad > 0:
        print(f"{col}  →  strings = {bad}")

print("\n===== Colonnes contenant des NaN =====")
print(df.isna().sum()[df.isna().sum() > 0])

print("\n===== Variance = 0 =====")
print(df[df.columns].var()[df.var() == 0])
