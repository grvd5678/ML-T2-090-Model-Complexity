import pandas as pd
from sklearn.datasets import fetch_openml

# Fetching the traditional German Credit dataset
print("Downloading the classic credit data... 🏦")
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
df = credit_data.frame

# Old-school data inspection
print(f"\nTotal Records: {df.shape[0]}")
print(f"Total Features (Columns): {df.shape[1]}")

print("\nSneak Peek at the First 5 Rows:")
print(df.head())

print("\nClass Balance (Good vs Bad Credit):")
print(df['class'].value_counts())