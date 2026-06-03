import pandas as pd

df = pd.read_csv("creditcard_cleaned.csv")

sample_df = df.sample(n=5000, random_state=42)

sample_df.to_csv("sample_data.csv", index=False)