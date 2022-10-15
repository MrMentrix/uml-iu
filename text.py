import pandas as pd
import numpy as np

df = pd.read_csv("rn_data.csv")

print(df.shape)
print(df['mh_coverage'].isnull().sum())
print(df['primary_tech_role'].isnull().sum())