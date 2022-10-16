import pandas as pd
import numpy as np

df = pd.read_csv("rn_data.csv")

# print number of missing values for each column
for column in df.columns:
    print("Number of missing values in column", column, ":", df[column].isnull().sum())