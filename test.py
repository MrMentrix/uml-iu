import pandas as pd

df = pd.read_csv("./compare.csv")

features = ["male_perc", "female_perc", "divers_perc"]

print(df["issue_score"])