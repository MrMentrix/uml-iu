import pandas as pd

df = pd.read_csv("clustered.csv")

print(df[df["mh_issues"] == 1][["mh_ph_equal", "prev_mh_ph_equal"]].value_counts().unstack())
print(df[df["mh_issues"] == 0][["mh_ph_equal", "prev_mh_ph_equal"]].value_counts().unstack())