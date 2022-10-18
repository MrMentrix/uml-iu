import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import PercentFormatter
import random

"""
DISCLAIMER:
The idea behind this was to create a visual grid that showed the percentage of mental issues based on specific features, also using PCA.
I stopped developing this system after a while, since almost all features I picked had 60% or more people with mental illness, making the color coding off green-yellow-orange-red
pretty hard to apply. When using 25%-intervals, almost all squares would have been either orange or red. Also, with too small sample sizes, speaking of sometimes just 1-5 people, 
the calculated percentage just wasn't very reliable. Therefore, I stopped developing this visualization approach.
You can take a look at "grouping.png" in the features directory to see what this was supposed to look like.

This code wasn't used to produce any of the visualizations in the final written assignment.
"""

# note: depending on what shell/ide you use to run this code, you may need to adjust the paths to the data files

"""First, we will start with some heavy feature reduction. We will use PCA to reduce the dimensionality of the data."""

df = pd.read_csv('./preprocessed.csv')
colors = json.load(open("./visualization/colors.json"))

def pca_reduction(df, n=6):
    """Insert a dataframe and the amount of n primary features that will be returned as a new dataframe."""

    # normalizing the df to a range of [-1, 1]
    normalized_df = StandardScaler().fit_transform(df)
    covariance = np.cov(normalized_df.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    # select the n eigenvectors with the highest eigenvalues
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1][:n]]

    # select the n features from the original dataset
    df = df.iloc[:, np.argsort(eigenvalues)[::-1][:n]]
    return df

# replace the values of df["gender"] with 0 for male, 1 for female and 2 for diverse
df["gender"] = df["gender"].apply(lambda x: 0 if x == "male" else 1 if x == "female" else 2)

pca_df = df.drop("mh_issues", axis=1)
new_df = pca_reduction(pca_df, 2)

# copy all mh_issues values of df to new_df
df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 80], labels=['19-25', '26-35', '36-45', '46+'])

new_df = new_df.assign(mh_issues=df["mh_issues"])
new_df = df[["gender", "age_group", "mh_issues"]]
features = new_df.columns

# get the value ranges of the new_df
x_u = new_df[features[0]].unique()
y_u = new_df[features[1]].unique()
x = len(x_u)
y = len(y_u)

# create a x * y figure
fig, ax = plt.subplots()
for i in range(x):
    for j in range(y):
        # get all rows which have the value i in the first column and the value j in the second column
        temp_df = new_df[(new_df[features[0]] == x_u[i]) & (new_df[features[1]] == y_u[j])]

        # get amount of mh_issues == 1 in temp_df
        length = len(temp_df)
        if length == 0:
            percentage = 0
        else:
            percentage = (len(temp_df[temp_df["mh_issues"] == 1]) / length) * 100

        if percentage < 25:
            color = colors["green"]
        elif percentage >= 25 and percentage < 50:
            color = colors["yellow"]
        elif percentage >= 50 and percentage < 75:
            color = colors["orange"]
        else:
            color = colors["red"]

        ax.add_patch(plt.Rectangle((i, j), i+1, j+1, color=color, alpha=percentage/100))

# edit axis labels
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])

# edit axis ticks
ax.set_xticks(np.arange(x))
ax.set_yticks(np.arange(y))

# set the tick labels
ax.set_xticklabels(["male", "female", "diverse"])
ax.set_yticklabels(y_u)

# setting the entire background color to light_blue
plt.gca().set_facecolor(colors["light_blue"])
plt.gcf().set_facecolor(colors["light_blue"])

ax.set_title("Percentage of people with mental health issues, featuring age and gender")

# show entire plot
plt.xlim(0, x)
plt.ylim(0, y)

plt.savefig("./figures/grouping.png")