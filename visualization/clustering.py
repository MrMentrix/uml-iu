import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score

"""
Explanation:
I found it hard to perform a clustering analysis on this data, because the individual range of the features tends to be very small. When clustering, e.g., two features with a range of [0, 4], there are just 25 different combinations of values and the distances between adjecent points are always the same, which makes the data pretty useless for clustering.
I tried to group the data differently, using the set distances between adjecent points in grouping.py, but the information gained from there didn't look to promising either.

Therefore, I'll now approach clustering from another direction, where I first calculate a mental health measure based on other features and then cluster people based on this mental health measure and their respective age. I choose age as the second feature because there are the most unique values and because it should help dividing the overall sample into smaller clusters. I'll use color coding to differntiate between gender.
"""

# note: depending on what shell/ide you use to run this code, you may need to adjust the paths to the data files

df = pd.read_csv("./preprocessed.csv")
colors = json.load(open("./visualization/colors.json"))

# creating a mental health score based on multiple features and their correlation with "mh_issues"
# I'm aware that I used "condition_belief", "diagnosed_professionally" and "treatment_professional" in preprocessing.py to form the "mh_issues" feature
# I'm using them here too because while their correlation shouldn't be influenced much or at all by this
features = ["family_mh_history", "mh_disorder_past", "mh_disorder_current", "mh_issue_interview", "condition_belief", "diagnosed_professionally"]

corr_dict = dict()

for feature in features:
    # calculate correlation between feature and "mh_issues"
    corr = df["mh_issues"].corr(df[feature])
    corr_dict[feature] = corr

# create a new column "issue_score" and calculate the score based on corr_dict
df["issue_score"] = df[list(corr_dict.keys())].apply(lambda x: sum([x[feature] * corr_dict[feature] for feature in list(corr_dict.keys())]), axis=1)

# normalizing the data
age_range = df["age"].max() - df["age"].min()
age_min = df["age"].min()
df["issue_score"] = df["issue_score"].apply(lambda x: (x - df["issue_score"].min()) / (df["issue_score"].max() - df["issue_score"].min()))
df["age"] = df["age"].apply(lambda x: (x - df["age"].min()) / age_range)

# start clustering with Agglomerative Clustering
features = ["issue_score", "age"]
X = df[features]

hac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
hac.fit(X)

dendro = sch.dendrogram(sch.linkage(X, method="ward", metric="euclidean"))

# show dendro
plt.title("Dendrogram using age and mental health score")
plt.xlabel("Participants")
plt.ylabel("Euclidean distances")
plt.xticks([])

# setting the entire background color to light_blue
plt.gca().set_facecolor(colors["light_blue"])
plt.gcf().set_facecolor(colors["light_blue"])

plt.savefig("./figures/dendrogram.png")

# reset plot
plt.clf()

# looking at silhouette score
print("Silhouette Score:", silhouette_score(X, hac.labels_)) # found the optimal at 4 clusters, with a score of ~0.43

df["cluster"] = hac.labels_

# plot the clusters

# restore age range
df["age"] = df["age"].apply(lambda x: x * age_range + age_min)

cluster_colors = ["red", "yellow", "green", "purple"]

for cluster in df["cluster"].unique():
    # plot the cluster   
    plt.scatter(df[df["cluster"] == cluster]["age"], df[df["cluster"] == cluster]["issue_score"], c=colors[cluster_colors[cluster]], label=cluster)

plt.title("Agglomerative Clustering using age and mental health score")
plt.xlabel("Age")
plt.ylabel("Mental Issue Score")

# setting the entire background color to light_blue
plt.gca().set_facecolor(colors["light_blue"])
plt.gcf().set_facecolor(colors["light_blue"])

plt.savefig("./figures/clusters.png")

# add color to df["color"] based on cluster
df["color"] = df["cluster"].apply(lambda x: cluster_colors[x])


df.to_csv("./clustered.csv", index=False)
