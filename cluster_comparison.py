import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import PercentFormatter

df = pd.read_csv("./clustered.csv")
colors = json.load(open("./visualization/colors.json"))

ignore = ["mh_issues", "issue_score", "cluster", "color", "treatment_professional", "diagnosed_professionally", "condition_belief", "gender"]

compare_df = pd.DataFrame()
compare_df["color"] = df.groupby("cluster")["color"].first()
compare_df["issue_score"] = df.groupby("cluster")["issue_score"].mean()

compare_df["male_perc"] = df.groupby("cluster").apply(lambda x: len(x[x["gender"] == "male"])/len(x))
compare_df["female_perc"] = df.groupby("cluster").apply(lambda x: len(x[x["gender"] == "female"])/len(x))
compare_df["diverse_perc"] = df.groupby("cluster").apply(lambda x: len(x[x["gender"] == "diverse"])/len(x))

for feature in df.columns:
        if feature not in ignore:
            # normalize the values of the feature
            df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
            # add mean of cluster to compare_df
            compare_df[feature] = df.groupby("cluster")[feature].mean()

# order rows by issue_score
compare_df = compare_df.sort_values(by="issue_score")
# reset index
compare_df = compare_df.reset_index()

def plot_all(df):

    # compare_df.to_csv("compare.csv")

    # plot every feature in compare_df into line chart
    counter = 0
    fig = 1
    ignore.remove("issue_score")

    for feature in df.columns:
        if counter >= 10:
            counter = 0

            # create legend
            plt.title(f"Comparison of clusters #{fig}")
            plt.xlabel("Cluster")
            plt.ylabel("Percentage")
            plt.xticks([0, 1, 2, 3], ["Red Cluser", "Green Cluster", "Yellow Cluster", "Purple Cluster"])
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

            # add legend to the right of the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # ensure that the legend is not cut off
            plt.tight_layout()
            # making plot wider
            plt.gcf().set_size_inches(10, 5)

            # setting the entire background color to light_blue
            plt.gca().set_facecolor(colors["light_blue"])
            plt.gcf().set_facecolor(colors["light_blue"])

            plt.savefig(f"./figures/cluster_compare_{fig}.png")
            plt.clf()
            fig += 1
        if feature not in ignore:
            plt.plot(df[feature], label=feature)
            counter += 1

    # create legend
            plt.title(f"Comparison of clusters #{fig}")
            plt.xlabel("Cluster")
            plt.ylabel("Percentage")
            plt.xticks([0, 1, 2, 3], ["Red Cluser", "Green Cluster", "Yellow Cluster", "Purple Cluster"])
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

            # add legend to the right of the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # ensure that the legend is not cut off
            plt.tight_layout()
            # making plot wider
            plt.gcf().set_size_inches(10, 5)

            # setting the entire background color to light_blue
            plt.gca().set_facecolor(colors["light_blue"])
            plt.gcf().set_facecolor(colors["light_blue"])

            plt.savefig(f"./figures/cluster_compare_{fig}.png")

def plot_top(df, n=10):
    # calculate absolut correlation between every feature and issue_score
    corr = df.corr().abs()["issue_score"].sort_values(ascending=False)
    # drop issue_score from corr
    corr.drop("issue_score", inplace=True)
    # set all values in corr to be absolute values
    corr = corr.abs()
    # get the n features with the highest correlation
    top = corr.nlargest(n).index

    # create a new dataframe with only the top features
    top_df = df[top]
    # plot the top features to line chart
    top_df.plot()
    # create legend
    plt.title(f"Top {n} correlating features")
    plt.xlabel("Cluster")
    plt.ylabel("Percentage")
    plt.xticks([0, 1, 2, 3], ["Red Cluser", "Green Cluster", "Yellow Cluster", "Purple Cluster"])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # add legend to the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # ensure that the legend is not cut off
    plt.tight_layout()
    # making plot wider
    plt.gcf().set_size_inches(10, 5)

    # setting the entire background color to light_blue
    plt.gca().set_facecolor(colors["light_blue"])
    plt.gcf().set_facecolor(colors["light_blue"])

    plt.savefig(f"./figures/top_correlation.png")
    plt.clf()

def plot_features(df, features):
    df[features].plot()
    # create legend
    plt.title(f"Personal features")
    plt.xlabel("Cluster")
    plt.ylabel("Percentage")
    plt.xticks([0, 1, 2, 3], ["Red Cluser", "Green Cluster", "Yellow Cluster", "Purple Cluster"])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # add legend to the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # ensure that the legend is not cut off
    plt.tight_layout()
    # making plot wider
    plt.gcf().set_size_inches(10, 5)

    # setting the entire background color to light_blue
    plt.gca().set_facecolor(colors["light_blue"])
    plt.gcf().set_facecolor(colors["light_blue"])

    plt.savefig(f"./figures/feature_development.png")
    plt.clf()

# plot_all(compare_df)
# plot_top(compare_df, 6)
plot_features(compare_df, ["male_perc", "female_perc", "diverse_perc", "family_mh_history", "age"])