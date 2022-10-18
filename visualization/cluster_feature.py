import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.ticker import PercentFormatter

"""
Wasn't used for visual presentation in the end.
This script will look at some manually chosen differences between the clusters from the agglomerative clustering"""

# note: depending on what shell/ide you use to run this code, you may need to adjust the paths to the data files

df = pd.read_csv("./clustered.csv")
colors = json.load(open("./visualization/colors.json"))

feature = "gender" # feature to investigate

color= [colors["green"], colors["pink"], colors["blue"], colors["yellow"], colors["purple"], colors["orange"], colors["pink"]]
edgecolors = [colors["red"], colors["yellow"], colors["blue"], colors["purple"]]

fig, ax = plt.subplots(figsize=(10, 5))

df.groupby("cluster")[feature].value_counts(normalize=True).unstack().plot(kind="bar", stacked=True, color=color, ax=ax)

# set edgecolors for the entire stacked bars to distinguish between clusters
cluster_count = len(df["cluster"].unique())

for i, bar in enumerate(ax.patches):
    print(bar)

# manually plotting a frame around the columns because I couldn't figure out how to do this automatically using edgecolor when plotting the stacked bars earlier
for i in range(cluster_count): 
    ax.plot([-0.25+(i), -0.25+(i)], [0, 1], color=edgecolors[i], linewidth=5)
    ax.plot([-0.25+(i), 0.25+(i)], [1, 1], color=edgecolors[i], linewidth=5)
    ax.plot([0.25+(i), 0.25+(i)], [0, 1], color=edgecolors[i], linewidth=5)

plt.title(f"Distribution of {feature} for every cluster")
plt.ylabel("Percentage")
plt.xlabel("")
# showing percentage on y axis
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Diverse", "Female", "Male"])

# replace ticks with df["color"]
plt.xticks([0, 1, 2, 3], ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"])
plt.xticks(rotation=0)

# showing individual percentage of people with and without mental health issues on centered on each bar in percent
for p in plt.gca().patches: # for each bar
    height = p.get_height() # get height of bar
    y_coordinate = p.get_y() # get starting_height of each bar
    
    if not np.isnan(height): # if height is not NaN/making sure that there is a value
        plt.gca().text(p.get_x()+p.get_width()/2.,  # x position of text
                y_coordinate + height/2,            # y position of text
                '{:2.0f}%'.format(height*100),      # adding percentage
                ha="center", va="center",           # text alignment
                fontsize=10)                        # text size

# setting the entire background color to light_blue
plt.gca().set_facecolor(colors["light_blue"])
plt.gcf().set_facecolor(colors["light_blue"])

plt.savefig(f"./figures/{feature}_clusters.png")
plt.show()