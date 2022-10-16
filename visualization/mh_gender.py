import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import json

# Reading the data
df = pd.read_csv("./preprocessed.csv")

# getting colors from colors.json
colors = json.load(open("./visualization/colors.json"))

"""Looks at the distribution of mental health issues across different genders of people. Visualizes this with a bar chart."""

df.groupby("gender", sort=False)["mh_issues"].value_counts(normalize=True).unstack().plot(kind="bar", stacked=True, color=[colors["green"], colors["red"]], sort_columns=True)

plt.title("Percentage of people with mental health, grouped by gender")
plt.xlabel("Gender")
plt.ylabel("Percentage")

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

# showing individual percentage of people with and without mental health issues on centered on each bar in percent
for p in plt.gca().patches: # for each bar
    height = p.get_height() # get height of bar
    y_coordinate = p.get_y() # get starting_height of each bar
    
    if not np.isnan(height): # if height is not NaN/making sure that there is a value
        if height == 0:
            continue
        plt.gca().text(p.get_x()+p.get_width()/2.,  # x position of text
                y_coordinate + height/2,            # y position of text
                '{:2.0f}%'.format(height*100),      # adding percentage
                ha="center", va="center",           # text alignment
                fontsize=10)                        # text size

# rename labels of legend for mh_issues from 0 to "No issues" and from 1 to "Issues" and locating it below the plot
plt.legend(["No issues", "Had/has issues"], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.xticks(rotation=0)

# adding the amount of people per age group to the plot
for i, v in enumerate(df['gender'].value_counts()):
    plt.text(i-0.1, 0.05, str(v), color="black")

# setting the entire background color to light_blue
plt.gca().set_facecolor(colors["light_blue"])
plt.gcf().set_facecolor(colors["light_blue"])

plt.savefig("./figures/mh_gender.png", bbox_inches="tight")
plt.show()