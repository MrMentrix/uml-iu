import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import json

# Reading the data
df = pd.read_csv("../preprocessed.csv")

# getting colors from colors.json
colors = json.load(open("./colors.json"))

"""Looks at the distribution of mental health issues across different age groups. Visualizes this with a bar chart."""

# categorizing participants of survey, based on their age
df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 80], labels=['19-25', '26-35', '36-45', '46+'])

# plotting age_group / mh_issues bar chart, showing the percentage of people with mental health issues, based on their age group
# show mentally healthy people green and mentally unhealthy people red
df.groupby('age_group')['mh_issues'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, color=[colors["green"], colors["red"]])

plt.title("Percentage of people with mental health issues, grouped by age")
plt.xlabel("Age group")
plt.ylabel("Percentage")
# showing percentage on y axis
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

"""Formatting the legend of the plot and adding some more information to the displayed data"""

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

# rename labels of legend for mh_issues from 0 to "No issues" and from 1 to "Issues" and locating it below the plot
plt.legend(["No issues", "Had/has issues"], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

# adding the amount of people per age group to the plot
for i, v in enumerate(df['age_group'].value_counts()):
    plt.text(i-0.1, 0.05, str(v), color="black")

"""Some slight general formatting of the plot to make it look nice"""

# setting the entire background color to light_blue
plt.gca().set_facecolor(colors["light_blue"])
plt.gcf().set_facecolor(colors["light_blue"])

# save plot as "mental_health_age_group.png" in figures folder
plt.savefig("../figures/mental_health_age_group.png", bbox_inches='tight')
plt.show()