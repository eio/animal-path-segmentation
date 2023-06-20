import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *

# Load the processed data
df = pd.read_csv("Cranes_downsampled.csv")

# Define color mappings for each status value
color_mappings = {
    "Winter": "lightblue",
    "Fall": "darkorange",
    "Spring": "green",
    "Summer": "gold",
}

# Create a figure
# which we clear after each iteration
plt.figure()

# Loop through each feature
for feature in FEATURE_COLUMNS:
    # Loop through each STATUS value
    for status_value in df[STATUS].unique():
        # Select data for the current STATUS value and feature
        subset = df[df[STATUS] == status_value][feature]
        # Plot the histogram for the current STATUS value
        label = "STATUS=" + str(status_value)
        sns.histplot(
            data=subset,
            label=label,
            kde=True,  # show Kernel Density Estimate (KDE) line too
            color=color_mappings[status_value],
        )
        # Set the title and labels for the plot
        plt.title("`{}` values when {}".format(feature, label))
        plt.xlabel("`{}` Values".format(feature))
        plt.ylabel("Frequency")
        # # Display a legend
        # plt.legend()
        # Save the plot in the format statusvalue_featurename.png
        filename = "histograms/{}_{}.png".format(feature, str(status_value))
        plt.savefig(filename)
        print("Saved plot: `{}`".format(filename))
        # # Show the plot
        # plt.show()
        # Clear the current figure
        plt.clf()
