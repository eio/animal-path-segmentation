import pandas as pd
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *


def calculate_intra_day_features(df):
    """
    Add new intra-day features to the dataframe
    """

    # Calculate mean bearing, step length, velocity, and turn angle for each day
    mean_distance = df.groupby([YEAR, MONTH, DAY])[DISTANCE].mean()
    mean_velocity = df.groupby([YEAR, MONTH, DAY])[VELOCITY].mean()
    mean_bearing = df.groupby([YEAR, MONTH, DAY])[BEARING].mean()
    mean_turn_angle = df.groupby([YEAR, MONTH, DAY])[TURN_ANGLE].mean()

    # Create a new dataframe to store the calculated features
    new_df = pd.DataFrame(
        {
            MEAN_DISTANCE: mean_distance,
            MEAN_VELOCITY: mean_velocity,
            MEAN_BEARING: mean_bearing,
            MEAN_TURN_ANGLE: mean_turn_angle,
        }
    ).reset_index()

    # Save the dataframe to a CSV file
    new_df.to_csv("TEST_intra_day_pre-merge.csv", index=False)

    # Merge the original dataframe with the new dataframe based on the common columns
    merged_df = pd.merge(df, new_df, on=[YEAR, MONTH, DAY], how="left")

    # Return the merged dataframe
    return merged_df
