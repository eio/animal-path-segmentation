import pandas as pd
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *

# Input: Labeled events data with all features
INPUT_CSV = "Cranes_all_features.csv"
# Output: The above data, but downsampled to one position per day
OUTPUT_CSV = "Cranes_downsampled.csv"


def downsample_to_daily_positions(df):
    """
    Filter the dataframe to only include
    the latest position update / waypoint per day
    """
    print("Downsampling data to daily position updates...")
    # df[[ID_YEAR, TIMESTAMP]].to_csv("before_downsample.csv", index=False)
    DATE = "date"
    # Assign a new name to df[TIMESTAMP].dt.date column
    df[DATE] = df[TIMESTAMP].dt.date
    # Only grab the last position update of each day
    # TODO: averaging/centroid methodology might be better
    df = df.groupby([IDENTIFIER, YEAR, DATE]).last()
    # Ungroup the data and reset the nominal index
    df.reset_index(inplace=True)
    # Drop the DATE column since we only need TIMESTAMP
    df.drop(DATE, axis=1, inplace=True)
    # df[[ID_YEAR, TIMESTAMP]].to_csv("after_downsample.csv", index=False)
    return df


def load_and_downsample(csv_file):
    """
    Load, trim, clean, and transform the data
    """
    # Load the animal location data
    df = pd.read_csv(csv_file)
    # Convert TIMESTAMP column to datetime type
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    # Only grab the latest position update per day
    # (i.e. one waypoint per day, per animal)
    df = downsample_to_daily_positions(df)
    return df


if __name__ == "__main__":
    # Load the CSV file into a dataframe
    df = load_and_downsample(INPUT_CSV)
    # Save the dataframe to a CSV file
    df.to_csv(OUTPUT_CSV, index=False)
    print(
        "CSV with daily-downsampled position updates saved to: `{}`".format(OUTPUT_CSV)
    )
