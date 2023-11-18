import shutil
import pandas as pd
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *
from time_features import *
from movement_features import *
from intra_day_features import *

# Input: Labeled and trimmed events data
INPUT_CSV = "Cranes_labeled.csv"
# Output: All of the above + derived time & movement features
OUTPUT_CSV = "Cranes_downsampled_all_features.csv"
OUTPUT_PATH = "output/" + OUTPUT_CSV


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
    # NOTE: averaging/centroid methodology might be worth trying too
    df = df.groupby([IDENTIFIER, YEAR, DATE]).last()
    # Ungroup the data and reset the nominal index
    df.reset_index(inplace=True)
    # Drop the DATE column since we only need TIMESTAMP
    df.drop(DATE, axis=1, inplace=True)
    # df[[ID_YEAR, TIMESTAMP]].to_csv("after_downsample.csv", index=False)
    return df


if __name__ == "__main__":
    # Load the CSV file into a dataframe
    df = pd.read_csv(INPUT_CSV)
    # Drop all columns except the ones we care about
    df = df[
        [
            IDENTIFIER,
            TIMESTAMP,
            STATUS,
            SPECIES,
            LATITUDE,
            LONGITUDE,
            # BEARING,
        ]
    ]
    # Drop rows with missing data
    df = df[df[LATITUDE].notna()]
    df = df[df[LONGITUDE].notna()]
    df = df[df[TIMESTAMP].notna()]
    df = df[df[IDENTIFIER].notna()]

    print("Deriving time features...")
    # Expand the time features to numerical values
    df = transform_time_features(df)

    print("Deriving movement features for all position records...")
    # Do this first in order to calculate mean values for intra-day features
    df = derive_movement_features(df)

    print("Deriving intra-day features...")
    # Add the intra-day features (daily means for movement features)
    df = calculate_intra_day_features(df)

    print("Downsampling to daily position records...")
    # Only grab the latest position update per day
    # (i.e. one waypoint per day, per animal)
    df = downsample_to_daily_positions(df)

    print("Deriving movement features for daily-downsampled data...")
    # Call the same function as above, but now for daily downsampled data
    df = derive_movement_features(df)

    # Cleanup
    df = df[df[DISTANCE].notna()]
    df = df[df[VELOCITY].notna()]
    df = df[df[BEARING].notna()]
    df = df[df[TURN_ANGLE].notna()]
    # Save the dataframe to a CSV file
    df.to_csv(OUTPUT_PATH, index=False)
    print("CSV with derived feature columns saved to: `{}`".format(OUTPUT_PATH))

    # Use shutil.copy to copy the file to the final destination
    destination_file = os.path.join("../3_normalize_data/" + OUTPUT_CSV)
    shutil.copy(OUTPUT_PATH, destination_file)
    # Print a message to confirm the copy operation
    print(f"Copied `{OUTPUT_PATH}` to `{destination_file}`")
    print("Done.\n")
