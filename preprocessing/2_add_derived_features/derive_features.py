import pandas as pd
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *
from time_features import *
from movement_features import *


if __name__ == "__main__":
    # Load the CSV file into a dataframe
    df = pd.read_csv("Cranes_processed.csv")
    # Drop all columns except the ones we care about
    df = df[
        [
            IDENTIFIER,
            LATITUDE,
            LONGITUDE,
            TIMESTAMP,
            # SPECIES,
            STATUS,
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
    # Group by individual trajectories
    trajectories = df.groupby([IDENTIFIER])

    print("Deriving movement features...")
    # Calculate velocity, bearing, and turn angle
    # across all waypoints for each trajectory
    df = trajectories.apply(calculate_velocity_bearing_turn).reset_index(
        drop=True  # reset index to obtain a new DataFrame with same shape as original one
    )
    # Cleanup
    df = df[df[VELOCITY].notna()]
    df = df[df[BEARING].notna()]
    df = df[df[TURN_ANGLE].notna()]
    # # Delete datetime column now that we're done using it
    # del df[TIMESTAMP]

    # Save the dataframe to a CSV file
    outname = "Cranes_all_features.csv"
    df.to_csv(outname, index=False)
    print("CSV with derived feature columns saved to: `{}`".format(outname))
