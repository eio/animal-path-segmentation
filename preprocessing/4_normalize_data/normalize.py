import json
import torch
import pandas as pd
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *
from utils.Normalizer import ScaleValues

# Input: Labeled events data with all features, downsampled to daily position updates
INPUT_CSV = "Cranes_downsampled.csv"
# Output: All of the above, but with model inputs normalized
OUTPUT_CSV = "Cranes_normalized.csv"
# More outputs from this script
NORMS_OUT = "norms.json"
INVERSE_CSV_OUT = "Cranes_normalized_Inverse.csv"


def normalize():
    # Load the input dataframe
    df = pd.read_csv(INPUT_CSV)

    # Define the min and max ranges for each column
    ranges: dict[str, any] = {
        LATITUDE: (-90, 90),
        LONGITUDE: (-180, 180),
        MONTH: (1, 12),
        DAY: (1, 31),
        SINTIME: (-1, 1),
        COSTIME: (-1, 1),
        DISTANCE: (df[DISTANCE].min(), df[DISTANCE].max()),
        VELOCITY: (df[VELOCITY].min(), df[VELOCITY].max()),
        BEARING: (0, 360),
        TURN_ANGLE: (-180, 180),
    }

    # Apply scaling to each column
    for column, (min_val, max_val) in ranges.items():
        print("Scaling column: `{}`...".format(column))
        scaler = ScaleValues(max_range=max_val, min_range=min_val)
        # Apply normalization and save only scalar value (.item()) from tensor
        df[column] = df[column].apply(lambda x: scaler(torch.tensor(x)).item())

    # Save the normalization scalars to a JSON file
    with open(NORMS_OUT, "w") as f:
        json.dump(ranges, f)
    print("Saved norms config to: `{}`".format(NORMS_OUT))

    # Save the normalized dataframe
    print("Saving normalized data...")
    df = df.to_csv(OUTPUT_CSV, index=False)
    print("Saved normalized data to: `{}`\n".format(OUTPUT_CSV))


def inverse_normalize():
    # Load the normalization scalars from the JSON file
    with open(NORMS_OUT, "r") as f:
        norm_config = json.load(f)

    # Load the normalized dataframe
    df_normalized = pd.read_csv(OUTPUT_CSV)

    # Apply inverse normalization to each column
    for column, (min_val, max_val) in norm_config.items():
        print("Inverse scaling column: `{}`...".format(column))
        scaler = ScaleValues(max_range=max_val, min_range=min_val)
        # Apply inverse normalization and save only scalar value (.item()) from tensor
        df_normalized[column] = df_normalized[column].apply(
            lambda x: scaler.inverse_normalize(torch.tensor(x)).item()
        )

    # Save the inverse normalized dataframe
    print("Saving inverse normalized data...")
    df_normalized.to_csv(INVERSE_CSV_OUT, index=False)
    print("Saved inverse normalized data to: `{}`\n".format(INVERSE_CSV_OUT))


if __name__ == "__main__":
    normalize()
    # inverse_normalize()
