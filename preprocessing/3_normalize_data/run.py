import shutil
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
INPUT_CSV = "Cranes_downsampled_all_features.csv"
# Output: All of the above, but with model inputs normalized
OUTPUT_CSV = "Cranes_normalized.csv"
OUTPUT_PATH = "output/" + OUTPUT_CSV
# More outputs from this script
NORMS_JSON = "norms.json"
NORMS_OUT_PATH = "output/" + NORMS_JSON
INVERSE_CSV_OUT = "Cranes_normalized_Inverse.csv"


def normalize():
    # Load the input dataframe
    df = pd.read_csv(INPUT_CSV)

    # Define the min and max parameter ranges for each column
    param_ranges: dict[str, any] = {
        LATITUDE: (-90, 90),
        LONGITUDE: (-180, 180),
        MONTH: (1, 12),
        DAY: (1, 31),
        SINTIME: (-1, 1),
        COSTIME: (-1, 1),
        # Intra-day average values:
        MEAN_DISTANCE: (df[MEAN_DISTANCE].min(), df[MEAN_DISTANCE].max()),
        MEAN_VELOCITY: (df[MEAN_VELOCITY].min(), df[MEAN_VELOCITY].max()),
        MEAN_BEARING: (-180, 180),
        MEAN_TURN_ANGLE: (-180, 180),
        # Inter-day (daily downsampled) values:
        DISTANCE: (df[DISTANCE].min(), df[DISTANCE].max()),
        VELOCITY: (df[VELOCITY].min(), df[VELOCITY].max()),
        BEARING: (-180, 180),
        TURN_ANGLE: (-180, 180),
    }

    # Apply scaling to each column
    for column, (min_val, max_val) in param_ranges.items():
        print("Scaling column: `{}`...".format(column))
        scaler = ScaleValues(max_range=max_val, min_range=min_val)
        # Apply normalization and save only scalar value (.item()) from tensor
        df[column] = df[column].apply(lambda x: scaler(torch.tensor(x)).item())

    # Save the normalization parameters to a JSON file
    with open(NORMS_OUT_PATH, "w") as f:
        json.dump(param_ranges, f)
    print("Saved scaling params to: `{}`".format(NORMS_OUT_PATH))

    # Use shutil.copy to copy the file to the final destination
    STORED_NORMS = f"utils/{NORMS_JSON}"
    destination_file = os.path.join(sys.path[-1], STORED_NORMS)
    shutil.copy(NORMS_OUT_PATH, destination_file)
    # Print a message to confirm the copy operation
    print(f"Copied `{NORMS_OUT_PATH}` to `{STORED_NORMS}`")
    return df


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
    # Normalize the data
    df = normalize()
    # inverse_normalize()
    # Save the normalized dataframe
    print("Saving normalized dataset...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved normalized data to: `{}`\n".format(OUTPUT_PATH))
    # Use shutil.copy to copy the file to the final destination
    destination_file = os.path.join("../4_split_data/" + OUTPUT_CSV)
    shutil.copy(OUTPUT_PATH, destination_file)
    # Print a message to confirm the copy operation
    print(f"Copied `{OUTPUT_PATH}` to `{destination_file}`")
    print("Done.")
