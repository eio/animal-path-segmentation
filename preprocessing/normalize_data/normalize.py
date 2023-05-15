import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os, sys

# Get the absolute path to the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Append the subdirectory containing the module to import to sys.path
module_dir = os.path.join(script_dir, "../../")
sys.path.append(module_dir)

# Local scripts
from consts import *


def get_norm_cfg_json():
    print(LATITUDE)


def NormalizeFeatures(sample):
    """
    Normalize latitude, longitude, and timestamp features
    """
    scaler = StandardScaler()
    features = sample["features"]
    # Separate latitude and longitude coordinates
    coords = features[[LATITUDE, LONGITUDE]]
    coords = torch.tensor(coords.values)
    coords = torch.nn.functional.normalize(coords, dim=0)
    features[LATITUDE] = coords[:, 0]
    features[LONGITUDE] = coords[:, 1]
    # Normalize the timestamp features using StandardScaler
    features[TIME_FEATURES] = self.scaler.fit_transform(features[TIME_FEATURES])
    # Reassign the normalized features
    sample["features"] = features
    return sample


if __name__ == "__main__":
    # Load the CSV file into a dataframe
    # df = pd.read_csv("Cranes_all_features.csv")
    # Normalize the feature, saving normalization config to JSON
    df, norm_cfg = get_norm_cfg_json()
