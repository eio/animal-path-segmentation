# Code adapted from:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Number of possible labels
N_CATEGORIES = 4
# One-hot encoding for seasonal labels
SEASON_LABELS = {
    "Winter": [1, 0, 0, 0],
    "Spring": [0, 1, 0, 0],
    "Summer": [0, 0, 1, 0],
    "Autumn": [0, 0, 0, 1],
}
# Define strings for the column/feature names used
IDENTIFIER = "individual_id"
LATITUDE = "lat"  # +1 feature
LONGITUDE = "lon"  # +1 feature
TIMESTAMP = "timestamp"  # # +6 features (Y, M, D, unix, sin, cos)
YEAR = "Year"
MONTH = "Month"
DAY = "Day"
UNIXTIME = "UnixTime"
SINTIME = "SinTime"
COSTIME = "CosTime"
# TODO: encode the Species
# SPECIES = "species"
# STOPOVER = "stopover"
STATUS = "status"  # the seasonal segmentation label
# CONFIDENCE = "confidence"
# Seconds in a year (i.e., 365.25 * 24 * 60 * 60)
SECONDS_IN_YEAR = 31_536_000
# Used for the output CSV
OUTPUT_FIELDNAMES = [
    "Correct",
    "Predicted",
    "Actual",
    IDENTIFIER,
    LATITUDE,
    LONGITUDE,
    YEAR,
    MONTH,
    DAY,
    UNIXTIME,
    SINTIME,
    COSTIME,
]
# # Keep track of total number of input features
# N_FEATURES = 8


class AnimalPathsDataset(torch.utils.data.Dataset):
    """Animal paths dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string):
                - Path to the csv file with path segment annotations.
            transform (callable, optional):
                - Optional transform to be applied on a sample.
        """
        # Load, trim, clean, and transform the data
        self.paths_df = self.load_and_transform_data(csv_file)
        # Group the data into individual animal trajectories
        # where a trajectory = the full time series per individual
        self.trajectories = self.paths_df.groupby([IDENTIFIER])
        # Get the unique group names (i.e. list of unique individuals)
        self.individual_ids = list(self.trajectories.groups.keys())
        # Initialize identifier lookup table
        self.individuals = {}
        # Data transform applied in DataLoaders
        self.transform = transform

    def __len__(self):
        """
        Count the number of individual trajectories
        which corresponds to unique individual IDs
        """
        return len(self.individual_ids)

    def total_records(self):
        """
        Count the total number of records
        in the original cleaned dataset
        """
        return len(self.paths_df)

    def load_and_transform_data(self, csv_file):
        """
        Load, trim, clean, and transform the data
        """
        # Load the animal location data
        df = pd.read_csv(csv_file)
        # Drop all columns except the ones we care about
        df = df[
            [
                IDENTIFIER,
                LATITUDE,
                LONGITUDE,
                TIMESTAMP,
                # CONFIDENCE,
                # STOPOVER,
                # SPECIES,
                STATUS,
            ]
        ]
        # Drop rows with missing data
        df = df[df[LATITUDE].notna()]
        df = df[df[LONGITUDE].notna()]
        df = df[df[TIMESTAMP].notna()]
        df = df[df[IDENTIFIER].notna()]
        # Replace 'Fall' with 'Autumn' because OCD
        df[STATUS] = df[STATUS].replace("Fall", "Autumn")
        # Expand the time features to numerical values
        df = self.transform_time_features(df)
        # # Print some stats about the data
        # print("Label stats:\n{}".format(df[STATUS].value_counts()))
        # print("Individual stats:\n{}".format(df[IDENTIFIER].value_counts()))
        return df

    def cyclic_time(self, dt):
        """
        Use a datetime input to calculate the
        number of seconds since the start of the year and the period of the cycle
        (e.g., number of seconds in a day, a week, or a year).
        Return the sine and cosine values of the angle between the time and the cycle.
        """
        # Build the datetime for the start of the timestamp's year
        start_of_year = datetime(dt.year, 1, 1)
        # Calculate the number of seconds since the start of the timestamp's year
        seconds_since_year_start = (dt - start_of_year).total_seconds()
        # Set the period of the cycle to the total number of seconds in a year
        period = SECONDS_IN_YEAR
        # Caculate cylic time
        angle = 2 * math.pi * seconds_since_year_start / period
        sin_time = math.sin(angle)
        cos_time = math.cos(angle)
        return sin_time, cos_time

    def transform_time_features(self, df):
        """
        Return vector of numerical time features:
        - integers for Year, Month, and Day values
        - float for Unix time
        - floats for sin/cos time (cyclical)
        """
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        df[YEAR] = df[TIMESTAMP].dt.year
        df[MONTH] = df[TIMESTAMP].dt.month
        df[DAY] = df[TIMESTAMP].dt.day
        df[UNIXTIME] = df[TIMESTAMP].apply(lambda x: x.timestamp())
        # Represent the time as a cyclic feature for seasons
        df[[SINTIME, COSTIME]] = df[TIMESTAMP].apply(
            lambda x: pd.Series(self.cyclic_time(x))
        )
        # Delete the original "timestamp" column
        del df[TIMESTAMP]
        # Return the transformed dataframe
        return df

    def __getitem__(self, idx):
        """
        Build the features vector and label
        for the provided data row
        """
        if torch.is_tensor(idx):
            row = idx.tolist()
        # Get the unique individual identifier
        identifier = self.individual_ids[idx]
        # Get the full trajectory of time-series data for this individual
        trajectory = self.trajectories.get_group(identifier).reset_index(drop=True)
        # Get the seasonal segmentation labels
        labels = trajectory[STATUS]
        # Delete the ID and Status columns, since we don't want them as data features
        del trajectory[IDENTIFIER]
        del trajectory[STATUS]
        # TODO: Normalize
        # 0 mean and standard deviation
        # self.normalize(trajectory)
        # Build the sample dictionary, including the animal ID for CSV output
        sample = {"id": identifier, "features": trajectory, "labels": labels}
        # Apply data transformations if any are specified
        if self.transform:
            sample = self.transform(sample)
        # Return the item/sample
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        identifier, features, label = (
            sample["id"],
            sample["features"],
            sample["labels"],
        )
        # Convert relevant data to tensors with dtype == torch.float32
        features = torch.from_numpy(features.values).type(torch.float)

        # Convert the string labels to numerical labels
        num_labels = np.array([SEASON_LABELS[l] for l in label])
        label = torch.from_numpy(num_labels).type(torch.float)

        return {
            "id": identifier,
            "features": features,
            "labels": label,
        }
