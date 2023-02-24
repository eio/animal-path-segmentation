# Code adapted from:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import torch
import numpy as np
import pandas as pd

# Numerical path segment labels
WINTER_HOME = 0
SUMMER_HOME = 1
MIGRATING = 2
# Keep track of all possible categories
ALL_CATEGORIES = ["WINTER_HOME", "SUMMER_HOME", "MIGRATING"]
# Define strings for the column/feature names used
IDENTIFIER = "individual-local-identifier"  # +1 feature
LATITUDE = "location-lat"  # +1 feature
LONGITUDE = "location-long"  # +1 feature
TIMESTAMP = "timestamp"  # # +4 features (year, month, day, unixtime)
YEAR = "Year"
MONTH = "Month"
DAY = "Day"
UNIXTIME = "UnixTime"
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
]
# # Keep track of total number of input features
# N_FEATURES = 7


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
        df = df[[IDENTIFIER, LATITUDE, LONGITUDE, TIMESTAMP]]
        # Drop rows with missing data
        df = df[df[LATITUDE].notna()]
        df = df[df[LONGITUDE].notna()]
        df = df[df[TIMESTAMP].notna()]
        df = df[df[IDENTIFIER].notna()]
        # Expand the time features to numerical values
        df = self.transform_time_features(df)
        return df

    def transform_time_features(self, df):
        """
        Return vector of numerical time features:
        integers for date values, and float for Unix time
        """
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        df[YEAR] = df[TIMESTAMP].dt.year
        df[MONTH] = df[TIMESTAMP].dt.month
        df[DAY] = df[TIMESTAMP].dt.day
        df[UNIXTIME] = df[TIMESTAMP].apply(lambda x: x.timestamp())
        # Delete the original "timestamp" column
        del df[TIMESTAMP]
        # TODO: Account for cyclical seasons in a year
        # sin/cos applied to `seconds_since_NYE` to capture circularity
        return df

    def get_segment_label(self, lat):
        # TODO: this function should be removed
        # once properly labeled data is acquired.
        # This is an Extremely Naive Approach to
        # animal range residency estimation.
        if lat < 0:
            # South of equator, assume winter home range
            return WINTER_HOME
        elif lat > 40:
            # North of 40 N, assume summer home range
            return SUMMER_HOME
        else:
            # Otherwise, assume migration-in-progress
            return MIGRATING

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
        # Delete the ID column, since we don't want it as a data feature
        del trajectory[IDENTIFIER]
        # TODO: Remove fake labels
        # Generate fake labels dataframe
        labels = trajectory[LATITUDE].apply(self.get_segment_label)
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
        label = torch.from_numpy(label.values).type(torch.float)
        return {
            "id": identifier,
            "features": features,
            "labels": label,
        }
