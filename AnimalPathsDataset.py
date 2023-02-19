# Code adapted from:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import os
import torch
import numpy as np
import pandas as pd

# Numerical path segment labels
WINTER_HOME = 0
SUMMER_HOME = 1
MIGRATING = 2
# Keep track of all possible categories
ALL_CATEGORIES = [WINTER_HOME, SUMMER_HOME, MIGRATING]
# Define strings for the column/feature names used
IDENTIFIER = "individual-local-identifier"  # +1 feature
LATITUDE = "location-lat"  # +1 feature
LONGITUDE = "location-long"  # +1 feature
TIMESTAMP = "timestamp"  # # +4 features (year, month, day, unixtime)
# Keep track of total number of input features
N_FEATURES = 7


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
        # Load the animal location data
        df = pd.read_csv(csv_file)
        # Clean the data by dropping unusable rows
        df = df[df[LATITUDE].notna()]
        df = df[df[LONGITUDE].notna()]
        df = df[df[TIMESTAMP].notna()]
        df = df[df[IDENTIFIER].notna()]
        self.paths_df = df
        # Initialize identifier lookup table
        self.individuals = {}
        self.transform = transform

    def __len__(self):
        self.paths_df
        return len(self.paths_df)

    def get_val(self, column_name, row):
        """
        Return a single column value from the provided row
        """
        idx = self.paths_df.columns.get_loc(column_name)
        return self.paths_df.iloc[row, idx]

    def get_time_features(self, dt):
        """
        Return vector of numerical time features:
        integers for date values, and float for Unix time
        """
        dt = pd.to_datetime(dt)
        d = dt.date()
        return [d.year, d.month, d.day, dt.timestamp()]

    def convert_id_to_num(self, ident):
        """
        Convert from an individual animal's identifier
        string to an integer feature value
        """
        if ident in self.individuals:
            # retrieve zero-indexed identifier
            return self.individuals[ident]
        else:
            # assign and return a new
            # zero-indexed identifier
            idx = len(self.individuals)
            self.individuals[ident] = idx
            return idx

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

    def __getitem__(self, row):
        """
        Build the features vector and label
        for the provided data row
        """
        if torch.is_tensor(row):
            row = row.tolist()
        # Get the geo-location data
        lat = self.get_val(LATITUDE, row)
        lon = self.get_val(LONGITUDE, row)
        # Get the time data, creating numerical features
        # for year, month, day, and Unix time
        time = self.get_val(TIMESTAMP, row)
        time_features = self.get_time_features(time)
        # Get the individual ID
        ident = self.get_val(IDENTIFIER, row)
        # Convert the individual ID to a numerical value
        idnum = self.convert_id_to_num(ident)
        # TODO: embed tag labels in the data itself
        # label = self.getval("segment-label")
        label = self.get_segment_label(lat)
        # Combine identifier, time, and location features
        features = [idnum] + time_features + [lat, lon]
        # Build the sample dictionary
        sample = {"features": np.array(features), "label": np.array([label])}
        # Apply data transformations if any are specified
        if self.transform:
            sample = self.transform(sample)
        # Return the item/sample
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        features, label = sample["features"], sample["label"]
        # convert to tensors
        features = torch.from_numpy(features)
        label = torch.from_numpy(label)
        return {
            "features": features,
            "label": label,
        }
