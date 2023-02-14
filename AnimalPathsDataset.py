# Code adapted from:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import os
import torch
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_ubyte, img_as_float

# Numerical path segment tags
WINTER_HOME = 0
SUMMER_HOME = 1
MIGRATING = 2


class AnimalPathsDataset(torch.utils.data.Dataset):
    """Animal paths dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with path segment annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths_df = pd.read_csv(csv_file)
        self.individuals = {}
        self.transform = transform

    def __len__(self):
        self.paths_df
        return len(self.paths_df)

    def get_val(self, row, column_name):
        idx = self.paths_df.columns.get_loc(column_name)
        return self.paths_df.iloc[row, idx]

    def get_time_features(self, dt):
        dt = pd.to_datetime(dt)
        t = dt.timestamp()
        date = dt.date()
        y = pd.to_datetime(t).date().year
        m = pd.to_datetime(t).date().month
        d = pd.to_datetime(t).date().day
        return [y, m, d, t]

    def convert_id_to_num(self, ident):
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
        if torch.is_tensor(row):
            row = row.tolist()
        # Get the geo-location data
        lat = float(self.get_val("location-lat"))
        lon = float(self.get_val("location-long"))
        # Get the time data, creating numerical features
        # for year, month, day, and Unix time
        time = self.get_val("timestamp")
        time_features = self.get_time_features(time)
        # Get the individual ID
        ident = self.get_val("individual-local-identifier")
        # Convert the individual ID to a numerical value
        idnum = self.convert_id_to_num(ident)
        # TODO: embed tag labels in the data itself
        # label = self.getval("segment-label")
        label = self.get_segment_label(lat)
        # Combine identifier, time, and location features
        features = [idnum] + time_features + [lat, lon]
        # Build the sample dictionary
        sample = {"features": features, "label": label}
        # Apply data transformations if any are specified
        if self.transform:
            sample = self.transform(sample)
        # Return the item/sample
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample["features"], sample["label"]
        # convert to tensors
        features = torch.from_numpy(features)
        label = torch.from_numpy(label)
        return {
            "features": features,
            "label": label,
        }
