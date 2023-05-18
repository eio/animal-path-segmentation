import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Local scripts
from utils.consts import *


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
        df = df[[IDENTIFIER, STATUS, TIMESTAMP] + FEATURE_COLUMNS]
        # Replace 'Fall' with 'Autumn' because OCD
        df[STATUS] = df[STATUS].replace("Fall", "Autumn")
        # Sort by timestamp to ensure sequential order
        df = df.sort_values(TIMESTAMP)
        return df

    def __getitem__(self, idx):
        """
        Build the features vector and label for the provided data row
        """
        if torch.is_tensor(idx):
            row = idx.tolist()
        # Get the unique individual identifier
        identifier = self.individual_ids[idx]
        # Get the full trajectory of time-series data for this individual
        trajectory = self.trajectories.get_group(identifier).reset_index(drop=True)
        # Get the seasonal segmentation labels
        labels = trajectory[STATUS]
        # Delete the ID, Status, and Timestamp columns, since we don't want them as data features
        del trajectory[IDENTIFIER]
        del trajectory[STATUS]
        del trajectory[TIMESTAMP]
        # Build the sample dictionary, including the animal ID for CSV output
        sample = {
            "id": identifier,
            "features": trajectory,
            "labels": labels,
        }
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
        # Convert the string labels to one-hot encoded labels
        num_labels = np.array([SEASON_LABELS[l] for l in label])
        label = torch.from_numpy(num_labels).type(torch.float)
        return {
            "id": identifier,
            "features": features,
            "labels": label,
        }
