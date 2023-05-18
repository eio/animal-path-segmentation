import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Local scripts
from utils.consts import *


class AnimalPathsDataset(torch.utils.data.Dataset):
    """Animal paths dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string):
                - path to the CSV file with labeled event data
        """
        # Load, trim, clean, and transform the data
        self.paths_df = self.load_and_transform_data(csv_file)
        # Group the data into individual animal trajectories
        # where a trajectory = the full time series per individual
        # (ordered sequentially by timestamp)
        self.trajectories = self.paths_df.groupby([IDENTIFIER])
        # Get the unique group names (i.e. list of unique individuals)
        self.individual_ids = list(self.trajectories.groups.keys())
        # Initialize a dictionary to store tensorized trajectories
        self.tensorized_trajectories = {}
        # Apply the `to_tensor` transformation to all groups in self.trajectories
        for identifier, trajectory_group in self.trajectories:
            # Convert features and labels to tensors
            tensorized_trajectory = self.to_tensor(identifier, trajectory_group)
            # Use `identifier` as key for easy lookup in __getitem__
            self.tensorized_trajectories[identifier] = tensorized_trajectory

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

    def to_tensor(self, identifier, trajectory):
        features = trajectory[FEATURE_COLUMNS].values
        labels = trajectory[STATUS].values
        # Convert relevant data to tensors with dtype == torch.float32
        features = torch.from_numpy(features).type(torch.float32)
        # Convert the string labels to one-hot encoded labels
        num_labels = np.array([SEASON_LABELS[label] for label in labels])
        label = torch.from_numpy(num_labels).type(torch.float32)
        return {
            "id": identifier,
            "features": features,
            "labels": label,
        }

    def __getitem__(self, idx):
        """
        Build the features vector and label for the provided data row
        """
        if torch.is_tensor(idx):
            row = idx.tolist()
        # Get the unique individual identifier
        identifier = self.individual_ids[idx]
        # Get the full trajectory of time-series data for this individual
        trajectory = self.tensorized_trajectories[identifier]
        # Build the sample dictionary, including the animal ID for CSV output
        sample = {
            "id": identifier,
            "features": trajectory["features"],
            "labels": trajectory["labels"],
        }
        # Return the item/sample
        return sample
