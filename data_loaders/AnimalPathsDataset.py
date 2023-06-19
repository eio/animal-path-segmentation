import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
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
        # Group the data into individual animal trajectories, where:
        # trajectory = daily positions per individual in one calendar year
        # (ordered chronologically)
        self.trajectories = self.paths_df.groupby([ID_YEAR])
        # Get the unique group names (i.e. list of unique individual+year combinations)
        self.trajectory_ids = list(self.trajectories.groups.keys())
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
        (i.e. length of the list of unique individual+year combinations)
        """
        return len(self.trajectory_ids)

    def total_records(self):
        """
        Count the total number of waypoints
        in the post-transformed, post-grouped DataFrame
        """
        return self.trajectories.size().sum()

    def load_and_transform_data(self, csv_file):
        """
        Load, trim, clean, and transform the data
        """
        # Load the animal location data
        df = pd.read_csv(csv_file)
        # Drop all columns except the ones we care about
        df = df[[IDENTIFIER, STATUS, TIMESTAMP, ID_YEAR] + FEATURE_COLUMNS]
        # Replace 'Fall' with 'Autumn' because OCD
        df[STATUS] = df[STATUS].replace("Fall", "Autumn")
        # Convert TIMESTAMP column to datetime type
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        # Sort by ID+Year (i.e. trajectory ID) and Timestamp to order chronologically
        df = df.sort_values([ID_YEAR, TIMESTAMP])
        return df

    def to_tensor(self, trajectory_id, trajectory):
        """
        Convert the provided trajectory's features and labels to tensors
        """
        features = trajectory[FEATURE_COLUMNS].values
        labels = trajectory[STATUS].values
        # Convert relevant data to tensors
        features = torch.from_numpy(features).type(torch.float32)
        # Convert the string labels to one-hot encoded labels
        encoded_labels = np.array([SEASON_LABELS[label] for label in labels])
        # Convert encoded labels to tensors
        encoded_labels = torch.from_numpy(encoded_labels).type(torch.float32)
        return {
            "id": trajectory_id,
            "features": features,
            "labels": encoded_labels,
        }

    def __getitem__(self, idx):
        """
        Build the features vector and label for the provided data row
        """
        if torch.is_tensor(idx):
            row = idx.tolist()
        # Get the unique trajectory identifier
        trajectory_id = self.trajectory_ids[idx]
        # Get the trajectory of downsampled time-series data for this individual+year
        trajectory = self.tensorized_trajectories[trajectory_id]
        # Build the sample dictionary, including the individual+year ID for CSV output
        sample = {
            "id": trajectory_id,
            "features": trajectory["features"],
            "labels": trajectory["labels"],
        }
        # Return the item/sample
        return sample
