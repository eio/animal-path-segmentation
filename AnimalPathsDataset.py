import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Local scripts
from utils.movement import calculate_velocity_bearing_turn

# Number of possible labels
N_CATEGORIES = 4
# One-hot encoding for seasonal labels
SEASON_LABELS = {
    "Winter": [1, 0, 0, 0],
    "Spring": [0, 1, 0, 0],
    "Summer": [0, 0, 1, 0],
    "Autumn": [0, 0, 0, 1],
}
# Define strings for column/feature names
IDENTIFIER = "individual_id"
# Coordinates
LATITUDE = "lat"  # +1 feature
LONGITUDE = "lon"  # +1 feature
# Original time column
TIMESTAMP = "timestamp"
# Derived movement features
VELOCITY = "velocity"
BEARING = "bearing"
TURN_ANGLE = "turn_angle"
# Derived time features
MONTH = "month"  # +1 feature
DAY = "day"  # +1 feature
SINTIME = "sin_time"  # +1 feature
COSTIME = "cos_time"  # +1 feature
# UNIXTIME = "UnixTime"
# YEAR = "Year"  #
# SPECIES = "species"
# CONFIDENCE = "confidence"
# # Stopover flag (binary)
# STOPOVER = "stopover"
STATUS = "status"  # the seasonal segmentation label
# Group time features for normalization
TIME_FEATURES = [
    # YEAR,
    MONTH,
    DAY,
    # UNIXTIME,
    SINTIME,
    COSTIME,
]
# Group derived movement features
MOVEMENT_FEATURES = [
    VELOCITY,
    BEARING,
    TURN_ANGLE,
]
# All input feature column names:
FEATURE_COLUMNS = (
    [
        LATITUDE,
        LONGITUDE,
    ]
    + TIME_FEATURES
    + MOVEMENT_FEATURES
)
# Number of input features: 6
N_FEATURES = len(FEATURE_COLUMNS)


class AnimalPathsDataset(torch.utils.data.Dataset):
    """Animal paths dataset."""

    def __init__(self, csv_file, burst_time_threshold, transform=None):
        """
        Args:
            csv_file (string):
                - Path to the csv file with path segment annotations.
            transform (callable, optional):
                - Optional transform to be applied on a sample.
        """
        # Seconds in a year (i.e., 365.25 * 24 * 60 * 60)
        self.SECONDS_IN_YEAR = 31_536_000
        # Set the time threshold for an animal movement "burst"
        self.BURST_TIME_THRESHOLD = burst_time_threshold
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
                # STOPOVER,
                LATITUDE,
                LONGITUDE,
                TIMESTAMP,
                # CONFIDENCE,
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
        # Group by individual trajectories
        trajectories = df.groupby([IDENTIFIER])
        # Calculate velocity, bearing, and turn angle
        # across all waypoints for each trajectory
        df = trajectories.apply(
            calculate_velocity_bearing_turn,
            burst_time_threshold=self.BURST_TIME_THRESHOLD,
        ).reset_index(
            drop=True  # reset index to obtain a new DataFrame with same shape as original one
        )
        # Delete datetime column now that we're done using it
        del df[TIMESTAMP]
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
        period = self.SECONDS_IN_YEAR
        # Caculate cylic time
        angle = 2 * math.pi * seconds_since_year_start / period
        sin_time = math.sin(angle)
        cos_time = math.cos(angle)
        return sin_time, cos_time

    def transform_time_features(self, df):
        """
        Add new time features to the dataframe:
        - integers for Year, Month, and Day values
        - float for Unix time
        - floats for sin/cos time (cyclical)
        """
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        df[DAY] = df[TIMESTAMP].dt.day
        df[MONTH] = df[TIMESTAMP].dt.month
        # df[YEAR] = df[TIMESTAMP].dt.year
        # df[UNIXTIME] = df[TIMESTAMP].apply(lambda x: x.timestamp())
        # Represent the time as a cyclic feature for seasons
        df[[SINTIME, COSTIME]] = df[TIMESTAMP].apply(
            lambda x: pd.Series(self.cyclic_time(x))
        )
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
        # Build the sample dictionary, including the animal ID for CSV output
        sample = {"id": identifier, "features": trajectory, "labels": labels}
        # Apply data transformations if any are specified
        if self.transform:
            sample = self.transform(sample)
        # Return the item/sample
        return sample


class NormalizeFeatures(object):
    """Normalize latitude, longitude, and timestamp features"""

    def __init__(self):
        self.scaler = StandardScaler()
        # Better to run self.inverseNormalize only when outputting CSV
        # but always storing the non-normalized features is ok for now
        self.orig_features = None

    def __call__(self, sample):
        features = sample["features"]
        # Store non-normalized features for writing output CSV
        self.orig_features = features.values.tolist()
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
