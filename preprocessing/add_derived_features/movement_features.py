from haversine import haversine, Unit
import numpy as np
import os, sys


# Get the absolute path to the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Append the subdirectory containing the module to import to sys.path
module_dir = os.path.join(script_dir, "../../")
sys.path.append(module_dir)
# Local scripts
from consts import *

# Constants
EARTH_RADIUS = 6371 * 1000  # meters
# Each position record is assigned to a "burst" of movement,
# where each burst lasts for the specified time duration.
# Model performance may change with different burst definitions.
BURST_TIME_THRESHOLD = 300  # seconds


def calculate_velocity_bearing_turn(df):
    # Ensure the index of the DataFrame is sequential after grouping
    df = df.reset_index(drop=True)
    # Convert latitude and longitude to radians
    lat_rad = np.radians(df[LATITUDE])
    lon_rad = np.radians(df[LONGITUDE])
    # Calculate spatial distances between consecutive rows
    dist = calculate_spatial_differences(lat_rad, lon_rad)  # meters
    # Calculate temporal differences between consecutive rows
    time_diff = calculate_time_differences(df)  # seconds
    # Identify the bursts
    burst_indices = identify_bursts(time_diff)
    # Calculate velocity, bearing, and turn angle for each burst
    for i in range(len(burst_indices) - 1):
        start_idx = burst_indices[i]
        end_idx = burst_indices[i + 1]
        if end_idx >= len(df):
            # Exit the loop if the end index is too high
            break
        # Calculate velocity, bearing, and turn angle:
        velocity = calculate_velocity(
            dist[start_idx:end_idx], time_diff[start_idx:end_idx]
        )
        bearing = calculate_bearing(
            lat_rad[start_idx : end_idx + 1], lon_rad[start_idx : end_idx + 1]
        )
        turn_angle = calculate_turn_angle(bearing)
        # Add velocity, bearing, and turn angle columns to the dataframe
        df.loc[start_idx + 1 : end_idx, VELOCITY] = velocity
        df.loc[start_idx + 1 : end_idx, BEARING] = bearing
        df.loc[start_idx + 1 : end_idx, TURN_ANGLE] = turn_angle
    # Fill NaN values with 0
    df.fillna(0, inplace=True)
    return df


def identify_bursts(time_diff):
    burst_indices = [0]
    for i in range(1, len(time_diff)):
        if time_diff[i - 1] > BURST_TIME_THRESHOLD:
            burst_indices.append(i)
    # print("burst_indices:", len(burst_indices))
    return burst_indices


def calculate_velocity(dist, dt):
    velocity = dist[0] / dt.iloc[0]
    return velocity


def calculate_bearing(lat_rad, lon_rad):
    # Pad with last value to ensure shapes match
    lon_diff = np.diff(np.concatenate([lon_rad, [lon_rad.iloc[-1]]]))
    # Calculate the bearing
    x = np.cos(lat_rad) * np.sin(lon_diff)
    y = (np.sin(lat_rad) * np.cos(lat_rad)) - (
        np.cos(lat_rad) * np.sin(lat_rad) * np.cos(lon_rad - lon_rad.iloc[0])
    )
    bearing = np.degrees(np.arctan2(x, y)) % 360
    return bearing


def calculate_turn_angle(bearing):
    turn_angle = np.degrees(
        np.arccos(np.clip(np.cos(np.radians(bearing - np.roll(bearing, 1))), -1, 1))
    )
    # Fill NaN values with 0
    turn_angle[np.isnan(turn_angle)] = 0
    return turn_angle


def calculate_spatial_differences(lat_rad, lon_rad):
    # Calculate spatial distances between consecutive rows
    dist = []
    for i in range(1, len(lat_rad)):
        lat1, lon1 = lat_rad[i - 1], lon_rad[i - 1]
        lat2, lon2 = lat_rad[i], lon_rad[i]
        # Apply the Haversine formula to calculate the spatial difference
        dist.append(haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS))
    # First element has distance of 0 meters
    return [0.0] + dist


def calculate_time_differences(df):
    # Calculate time differences between consecutive rows
    time_diff = df[TIMESTAMP].diff().dt.total_seconds().astype("float64")
    # Replace null first element with time difference of 0
    time_diff[0] = 0
    return time_diff
