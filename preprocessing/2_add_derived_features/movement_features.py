import pandas as pd
import numpy as np
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *

# Planet Earth's average radius in meters
EARTH_RADIUS = 6371 * 1000  # meters


def derive_movement_features(df):
    # Group by individual trajectories
    trajectories = df.groupby([IDENTIFIER])
    # Calculate velocity, bearing, and turn angle across all waypoints for each trajectory
    df = trajectories.apply(calculate_velocity_bearing_turn).reset_index(
        drop=True  # reset index to obtain a new DataFrame with same shape as original one
    )
    return df


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
    # Calculate velocity, bearing, and turn angle
    velocity = calculate_velocity(dist, time_diff)  # meters/second
    bearing = calculate_bearing(lat_rad, lon_rad)
    turn_angle = calculate_turn_angle(bearing)
    # Add (or update) df columns:
    df[DISTANCE] = dist
    df[VELOCITY] = velocity
    df[BEARING] = bearing
    df[TURN_ANGLE] = turn_angle
    # Fill NaN values with 0
    df.fillna(0, inplace=True)
    return df


def calculate_spatial_differences(lat_rad, lon_rad):
    """
    Use the Haversine formula to calculate the spatial difference
    between consecutive trajectory coordinates in radians
    Formula: https://www.movable-type.co.uk/scripts/latlong.html
    """
    # Calculate spatial differences between consecutive rows
    lat_diff = lat_rad.diff()
    lon_diff = lon_rad.diff()
    # Apply the Haversine formula to calculate the spatial difference
    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(lat_rad.shift()) * np.cos(lat_rad) * np.sin(lon_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = EARTH_RADIUS * c
    # Set first element to 0 meters
    dist.iloc[0] = 0
    return dist


def calculate_time_differences(df):
    """
    Calculate time differences between consecutive trajectory waypoints
    """
    time_diff = df[TIMESTAMP].diff().dt.total_seconds().astype("float64")
    # Replace null first element with time difference of 0
    time_diff[0] = 0
    return time_diff


def calculate_velocity(dist, dt):
    """
    Calculate velocity across consecutive trajectory waypoints
    """
    velocity = dist / dt
    return velocity


def calculate_bearing(lat_rad, lon_rad):
    """
    Calculate bearing for each trajectory waypoint.
    Args:
        lat_rad (numpy.ndarray): Array of latitude values in radians.
        lon_rad (numpy.ndarray): Array of longitude values in radians.
    Returns:
        numpy.ndarray: Array of bearing values in the range of -180 to 180 degrees.
    """
    # Pad with last value to ensure shapes match
    lon_diff = np.diff(np.concatenate([lon_rad, [lon_rad.iloc[-1]]]))
    # Calculate the bearing
    x = np.cos(lat_rad) * np.sin(lon_diff)
    y = (np.sin(lat_rad) * np.cos(lat_rad)) - (
        np.cos(lat_rad) * np.sin(lat_rad) * np.cos(lon_rad - lon_rad.iloc[0])
    )
    bearing = np.degrees(np.arctan2(x, y)) % 360
    # Convert bearing values in the range of 0 to 360 to -180 to 180
    bearing[bearing > 180] -= 360
    return bearing


def calculate_turn_angle(bearing):
    """
    Calculate the turning angle for each trajectory waypoint.
    Args:
        bearing (numpy.ndarray): Array of bearing values in the range of -180 to 180 degrees.
    Returns:
        numpy.ndarray: Array of turn angles corresponding to each waypoint.
    Raises:
        ValueError: If the input bearing values are outside the valid range.
    """
    if not np.all((-180 <= bearing) & (bearing <= 180)):
        raise ValueError(
            "Bearing values should be in the range of -180 to 180 degrees."
        )
    turn_angle = np.degrees(
        np.arctan2(
            np.sin(np.radians(bearing - np.roll(bearing, 1))),
            np.cos(np.radians(bearing - np.roll(bearing, 1))),
        )
    )
    # Fill NaN values with 0
    turn_angle[np.isnan(turn_angle)] = 0
    return turn_angle
