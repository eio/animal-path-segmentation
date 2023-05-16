import pandas as pd
import numpy as np
import os, sys

# Get the absolute path to the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Append the subdirectory containing the module to import to sys.path
module_dir = os.path.join(script_dir, "../../")
sys.path.append(module_dir)
# Local scripts
from consts import *

# Planet Earth's average radius in meters
EARTH_RADIUS = 6371 * 1000  # meters


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
    # Add df columns for: distance, velocity, bearing, and turn angle
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
    Calculate bearing for each trajectory waypoint
    """
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
    """
    Calculate turning angle for each trajectory waypoint
    """
    turn_angle = np.degrees(
        np.arccos(np.clip(np.cos(np.radians(bearing - np.roll(bearing, 1))), -1, 1))
    )
    # Fill NaN values with 0
    turn_angle[np.isnan(turn_angle)] = 0
    return turn_angle
