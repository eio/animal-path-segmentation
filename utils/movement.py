import numpy as np

# Assign each animal movement event to a "burst"
# (assuming a burst ends if the time difference is
# greater than some threshold, e.g. 300 seconds)
BURST_TIME_THRESHOLD = 300  # seconds

# Approximate Earth radius in km
EARTH_RADIUS = 6371  # kilometers

# Define strings for column/feature names
LATITUDE = "lat"
LONGITUDE = "lon"
TIMESTAMP = "timestamp"
VELOCITY = "velocity"
BEARING = "bearing"
TURN_ANGLE = "turn_angle"


def calculate_velocity_bearing_turn(df):
    # Convert latitude and longitude to radians
    lat_rad = np.radians(df[LATITUDE])
    lon_rad = np.radians(df[LONGITUDE])
    # Calculate spatial differences
    dist = calculate_spatial_differences(df)
    # Calculate time differences
    dt = np.diff(df[TIMESTAMP])
    # Identify the bursts
    burst_indices = identify_bursts(df)
    # Calculate velocity, bearing, and turn angle for each burst
    for i in range(len(burst_indices) - 1):
        start_idx = burst_indices[i]
        end_idx = burst_indices[i + 1]
        if end_idx >= len(df):
            # Exit the loop if the end index is too high
            break
        # Calculate velocity, bearing, and turn angle:
        vel = calculate_velocity(dist[start_idx:end_idx], dt[start_idx:end_idx])
        bearing = calculate_bearing(
            lat_rad[start_idx : end_idx + 1], lon_rad[start_idx : end_idx + 1]
        )
        turn_angle = calculate_turn_angle(bearing)
        # Check if the length of the slice of the dataframe
        # and the length of the turn_angle array are the same
        if len(df.loc[start_idx + 1 : end_idx]) != len(turn_angle):
            raise ValueError("Length of slice and turn_angle array are not the same")
        # Add velocity, bearing, and turn angle columns to the dataframe
        print("\n\nturn_angle: ", turn_angle.shape)
        print("\n\nbearing: ", bearing.shape)
        print("indices:", start_idx + 1, end_idx)
        print("df.loc:", df.loc[start_idx + 1 : end_idx])
        print()
        df.loc[start_idx + 1 : end_idx, VELOCITY] = vel
        df.loc[start_idx + 1 : end_idx, BEARING] = bearing[:-1]
        df.loc[start_idx + 1 : end_idx, TURN_ANGLE] = turn_angle
    # Fill NaN values with 0
    df.fillna(0, inplace=True)
    return df


def identify_bursts(df):
    dt = np.diff(df[TIMESTAMP])
    burst_indices = [0]
    for i in range(1, len(dt)):
        if dt[i - 1] > BURST_TIME_THRESHOLD:
            burst_indices.append(i)
    return burst_indices


def calculate_velocity(dist, dt):
    dt = dt.astype("timedelta64[s]").astype("float64")
    return dist / dt


def calculate_bearing(lat_rad, lon_rad):
    x = np.cos(lat_rad) * np.sin(np.diff(lon_rad))
    y = (np.sin(lat_rad) * np.cos(lat_rad[1:])) - (
        np.cos(lat_rad) * np.sin(lat_rad[1:]) * np.cos(np.diff(lon_rad))
    )
    # Handle wrap-around at the international dateline
    bearing = np.degrees(np.arctan2(x, y)) % 360
    return bearing


def calculate_turn_angle(bearing):
    turn_angle = np.degrees(
        np.arccos(np.clip(np.cos(np.radians(bearing - np.roll(bearing, 1))), -1, 1))
    )
    # Fill NaN values with 0
    turn_angle[np.isnan(turn_angle)] = 0
    return turn_angle[0]  # only return the first element of the array


def calculate_spatial_differences(df):
    lat_rad = np.radians(df[LATITUDE])
    lon_rad = np.radians(df[LONGITUDE])
    x = np.diff(lon_rad)
    y = np.diff(lat_rad)
    dist = np.sqrt(
        np.power(np.cos(lat_rad[:-1]) * np.sin(y / 2), 2)
        + np.power(np.sin(x / 2), 2) * np.power(np.sin(y / 2), 2)
    )
    return dist * EARTH_RADIUS
