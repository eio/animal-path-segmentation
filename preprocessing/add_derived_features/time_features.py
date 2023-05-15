import math
import pandas as pd
from datetime import datetime

# Define strings for column/feature names
SINTIME = "sin_time"
COSTIME = "cos_time"
TIMESTAMP = "timestamp"
MONTH = "month"
DAY = "day"
# Seconds in a year (i.e., 365.25 * 24 * 60 * 60)
SECONDS_IN_YEAR = 31_536_000


def cyclic_time(dt):
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
    period = SECONDS_IN_YEAR
    # Calculate cylic time
    angle = 2 * math.pi * seconds_since_year_start / period
    sin_time = math.sin(angle)
    cos_time = math.cos(angle)
    return sin_time, cos_time


def transform_time_features(df):
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
    df[[SINTIME, COSTIME]] = df[TIMESTAMP].apply(lambda x: pd.Series(cyclic_time(x)))
    # Return the transformed dataframe
    return df
