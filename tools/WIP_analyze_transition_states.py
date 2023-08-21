from datetime import timedelta
import pandas as pd
import argparse
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
# Local scripts
from utils.consts import *

# The number of days (+/-) surrounding a transition time
# to build the TIMESTAMP dataframe filter
TRANSITION_DAYS_DELTA = 10


def collect_transition_states(filepath, status_column):
    # 1. Load CSV data from filepath into pd.df
    df = pd.read_csv(filepath)
    # 2. Group by ID_YEAR column
    grouped_df = df.groupby(ID_YEAR)
    # 3. Order by TIMESTAMP column
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    ordered_df = df.sort_values(TIMESTAMP)
    # 4. Group the df further into transition windows
    transition_windows = []
    for _, group in grouped_df:
        transition_indices = []
        previous_status = None
        for i in range(0, len(group)):
            current_status = group.iloc[i][status_column]
            if current_status != previous_status:
                # If it's not the first iteration...
                if previous_status != None:
                    # Keep track of the transition indices
                    transition_indices.append(i)
                # Change previous status for next iteration
                previous_status = current_status
        # Process if there are at least 2 transition indices
        if len(transition_indices) >= 2:
            print("Found transitions at indices:", transition_indices)
            for i in range(len(transition_indices) - 1):
                # Get the transition's start and end rows (two adjacent indices)
                transition_start = transition_indices[i]
                transition_end = transition_indices[i + 1]
                # Set the transition timestamp using the first of the two rows
                transition_timestamp = group.iloc[transition_start][TIMESTAMP]
                # Add and subtract a time delta in days from the transition timestamp
                # to construct the start and end timestamps of the time-filter window
                window_start = transition_timestamp - timedelta(
                    days=TRANSITION_DAYS_DELTA
                )
                window_end = transition_timestamp + timedelta(
                    days=TRANSITION_DAYS_DELTA
                )
                # Filter to the transition window we care about
                # by only selecting rows with a TIMESTAMP value in the window range
                transition_window = group[
                    (group[TIMESTAMP] >= window_start)
                    & (group[TIMESTAMP] <= window_end)
                ].copy()
                # Keep track of all transition windows
                transition_windows.append(transition_window)
    # 5. Print the first 3 transition windows in the new df
    for i, window in enumerate(transition_windows[:3]):
        print(f"Transition Window {i + 1}:")
        print(window)
        print("\n" * 3)
    return transition_windows


if __name__ == "__main__":
    # Check command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True, help="Path to the CSV file")
    args = parser.parse_args()
    # Determine which column to look for the STATUS transitions
    # based on whether this script is called on the test data or predicted values
    status_column = None
    if "final_results.csv" in args.filepath:
        status_column = PREDICTED
    elif "test.csv" in args.filepath:
        status_column = STATUS
    # Call the function with the provided filepath
    transition_windows = collect_transition_states(args.filepath, status_column)
    # Create the "transitions" directory if it doesn't exist
    directory = "transitions_{}".format(status_column)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save each DataFrame in transition_windows as a CSV
    for i, df in enumerate(transition_windows):
        filename = "transition_{}_{}.csv".format(status_column, i + 1)
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
    print("{} total transitions found.".format(len(transition_windows)))
    print("Transition snippets saved to `{}/`\n".format(directory))
