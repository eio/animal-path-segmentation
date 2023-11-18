import os
import glob
import pandas as pd

# Data filepaths
DATA_DIR = "../../data/raw_inputs/"
# Data filepaths
CRANE_EVENTS = DATA_DIR + "crane_events_20220223.csv"
SEGMENTATIONS = DATA_DIR + "segmentations/"
CSV_OUT = "output/"


def load_labels_data():
    """
    Load, merge, and clean the CSV files with labels data
    """
    print("Loading labeled data...")
    # Get all the local CSV files with labels
    all_files = glob.glob(os.path.join(SEGMENTATIONS, "*.csv"))
    # Read all the CSV data into DataFrames
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    # Concatenate all the DataFrames into a single one
    df = pd.concat(df_from_each_file, ignore_index=True)
    # Only grab the fields we care about for now
    df = df[["Individual", "Status", "Date"]]
    # Remove rows we can't use due to missing data
    df = df[df.Status != "Not enough data"]
    df = df.dropna(subset=["Date"])
    # unique_dates = sorted(list(df.Date.unique()))
    # Convert Date field to datetime type
    df["Date"] = pd.to_datetime(df["Date"])
    # Return the DataFrame
    df.to_csv(CSV_OUT + "LABELS_trimmed.csv", index=False)
    print("Labels loaded successfully.")
    return df


def load_events_data():
    """
    Load and clean the CSV with events data
    """
    print("Loading events data...")
    # Load the complete events dataset
    df = pd.read_csv(CRANE_EVENTS)
    # Only grab the fields we care about for now
    df = df[
        [
            "lat",
            "lon",
            "timestamp",
            # "event_id",
            # "tag_id",
            "individual_id",
            # "taxon_canonical_name",
            "species",  # seems to be a dupe of taxon field
            # "bearing",  # recalculated later
        ]
    ]
    # Clean up data we can't use (safety check)
    df = df[(df.lat != None) & (df.lon != None) & (df.timestamp != None)]
    # Convert Timestamp field to datetime type
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Save cropped events data
    df.to_csv(CSV_OUT + "EVENTS_trimmed.csv", index=False)
    print("Events loaded successfully.")
    return df


def clean_status(status):
    """
    Remove the "Start " or "End " substring,
    returning either: Fall, Winter, Spring, Summer, or Stopover
    """
    if "Start" in status:
        status = status.replace("Start ", "")
    elif "End" in status:
        status = status.replace("End ", "")
    return status


def get_closest_date(possibilities, date):
    """
    From the provided possible dates,
    find the one that is closest to the input
    """
    # Find index of closest date
    idx_closest = (possibilities["Date"] - date).abs().idxmin()
    # Return the closest date
    return possibilities.loc[idx_closest]


def ditch_unmatched_data(df_events, df_labels):
    """
    Filter out data without a matching event or label
    """
    print("Remove label rows for individuals without events...")
    # Get rid of labels with no matching events
    individuals = df_events["individual_id"].unique()
    df_labels = df_labels[df_labels["individual"].isin(individuals)]
    print("Remove event rows for unlabeled individuals...")
    # Get rid of events with no matching labels
    individuals = df_labels["individual"].unique()
    df_events = df_events[df_events["individual_id"].isin(individuals)]
    return df_events, df_labels
