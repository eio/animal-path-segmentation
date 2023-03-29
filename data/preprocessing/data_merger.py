import os
import glob
import pandas as pd

# Data filepaths
CRANE_EVENTS = "crane_events_20220223.csv"
SEGMENTATIONS = "segmentations"
# Unique status strings: `df.Status.unique()`
POSSIBLE_STATES = [
    "Start Fall",
    "End Fall",
    "Start Spring",
    "End Spring",
    "Presumed Start Fall",
    "Presumed End Fall",
    "Presumed Start Spring",
    "Presumed End Spring",
    # Stopovers are called out.
    # For now, maybe gloss over these
    #   (or make a secondary behavioral state that captures those.)
    "Start Stopover",
    "End Stopover",
    # This status value is not present in any of the usable data
    # after the un-usable data has been purged
    "No migration",
    # Records with `Not enough data` are purged on load
    "Not enough data",
]


def load_segmentation_data():
    """
    Look for CSV files containing Spire API responses
    in a local directory called `spire_api_reponses/`
    with headers of:
    headers = ["", "Identifier", "Species", "Individual", "Status", "Date"]
    # "Date" = the transition date from one status to another
    # Note: if the timeseries starts/ends mid-season (basically always does)
    # you only have the end/start of a season.
    """
    print("Loading labeled data...")
    # Get all the local CSV files with labels
    # Unique species with labels:
    #   ['Anthropoides paradiseus', 'Anthropoides virgo', 'Grus grus', 'Grus nigricollis', 'Grus vipio']
    #   ...and also ['Balearica pavonina'] but there's "Not enough data" for any of those records.
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
    # Convert Date field to datetime type
    df["Date"] = pd.to_datetime(df["Date"])
    # Return the DataFrame
    df.to_csv("SEGMENTS_TRIMMED.csv", index=False)
    return df


def load_events_data():
    """
    Important headers:
        lat, lon, event_id, individual_id, timestamp, tag_id, taxon_canonical_name, species
    Unique species with events: `list(df.species.unique())`
        ['Anthropoides paradiseus', 'Anthropoides virgo', 'Grus grus', 'Grus nigricollis', 'Grus vipio']
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
        ]
    ]
    # Clean up data we can't use (safety check)
    df = df[(df.lat != None) & (df.lon != None) & (df.timestamp != None)]
    # Convert Timestamp field to datetime type
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Save cropped events data
    df.to_csv("EVENTS_TRIMMED.csv", index=False)
    return df


def add_status_label_to_events(df_events, df_labels):
    """
    Use the "timestamp" field in `df_events`
    and the "Date" field in `df_labels`
    to find the correct `Status` label for each event
    """
    print("Adding `Status` label to events...")
    # Sort the Date column in df_labels in ascending order
    df_labels = df_labels.sort_values(["Individual", "Date"])
    # Group the dataframes by the identifier column
    df_events_grouped = df_events.groupby("individual_id")
    df_labels_grouped = df_labels.groupby("Individual")
    # Perform the merge_asof on the timestamps only for the records that share the same identifier
    merged_dfs = []
    for identifier, df_events_subset in df_events_grouped:
        if identifier in df_labels_grouped.groups:
            # Note that this implementation only merges the events
            # that have at least one corresponding label with the same identifier.
            # To include events that don't have a corresponding label,
            # we can modify the code to use an outer join instead of a left join.
            df_labels_subset = df_labels_grouped.get_group(identifier)
            merged_df = pd.merge_asof(
                df_events_subset,
                df_labels_subset,
                left_on=["timestamp"],  # example: "2011-07-27 07:30:13.999"
                right_on=["Date"],  # example val: "2021-03-05"
                direction="backward",
            )
            # Forward fill the status values for missing rows
            merged_df["Status"] = merged_df["Status"].fillna(method="ffill")
            # Store the merged dataframe
            merged_dfs.append(merged_df)
    # Concatenate the merged dataframes,
    # remove rows without label matches,
    # and return the final merged dataframe
    if len(merged_dfs) > 0:
        # Concatenate the dataframes
        df = pd.concat(merged_dfs)
        # TODO: Decide if we really want to purge unmatched data
        # or if we want to set some kind of Default label instead...
        # For now: clean up data without matches
        df = df.dropna(subset=["Status"])
        # Remove the (now duplicate) `Individual` column
        df = df.drop(columns=["Individual"])
        # Return the final merged dataframe
        return df
    else:
        return None


def build_confidence_column(df):
    """
    Remove the "Presumed " sub-string from `Status` values
    and replace with a binary integer `Confidence` column.
    i.e. Select rows with "Presumed" in `Status` column,
    add a `Confidence` value of 0 to indicate it's presumed
    and remove the "Presumed " text.
    Otherwise, if not "Presumed", add `Confidence` value of 1.
    """
    print("Building `Confidence` column...")
    # Update the confidence and Status columns for rows that contain "Presumed" in Status
    def update_row(x):
        # Check if "Presumed" is in the Status value of the group
        is_presumed = any("Presumed" in status for status in x["Status"])
        # Add a binary integer `Confidence` column
        # instead of the "Presumed" status fields
        if is_presumed:
            x["Confidence"] = 0
            x["Status"] = x["Status"].str.replace("Presumed ", "")
        else:
            # If `Status` value does not contain "Presumed" substring
            # then we set `Confidence` value to 1
            x["Confidence"] = 1
        return x

    # Group the data by the `Status` column
    grouped = df.groupby("Status", group_keys=False)
    # Apply the data transformation, then
    # reset the dataframe index afterwards
    df = grouped.apply(update_row).reset_index(drop=True)
    # Return processed dataframe
    return df


def reduce_status_values(df):
    """
    Replace `Status` fields like "Start Spring" and "End Spring"
    with a single value of "Spring" for the relevant timestamps
    """
    print("Reducing `Status` column values...")
    # Process the data
    def update_row(x):
        # Filter rows with "End " in Status column
        end_rows = x.loc[x["Status"].str.contains("End")]
        # Sort by timestamp in ascending order
        end_rows = end_rows.sort_values("timestamp")
        # Keep only the first row and clear Status for the others
        keep_row = end_rows.iloc[0]
        # Remove "End " substring from the row we're keeping
        x.loc[x["timestamp"] == keep_row["timestamp"], "Status"] = keep_row[
            "Status"
        ].replace("End ", "")
        # Replace all other "End " values with... nothing...?
        x.loc[x["Status"].str.contains("End"), "Status"] = "no_label"
        return x

    # Group the data by the `individual_id` column
    grouped = df.groupby("individual_id", group_keys=False)
    # Apply the data transformation, then
    # reset the dataframe index afterwards
    df = grouped.apply(update_row).reset_index(drop=True)
    # Now that all `Status` values with the "End" substring are adjusted
    # remove the "Start " substring from all `Status` values
    df["Status"] = df["Status"].str.replace("Start ", "")
    # TODO: Decide if we really want to purge unmatched data
    # or if we want to set some kind of Default label instead...
    df = df[df.Status != "no_label"]
    # Return processed dataframe
    return df


def stats(df):
    """
    Print some stats on the processed data
    """
    print()
    print(df)
    print()
    print(df["Status"].value_counts())
    print()


if __name__ == "__main__":
    print()
    # Set the output file for the processed data
    output_file = "PROCESSED_OUTPUT.csv"
    # Load the locally stored CSV data (labels and events)
    df_labels = load_segmentation_data()
    df_events = load_events_data()
    print("\nData loaded successfully.\nNow transforming data...\n")
    # Add the `Status` label field to the events data
    df = add_status_label_to_events(df_events, df_labels)
    # Replace "Presumed" `Status` values with binary integer `Confidence` column
    df = build_confidence_column(df)
    # Replace "Start" and "End" status values with a simplified `Status` tag
    df = reduce_status_values(df)
    # Output the data in CSV format
    df.to_csv(output_file, index=False)
    # Print some quick stats
    stats(df)

    print("Done.")
