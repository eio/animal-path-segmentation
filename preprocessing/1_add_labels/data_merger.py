import pandas as pd
from tqdm.auto import tqdm

# Local scripts
from utils import *

# Initialize progress bar for Dataframe actions
tqdm.pandas()


def build_confidence_column(df):
    """
    Remove the "Presumed " sub-string from `Status` values
    and replace with a binary integer `Confidence` column.
    if "Presumed": `Confidence` = 0; else: `Confidence` = 1
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


def build_states_dataframe(df):
    """
    Convert an input dataframe of explicit and implicit state labels
    into a transformed dataframe with explicit and discrete states
    """
    print("Transforming labels data...")
    # Time delta shortcut
    ONE_DAY = pd.Timedelta(1, "d")

    # Function to add implicit Winter or Summer states
    def make_implicit_label(label):
        new_status = None
        # Add Summer labels
        if label.Status == "End Spring":
            new_date = label.Date + ONE_DAY
            new_status = "Start Summer"
        elif label.Status == "Start Fall":
            new_date = label.Date - ONE_DAY
            new_status = "End Summer"
        # Add Winter labels
        elif label.Status == "End Fall":
            new_date = label.Date + ONE_DAY
            new_status = "Start Winter"
        elif label.Status == "Start Spring":
            new_date = label.Date - ONE_DAY
            new_status = "End Winter"
        # Check if a new status was found
        if new_status:
            # First create a deep copy of the explicit label
            # with columns: Individual,Status,Date,Confidence
            new_label = label.copy(deep=True)
            # Then replace the relevant cells in the copy
            new_label.Status = new_status
            new_label.Date = new_date
            return new_label
        # Otherwise this is a stopover
        # and no new state label is needed
        else:
            return None

    # Function applied to each group of rows
    def transform_status_values(individual, labels):
        # Output dataframe
        df_output = pd.DataFrame(
            columns=[
                "individual",
                "confidence",
                "start",
                "end",
                "label",
            ],
        )
        # Sort the group's records chronologically
        labels = labels.sort_values(["Date"]).reset_index(drop=True)
        # First create a deep copy of the explicit labels
        new_labels = labels.copy(deep=True)
        # Iterate through each label
        for index, explicit_label in labels.iterrows():
            implicit_label = make_implicit_label(explicit_label)
            # If there's an associated implicit summer/winter label, add it
            if isinstance(implicit_label, pd.Series):
                # Add the new summer/winter label
                new_labels.loc[len(new_labels)] = implicit_label
        # Reorder new labels by date, and reset the index
        new_labels = new_labels.sort_values(["Date"]).reset_index(drop=True)
        # Iterate through each label
        for index, label in new_labels.iterrows():
            date = label.Date
            status = label.Status
            confidence = label.Confidence
            # Initialize the state
            state = {
                "start": pd.NaT,
                "end": pd.NaT,
                "confidence": confidence,
                "label": clean_status(status),
                "individual": individual,
            }
            # Find the label match (if there is one)
            if "Start" in status:
                state["start"] = date
                end_status = status.replace("Start", "End")
                # "Possible ends": records with the correct `Status`,
                # but we'll still need to test the `Date`
                pe = new_labels.loc[new_labels["Status"].str.contains(end_status)]
                # Get the matching end time
                if len(pe) > 0:
                    # Only get dates that are later (or the same), since earlier can't be relevant
                    later_dates = pe.loc[pe["Date"] >= date]
                    # Check if there's still potential matches
                    if len(later_dates) > 0:
                        # Get the closest date from the potential matches
                        match = get_closest_date(later_dates, date)
                        # Update this state's end time
                        state["end"] = match.Date
            elif "End" in status:
                state["end"] = date
                start_status = status.replace("End", "Start")
                # "Possible beginnings": records with the correct `Status`,
                # but we'll still need to test the `Date`
                pb = new_labels.loc[new_labels["Status"].str.contains(start_status)]
                # Get the matching start time
                if len(pb) > 0:
                    # Only get dates that are earlier (or the same), since later can't be relevant
                    earlier_dates = pb.loc[pb["Date"] <= date]
                    # Check if there's still potential matches
                    if len(earlier_dates) > 0:
                        # Get the closest date from the potential matches
                        match = get_closest_date(earlier_dates, date)
                        # Update this state's start time
                        state["start"] = match.Date
            # Add the new state row to the end of the output sdataframe
            df_output.loc[len(df_output)] = state
        # Return the dataframe of states
        return df_output

    # Create the output dataframe from the dictionary
    all_states = pd.DataFrame()
    # Group the data by Individual
    grouped_labels = df.groupby(["Individual"], group_keys=False)
    for individual, df_label_group in grouped_labels:
        # Get the transformed data row
        new_states_df = transform_status_values(individual, df_label_group)
        # Concatenate the original dataframe with the new row dataframe
        all_states = pd.concat([all_states, new_states_df], ignore_index=True)
    # Return the newly created dataframe
    return all_states


def add_status_label_to_events(df_events, grouped_labels):
    """
    Use the "timestamp" field in `df_events`
    and the "Date" field in `df_labels`
    to find the correct `Status` label for each event
    """
    print("Adding `Status` label to events...")
    # Get the earliest and latest events, based on `timestamp`
    # rounding to the day for matching on label `Date`
    earliest_timestamp = df_events.timestamp.min().floor("D")
    latest_timestamp = df_events.timestamp.max().floor("D")

    # Add status label to each event
    def add_status(event):
        # Round timestamp to day for matching on label `Date`
        timestamp = event.timestamp.floor("D")
        # Get the individual ID for this event
        individual = event.individual_id
        # Get the states associated with this individual
        states = grouped_labels.get_group(individual)
        # Iterate through the states
        for index, s in states.iterrows():
            start = s["start"]
            end = s["end"]
            # Check if start/end values are empty/pd.NaT
            # and replace them with earliest/latest event timestamps.
            # TODO: this is probably not safe...
            # ...should really use earliest event *per-individual*
            if pd.isna(start) or start == None:
                start = earliest_timestamp
            if pd.isna(end) or end == None:
                end = latest_timestamp
            # Check if event timestamp correlates to this state's timerange
            if timestamp >= start and timestamp <= end:
                # Change the Stopover flag if relevant
                if s["label"] == "Stopover":
                    # Stopovers are considered as a secondary state
                    # so rather than changing the seasonal label,
                    # we simply flag that the event is a stopover
                    event.stopover = 1
                else:
                    # Update the label and confidence factor for this event
                    event.status = s["label"]
                    event.confidence = s["confidence"]
                    # TODO: early exiting could miss stopovers...
                    # but the current implementation is very inefficient.
                    # Probaly, stopovers should be added first.
                    # break
        # Check that a status was added
        if event.status == None:
            print("\nNo status match:\n", event)
        # Return the updated event row
        return event

    # Add status to each event in the dataframe (with progress bar)
    # then return the transformed structure
    return df_events.progress_apply(add_status, axis=1)


if __name__ == "__main__":
    print()

    # Load the locally stored CSV - Labels data
    df_labels = load_labels_data()
    # Replace "Presumed" `Status` values with binary integer `Confidence` column
    df_labels = build_confidence_column(df_labels)
    # Add new seasonal states based on explicit and implicit `Status` vals
    df_labels = build_states_dataframe(df_labels)
    # Drop exact duplicate rows
    df_labels.drop_duplicates(inplace=True)
    df_labels.to_csv(CSV_OUT + "DataFrame_States.csv", index=False)
    # print("Transformed state labels:\n{}\n".format(df_labels))

    # Load the locally stored CSV - Events data
    df_events = load_events_data()
    # Ditch labels for individuals without events,
    # and ditch events for individuals without labels
    df_events, df_labels = ditch_unmatched_data(df_events, df_labels)
    # Group the labels by individual
    grouped_labels = df_labels.groupby(["individual"], group_keys=False)
    # Initialize new columns we're about to populate in the events data
    df_events = df_events.assign(status=None, confidence=1, stopover=0)
    # df_events.loc[:, "stopover"] = 0
    # df_events.loc[:, "confidence"] = 1
    # df_events.loc[:, "status"] = None

    # Add the labels to the events data
    df = add_status_label_to_events(df_events, grouped_labels)

    # Output the data in CSV format
    outfile = CSV_OUT + "PROCESSED_OUTPUT.csv"
    print("Saving processed data to `{}`...".format(outfile))
    df.to_csv(outfile, index=False)

    print("Done.")
