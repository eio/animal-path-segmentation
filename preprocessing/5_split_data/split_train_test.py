import pandas as pd
import numpy as np
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *

# Input: Labeled events data with normalized features
INPUT_CSV = "Cranes_normalized.csv"
# Outputs:
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
VALIDATION_CSV = "validation.csv"
OUTPUT_STATS = "output_stats.txt"
# Default ratio values
DEFAULT_TRAIN_RATIO = 80
DEFAULT_VALIDATION_RATIO = 10
DEFAULT_TEST_RATIO = 10
# Common split ratios: 70-15-15, 80-10-10, 60-10-30


def write_and_print(file, text):
    # Write to the specified output file
    file.write(text)
    # Also print to stdout
    print(text.rstrip())


def write_dataset_info(file, dataset_name, dataset):
    write_and_print(file, f"\n-------------------------")
    write_and_print(file, f"\n{dataset_name} dataset:\n")
    write_and_print(file, f"Waypoints: {dataset.shape[0]}\n")
    write_and_print(file, f"Trajectories: {dataset[ID_YEAR].nunique()}\n")
    species_counts = dataset[SPECIES].value_counts()
    unique_id_years = dataset.groupby(SPECIES)[ID_YEAR].nunique()
    species_info = pd.concat([species_counts, unique_id_years], axis=1)
    species_info.columns = ["Waypoints", "Trajectories"]
    write_and_print(file, "Species distribution:\n")
    write_and_print(file, species_info.to_string())
    write_and_print(file, "\n")


def stats(df, train_data, validation_data, test_data):
    # Count individual trajectories
    individual_trajectories = df[ID_YEAR].nunique()
    # Group data by IDENTIFIER and SPECIES, and count unique combinations
    species_counts = (
        df.groupby([ID_YEAR, SPECIES]).size().reset_index().groupby(SPECIES).size()
    )
    # Calculate species ratios
    species_ratios = species_counts / individual_trajectories * 100
    # Print the count and ratio for each species (and write to output text file)
    with open(OUTPUT_STATS, "w") as file:
        # Print the split ratios being used
        write_and_print(file, "Train ratio: {}%\n".format(train_ratio))
        write_and_print(file, "Test ratio: {}%\n".format(test_ratio))
        write_and_print(file, "Validation ratio: {}%\n".format(validation_ratio))
        # Print the count of individual trajectories
        write_and_print(
            file,
            "\nTotal Individual Trajectories: {}\n".format(individual_trajectories),
        )
        for species, count in species_counts.items():
            ratio = species_ratios[species]
            species = species.ljust(30)
            count = str(count).ljust(5)
            ratio = "{:.2f}%".format(ratio)
            line = f"Species: {species}\tTrajectories: {count}\tRatio: {ratio}\n"
            write_and_print(file, line)
        # Print additional details
        write_dataset_info(file, "Train", train_data)
        write_dataset_info(file, "Validation", validation_data)
        write_dataset_info(file, "Test", test_data)


def main(df, train_ratio, validation_ratio, test_ratio):
    # Initialize empty datasets
    train_data = pd.DataFrame()
    validation_data = pd.DataFrame()
    test_data = pd.DataFrame()
    # Establish the `train:validation:test` ratio
    train_ratio = DEFAULT_TRAIN_RATIO / 100
    validation_ratio = DEFAULT_VALIDATION_RATIO / 100
    test_ratio = DEFAULT_TEST_RATIO / 100
    # Group records by IDENTIFIER and aggregate them into trajectories
    grouped_data = df.groupby(IDENTIFIER)
    trajectories = [group for _, group in grouped_data]
    # Get total number of individual animal trajectories
    total_records = len(trajectories)
    # Assign each trajectory to higher-level groups based on SPECIES
    grouped_trajectories = {}
    for trajectory in trajectories:
        species = trajectory[SPECIES].iloc[0]
        if species not in grouped_trajectories:
            grouped_trajectories[species] = []
        grouped_trajectories[species].append(trajectory)
    # Split trajectories into train, test, and validation datasets
    for species_trajectories in grouped_trajectories.values():
        species_records = sum(
            len(trajectories) for trajectories in species_trajectories
        )
        species_train_count = int(species_records * train_ratio)
        species_test_count = int(species_records * test_ratio)
        species_validation_count = int(species_records * validation_ratio)
        # Make a copy
        species_trajectories_copy = species_trajectories.copy()
        while species_train_count > 0 and len(species_trajectories_copy) > 0:
            trajectories = species_trajectories_copy.pop(0)
            train_data = pd.concat([train_data, trajectories])
            species_train_count -= len(trajectories)
        while species_test_count > 0 and len(species_trajectories_copy) > 0:
            trajectories = species_trajectories_copy.pop(0)
            test_data = pd.concat([test_data, trajectories])
            species_test_count -= len(trajectories)
        while species_validation_count > 0 and len(species_trajectories_copy) > 0:
            trajectories = species_trajectories_copy.pop(0)
            validation_data = pd.concat([validation_data, trajectories])
            species_validation_count -= len(trajectories)
    # If any of the datasets still require more records, add remaining trajectories
    if species_train_count > 0:
        for species_trajectories in grouped_trajectories.values():
            while species_train_count > 0 and len(species_trajectories) > 0:
                trajectories = species_trajectories.pop(0)
                train_data = pd.concat([train_data, trajectories])
                species_train_count -= len(trajectories)
            if species_train_count <= 0:
                break
    if species_test_count > 0:
        for species_trajectories in grouped_trajectories.values():
            while species_test_count > 0 and len(species_trajectories) > 0:
                trajectories = species_trajectories.pop(0)
                test_data = pd.concat([test_data, trajectories])
                species_test_count -= len(trajectories)
            if species_test_count <= 0:
                break
    if species_validation_count > 0:
        for species_trajectories in grouped_trajectories.values():
            while species_validation_count > 0 and len(species_trajectories) > 0:
                trajectories = species_trajectories.pop(0)
                validation_data = pd.concat([validation_data, trajectories])
                species_validation_count -= len(trajectories)
            if species_validation_count <= 0:
                break
    # Print stats on the split data (and write to output file)
    stats(
        df,
        train_data,
        validation_data,
        test_data,
    )
    # Save the datasets to separate CSV files
    train_data.to_csv(TRAIN_CSV, index=False)
    validation_data.to_csv(VALIDATION_CSV, index=False)
    test_data.to_csv(TEST_CSV, index=False)
    print("\n\nData split and saved successfully.")
    print("Saved three local CSVs:")
    print("\t" + TRAIN_CSV)
    print("\t" + VALIDATION_CSV)
    print("\t" + TEST_CSV)
    print("Finito.\n")


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) == 4:
        try:
            # Three numerical command-line arguments expected:
            #   1. the train ratio
            #   2. the test ratio
            #   3. the validation ratio
            #   - Example:
            #              python split_train_test.py 60 10 30
            #
            train_ratio = int(sys.argv[1])
            test_ratio = int(sys.argv[2])
            validation_ratio = int(sys.argv[3])
        except ValueError:
            print("Invalid input. Using default ratios:")
            train_ratio = DEFAULT_TRAIN_RATIO
            test_ratio = DEFAULT_TEST_RATIO
            validation_ratio = DEFAULT_VALIDATION_RATIO
    else:
        print("No ratio arguments provided. Using default ratios:")
        train_ratio = DEFAULT_TRAIN_RATIO
        test_ratio = DEFAULT_TEST_RATIO
        validation_ratio = DEFAULT_VALIDATION_RATIO

    # Read the CSV file into a dataframe
    df = pd.read_csv(INPUT_CSV)
    # Add ID_YEAR column to serve as a trajectory ID
    df[ID_YEAR] = df[IDENTIFIER].astype(str) + "-" + df[YEAR].astype(str)
    # Run the main function
    main(
        df,
        train_ratio,
        validation_ratio,
        test_ratio,
    )
