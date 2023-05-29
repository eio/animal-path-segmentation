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
# Default ratio values
DEFAULT_TRAIN_RATIO = 80
DEFAULT_TEST_RATIO = 10
DEFAULT_VALIDATION_RATIO = 10


def stats():
    # Read the CSV file into a dataframe
    data = pd.read_csv(INPUT_CSV)
    # Count individual trajectories
    individual_trajectories = data[IDENTIFIER].nunique()
    # Group data by IDENTIFIER and SPECIES, and count unique combinations
    species_counts = (
        data.groupby([IDENTIFIER, SPECIES]).size().reset_index().groupby(SPECIES).size()
    )
    # Calculate species ratios
    species_ratios = species_counts / individual_trajectories * 100
    # Print the count of individual trajectories
    print("\nIndividual Trajectories:", individual_trajectories)
    # Print the count and ratio for each species
    for species, count in species_counts.items():
        ratio = species_ratios[species]
        species = species.ljust(30)
        count = str(count).ljust(5)
        ratio = "{:.2f}%".format(ratio)
        print(f"Species: {species}\tCount: {count}\tRatio: {ratio}")


def main(train_ratio, validation_ratio, test_ratio):
    # Read the CSV file into a dataframe
    data = pd.read_csv(INPUT_CSV)
    # Initialize empty datasets
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    validation_data = pd.DataFrame()
    # Establish the `train:test:validation` ratio
    train_ratio = DEFAULT_TRAIN_RATIO / 100
    test_ratio = DEFAULT_TEST_RATIO / 100
    validation_ratio = DEFAULT_VALIDATION_RATIO / 100

    # Group records by IDENTIFIER and aggregate them into trajectories
    grouped_data = data.groupby(IDENTIFIER)
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

    # Save the datasets to separate CSV files
    train_data.to_csv(TRAIN_CSV, index=False)
    test_data.to_csv(TEST_CSV, index=False)
    validation_data.to_csv(VALIDATION_CSV, index=False)

    print("\nData split and saved successfully.")

    # Print the datasets' shapes and species distribution
    print("\nTrain dataset:")
    print(train_data.shape)
    print(train_data["species"].value_counts())

    print("\nValidation dataset:")
    print(validation_data.shape)
    print(validation_data["species"].value_counts())

    print("\nTest dataset:")
    print(test_data.shape)
    print(test_data["species"].value_counts())

    # Save the datasets to separate CSV files
    train_data.to_csv(TRAIN_CSV, index=False)
    validation_data.to_csv(VALIDATION_CSV, index=False)
    test_data.to_csv(TEST_CSV, index=False)

    print("\nSaved three local CSVs:")
    print("\t" + TRAIN_CSV)
    print("\t" + TEST_CSV)
    print("\t" + VALIDATION_CSV)
    print("Finito.")


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) == 4:
        try:
            # Three numerical command-line arguments expected:
            #   1. the train ratio
            #   2. the test ratio
            #   3. the validation ratio
            #   - Example:
            #              python split_train_test.py 80 10 10
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

    print("Train ratio: {}%".format(train_ratio))
    print("Test ratio: {}%".format(test_ratio))
    print("Validation ratio: {}%".format(validation_ratio))
    # Print stats on the data
    stats()
    # Run the main function
    main(
        train_ratio,
        validation_ratio,
        test_ratio,
    )
