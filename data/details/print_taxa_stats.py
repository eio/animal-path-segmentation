import pandas as pd
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
# Local scripts
from utils.consts import *

DOWNSAMPLED = True
df = None
UNIQUE_ID = None

if DOWNSAMPLED == True:
    # Get stats on daily downsampled positions
    df = pd.read_csv("Cranes_downsampled_all_features.csv")
    # Add ID_YEAR column to serve as a trajectory ID
    df[ID_YEAR] = df[IDENTIFIER].astype(str) + "-" + df[YEAR].astype(str)
    UNIQUE_ID = ID_YEAR
else:
    # Get stats on all original waypoints
    df = pd.read_csv("Cranes_labeled.csv")
    UNIQUE_ID = IDENTIFIER

# Determine the unique sequence IDs
unique_individual_ids = df[UNIQUE_ID].unique()
# Sort the unique individual_ids based on the species value
sorted_individual_ids = sorted(
    unique_individual_ids, key=lambda x: df[df[UNIQUE_ID] == x][SPECIES].iloc[0]
)

current_species = None
species_record_count = 0

for individual_id in sorted_individual_ids:
    species = df[df[UNIQUE_ID] == individual_id][SPECIES].iloc[0]
    record_count = df[df[UNIQUE_ID] == individual_id].shape[0]

    if species != current_species:
        if current_species is not None:
            print(
                f">>> Total record count for ({current_species}):  {species_record_count} records"
            )
            print()
        current_species = species
        species_record_count = record_count

    print(f"{individual_id} - ({species})  -  {record_count} records")
    species_record_count += record_count

# Print total record count for the last species
if current_species is not None:
    print(
        f">>> Total record count for ({current_species}):  {species_record_count} records"
    )
