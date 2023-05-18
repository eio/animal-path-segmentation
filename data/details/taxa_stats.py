import pandas as pd

df = pd.read_csv("cranes_events_labeled.csv")
unique_individual_ids = df["individual_id"].unique()
# Sort the unique individual_ids based on the species value
sorted_individual_ids = sorted(
    unique_individual_ids, key=lambda x: df[df["individual_id"] == x]["species"].iloc[0]
)

current_species = None
species_record_count = 0

for individual_id in sorted_individual_ids:
    species = df[df["individual_id"] == individual_id]["species"].iloc[0]
    record_count = df[df["individual_id"] == individual_id].shape[0]

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
