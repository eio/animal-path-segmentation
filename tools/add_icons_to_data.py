# This script adds an "icons" column for https://kepler.gl visualization
import pandas as pd

# Load CSV file into a pandas DataFrame
csv_file_path = "Cranes_labeled.csv"
data = pd.read_csv(csv_file_path)

# Define a function to assign icon values based on species
def assign_icon(species):
    if species == "Anthropoides virgo":
        return "plus"
    elif species == "Grus grus":
        return "delete"
    elif species == "Grus vipio":
        return "star"
    elif species == "Grus nigricollis":
        return "circle"
    elif species == "Anthropoides paradiseus":
        return "play"
    else:
        return None


# Add a new 'icon' column using the assign_icon function
data["icon"] = data["species"].apply(assign_icon)

# Save the modified DataFrame back to a CSV file (optional)
output_csv_file_path = "with_icons_column.csv"
data.to_csv(output_csv_file_path, index=False)
