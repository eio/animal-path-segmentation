# This script adds an "icons" column for https://kepler.gl visualization
import pandas as pd

# Load CSV file into a pandas DataFrame
csv_file_path = "final_results.csv"
data = pd.read_csv(csv_file_path)

# Define a function to assign icon values based on correctness
def assign_icon(correct):
    if correct == True:
        return "plus-alt"
    elif correct == False:
        return "delete"
    else:
        return None


# Add a new 'icon' column using the assign_icon function
data["icon"] = data["Correct"].apply(assign_icon)

# Save the modified DataFrame back to a CSV file (optional)
output_csv_file_path = "results_true_false_icons.csv"
data.to_csv(output_csv_file_path, index=False)
