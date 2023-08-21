import pandas as pd

ID_YEAR = "542102791-2019"  # Replace with the value you're looking for
print("ID_YEAR:\t{}\n".format(ID_YEAR))

# Step 1: Load CSV data
input_csv_path = "final_results.csv"  # Change this to your input CSV file path
data = pd.read_csv(input_csv_path)

# Step 2: Filter data based on "identifier-year" column
filtered_data = data[data["identifier-year"] == ID_YEAR]

# Step 3: Output filtered data as a new CSV
output_csv_path = "filtered_output_{}.csv".format(ID_YEAR)
filtered_data.to_csv(output_csv_path, index=False)

# Count the total number of filtered rows
total_filtered_rows = len(filtered_data)
print(f"Total filtered rows: {total_filtered_rows}")

# Count the number of True and False values in the "Correct" column
correct_counts = filtered_data["Correct"].value_counts()
num_true = correct_counts.get(True, 0)
num_false = correct_counts.get(False, 0)
print(f"Number of True values: {num_true}")
print(f"Number of False values: {num_false}")

print(
    "\n{}/{} = {}".format(num_true, total_filtered_rows, num_true / total_filtered_rows)
)
