import pandas as pd

# Step 1: Load CSV data
input_csv_path = "final_results.csv"  # Change this to your input CSV file path
data = pd.read_csv(input_csv_path)

# Count the total number of filtered rows
total_rows = len(data)
print(f"Total rows: {total_rows}")

# Count the number of True and False values in the "Correct" column
correct_counts = data["Correct"].value_counts()
num_true = correct_counts.get(True, 0)
num_false = correct_counts.get(False, 0)
print(f"Number of True values: {num_true}")
print(f"Number of False values: {num_false}")

print("\n{}/{} = {}".format(num_true, total_rows, num_true / total_rows))
