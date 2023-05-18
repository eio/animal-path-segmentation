import pandas as pd
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
# Local scripts
from utils.consts import STATUS

if __name__ == "__main__":
    # Check if a filename is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide a CSV filename as a command-line argument.")
        sys.exit(1)

    # Read the CSV file into a pandas DataFrame
    filename = sys.argv[1]
    df = pd.read_csv(filename)

    # Calculate the count and percentage of each unique STATUS value
    status_counts = df[STATUS].value_counts()
    total_records = len(df)
    status_percentages = (status_counts / total_records) * 100

    # Print the stats
    print()
    for status, count in status_counts.items():
        percentage = status_percentages[status]
        print(f"{status}:\t {count}  ({percentage:.2f}%)")
    print()
