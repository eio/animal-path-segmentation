import os
import subprocess

# List of preprocessing subdirectory names
subdirectories = [
    "1_add_labels",
    "2_add_derived_features",
    "3_normalize_data",
    "4_split_data",
]

# Sequentially run the code in each subdirectory
for subdir in subdirectories:
    print(f"\nExecuting `{subdir}/run.py`...\n")
    # Navigate to the subdirectory
    os.chdir(subdir)
    # Run the Python script in the current subdirectory
    subprocess.run(["python", "run.py"])
    # Return to the parent directory
    os.chdir("..")
