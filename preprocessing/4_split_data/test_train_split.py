import pandas as pd
from sklearn.model_selection import train_test_split

# Labeled data with normalized features
INPUT_CSV = "Cranes_normalized.csv"
# Setup paths for data output
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
VALIDATION_CSV = "validation.csv"
# Random seed
seed = 42  # you know why.

# Load the CSV file into a dataframe
data = pd.read_csv(INPUT_CSV)
# Split the complete data into training and test sets
train_data, test_data = train_test_split(
    data,
    test_size=0.2,  # use 20% of the total data for Test,
    random_state=seed,
)
# Split the TEST data further into test and validation sets
test_data, validation_data = train_test_split(
    test_data,
    test_size=0.25,  # use 25% of the TEST data for Validation (25% of 20% of total)
    random_state=seed,
)

# Save the training and validation data to separate CSV files
train_data.to_csv(TRAIN_CSV, index=False)
test_data.to_csv(TEST_CSV, index=False)
validation_data.to_csv(VALIDATION_CSV, index=False)
