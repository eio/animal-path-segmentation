import pandas as pd
from sklearn.model_selection import train_test_split

# Setup paths for data output
TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

# Load the CSV file into a dataframe
data = pd.read_csv("Cranes_processed.csv")

# Split the data into training and validation sets
size = 0.2  # use 20% of the data for validation
train_data, validation_data = train_test_split(
    data,
    test_size=size,
    random_state=42,  # you know why
)

# Save the training and validation data to separate CSV files
train_data.to_csv(TRAIN_CSV, index=False)
validation_data.to_csv(VALIDATION_CSV, index=False)
