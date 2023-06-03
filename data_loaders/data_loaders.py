from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
# Local scripts
import data_loaders.AnimalPathsDataset as APD

# During the training process, shuffle the order
# individual animal trajectories are presented to the model
# rather than always showing the records in the same order.
TRAIN_SHUFFLE = True
TEST_SHUFFLE = False
# Setup paths for accessing data
TRAIN_CSV = "data/train.csv"
VALIDATION_CSV = "data/validation.csv"
FINAL_TEST_CSV = "data/test.csv"


def build_data_loaders(batch_size):
    print("Building datasets...")
    # Create the Train dataset
    train_dataset = APD.AnimalPathsDataset(
        csv_file=TRAIN_CSV,
    )
    print("\tTraining dataset built.")
    # Create the Test dataset
    validation_dataset = APD.AnimalPathsDataset(
        csv_file=VALIDATION_CSV,
    )
    print("\tValidation dataset built.")
    print("Building data loaders...")
    # Build the Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=TRAIN_SHUFFLE,
    )
    print("\tTraining loader built.")
    # Build the Test loader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=TEST_SHUFFLE,
    )
    print("\tValidation loader built.")
    return {
        "train": train_loader,
        "test": validation_loader,
    }


def build_final_test_data_loader(batch_size):
    """ Final, unlabeled, test dataset """
    print("Building dataset...")
    dataset = APD.AnimalPathsDataset(
        csv_file=FINAL_TEST_CSV,
    )
    print("Building data loader...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=TEST_SHUFFLE,
    )
    return loader


if __name__ == "__main__":
    # Test creation of the loaders
    loaders = build_data_loaders(1)
    for k, v in loaders.items():
        print("{}: {}".format(k, v))
