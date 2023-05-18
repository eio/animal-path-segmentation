import torch
from torchvision.transforms import Compose

# Local scripts
import AnimalPathsDataset as APD

# Shuffling time-series data is generally not appropriate,
# and preserving the order of the records is important
# for training an RNN on this type of data.
SHUFFLE = False
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )
    print("\tTraining loader built.")
    # Build the Test loader
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=SHUFFLE,
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
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )
    return loader


if __name__ == "__main__":
    # Test creation of the loaders
    loaders = build_data_loaders(1)
    for k, v in loaders.items():
        print("{}: {}".format(k, v))
