import torch
from torchvision.transforms import Compose

# Local scripts
import AnimalPathsDataset as APD

# Shuffling time-series data is generally not appropriate,
# and preserving the order of the records is important
# for training an RNN on this type of data.
# TODO: since time is added as a feature, maybe shuffling can be True?
SHUFFLE = False
# Default batch size
BATCH_SIZE = 1
# Setup paths for accessing data
TRAIN_CSV = "data/ICARUS Mongolia cuckoos Nymba.csv"
# TODO: these should be different, of course
VALIDATION_CSV = "data/ICARUS Mongolia cuckoos Nymba.csv"
FINAL_TEST_CSV = "data/ICARUS Mongolia cuckoos Nymba.csv"


def build_data_loaders(batch_size=BATCH_SIZE):
    # Create the Train dataset
    train_dataset = APD.AnimalPathsDataset(
        csv_file=TRAIN_CSV,
        transform=Compose([APD.ToTensor()]),
    )
    # Create the Test dataset
    validation_dataset = APD.AnimalPathsDataset(
        csv_file=VALIDATION_CSV,
        transform=Compose([APD.ToTensor()]),
    )
    # Build the Train loader
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )
    # Build the Test loader
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )
    return {
        "train": train_loader,
        "test": validation_loader,
    }


def build_final_test_data_loader(batch_size=BATCH_SIZE):
    """ Final, unlabeled, test dataset """
    dataset = APD.AnimalPathsDataset(
        csv_file=FINAL_TEST_CSV,
        transform=Compose([APD.ToTensor()]),
    )
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
