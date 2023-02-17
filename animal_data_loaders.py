import torch
from torchvision.transforms import Compose

# Local scripts
import AnimalPathsDataset as APD

# Setup paths for accessing data
TRAIN_CSV = "data/ICARUS Mongolia cuckoos Nymba.csv"
# TODO: this should be different, of course
VALIDATION_CSV = "data/ICARUS Mongolia cuckoos Nymba.csv"


def build_data_loaders(batch_size=1):
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
    ### Docs:
    ### https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # Build the Train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    # Build the Test loader
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return {
        "train": train_loader,
        "test": validation_loader,
    }


if __name__ == "__main__":
    # Test creation of the loaders
    loaders = build_data_loaders(1)
    print(loaders["train"])
