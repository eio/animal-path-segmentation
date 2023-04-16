import csv
import json
from numpy import argmin
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import save as torch_save, load as torch_load

# Local scripts
from AnimalPathsDataset import IDENTIFIER, FEATURE_COLUMNS
from utils import color

# Setup CSV output columns
OUTPUT_FIELDNAMES = [
    "Correct",
    "Predicted",
    "Actual",
    IDENTIFIER,
] + FEATURE_COLUMNS

# Setup output paths
OUTPUT = "output/"
SAVED_MODEL_PATH = OUTPUT + "saved_model/model+optimizer.pth"
ACCURACY_PLOT_PATH = OUTPUT + "figures/accuracy.png"
LOSS_PLOT_PATH = OUTPUT + "figures/loss.png"
CONFIG_PATH = OUTPUT + "config.json"
PREDICTIONS_DIR = OUTPUT + "predictions/"


def save_model(epoch, model, optimizer):
    """
    Save the current state of the Model
    so we can load the latest state later on
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    torch_save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        SAVED_MODEL_PATH,
    )


def load_model(model, optimizer):
    """
    Load and return the pre-trained Model,
    Optimizer, and final epoch number
    """
    print("Loading the saved model: `{}`".format(SAVED_MODEL_PATH))
    saved_state = torch_load(SAVED_MODEL_PATH)
    model.load_state_dict(saved_state["model_state_dict"])
    optimizer.load_state_dict(saved_state["optimizer_state_dict"])
    print("Model loaded.")
    return model, optimizer, saved_state["epoch"]


def write_config_json(cfg):
    """
    Write hyperparameter configuration
    to output JSON
    """
    print("Current config saved to: `{}`\n".format(CONFIG_PATH))
    with open(CONFIG_PATH, "w", newline="") as jsonfile:
        output = {
            "N_EPOCHS": cfg.N_EPOCHS,
            "HIDDEN_SIZE": cfg.HIDDEN_SIZE,
            "NUM_LAYERS": cfg.NUM_LAYERS,
            "DROPOUT": cfg.DROPOUT,
            "INIT_LEARNING_RATE": cfg.INIT_LEARNING_RATE,
            "LR_PATIENCE": cfg.LR_PATIENCE,
            "LR_FACTOR": cfg.LR_FACTOR,
            "LR_MIN": cfg.LR_MIN,
        }
        json.dump(output, jsonfile, indent=4)


def write_output_csv(csv_name, predictions):
    """
    Write model predictions to output CSV
    """
    output_csv = PREDICTIONS_DIR + csv_name
    print("Write the predictions output to: `{}`...".format(output_csv))
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(OUTPUT_FIELDNAMES)
        for i in range(0, len(predictions)):
            writer.writerow(predictions[i])


def plot_loss(completed_epochs, avg_train_losses, avg_test_losses):
    """
    Generate a plot showing the loss-per-epoch
    for both the training and test datasets
    """
    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(completed_epochs, avg_train_losses, color="blue")
    plt.scatter(completed_epochs, avg_test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Entropy (CE) Loss")
    # Force integer X-axis tick marks,
    # since fractional epochs aren't a thing
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(LOSS_PLOT_PATH)
    print("\nLoss plot saved to: `{}`".format(LOSS_PLOT_PATH))


def plot_accuracy(completed_epochs, train_accuracies, test_accuracies):
    """
    Generate a plot showing the accuracy-per-epoch
    for both the training and test datasets
    """
    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(completed_epochs, train_accuracies, color="blue")
    plt.scatter(completed_epochs, test_accuracies, color="red")
    plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy (%)")
    # Force integer X-axis tick marks,
    # since fractional epochs aren't a thing
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(ACCURACY_PLOT_PATH)
    print("Accuracy plot saved to: `{}`\n".format(ACCURACY_PLOT_PATH))


def print_best(completed_epochs, avg_test_losses, test_accuracies):
    """
    Print the epoch with the best loss and accuracy results
    """
    best_epoch = argmin(avg_test_losses)
    best_loss = avg_test_losses[best_epoch]
    best_accuracy = test_accuracies[best_epoch]
    best_epoch_num = completed_epochs[best_epoch]
    print(
        "\t{}Best Accuracy:\t {} {:.2f}% {}".format(
            color.BOLD, color.GREEN, best_accuracy, color.END
        )
    )
    print("\t{}Best Loss:\t {} {:.4f}".format(color.BOLD, color.END, best_loss))
    print("\t{}Best Epoch:\t {} {}".format(color.BOLD, color.END, best_epoch_num))
