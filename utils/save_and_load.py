import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import save as torch_save, load as torch_load
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
# Local scripts
from utils.misc import color
from utils.consts import (
    IDENTIFIER,
    FEATURE_COLUMNS,
    SEASON_LABELS,
)

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
EVALUATION_PATH = OUTPUT + "performance/model_evaluation.txt"
CONFUSION_MATRIX_PATH = OUTPUT + "performance/confusion_matrix.png"
CONFUSION_MATRIX_PERCENT_PATH = OUTPUT + "performance/confusion_matrix_percent.png"
ACCURACY_PLOT_PATH = OUTPUT + "performance/accuracy.png"
LOSS_PLOT_PATH = OUTPUT + "performance/loss.png"
CONFIG_PATH = OUTPUT + "config.json"
PREDICTIONS_DIR = OUTPUT + "predictions/"
LINE = "\n------------------------------------------------------\n\n"


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
            "BATCH_SIZE": cfg.BATCH_SIZE,
            "HIDDEN_SIZE": cfg.HIDDEN_SIZE,
            "NUM_LAYERS": cfg.NUM_LAYERS,
            "DROPOUT": cfg.DROPOUT,
            "INIT_LEARNING_RATE": cfg.INIT_LEARNING_RATE,
            "SGD_MOMENTUM": cfg.MOMENTUM,
            "SGD_WEIGHT_DECAY": cfg.WEIGHT_DECAY,
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
    print("Predictions saved to: `{}`...".format(output_csv))
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(OUTPUT_FIELDNAMES)
        for i in range(0, len(predictions)):
            writer.writerow(predictions[i])


def write_accuracy_and_loss_plots(
    completed_epochs,
    avg_train_losses,
    avg_test_losses,
    train_accuracies,
    test_accuracies,
):
    """
    Output model accuracy and loss plots
    """
    plot_loss(completed_epochs, avg_train_losses, avg_test_losses)
    plot_accuracy(completed_epochs, train_accuracies, test_accuracies)


def write_performance_eval(labels, guesses):
    """
    Evaluate model performance using:
    - the accuracy score
    - the confusion matrix
    - the classification report (precision, recall, f1-score, support)
    """
    confmat = confusion_matrix(labels, guesses)
    # Generate a text file with the confusion matrix and other metrics
    with open(EVALUATION_PATH, "w") as f:
        f.write("[ Model Evaluation ]\n{}".format(LINE))
        f.write("Accuracy Score:\t\t{}\n".format(accuracy_score(labels, guesses)))
        f.write("{}Confusion Matrix:\n\n{}\n".format(LINE, confmat))
        f.write(
            "{}Classification Report:\n\n{}".format(
                LINE,
                # `zero_division=1` sets precision and F-score to 1
                # for classes with no predicted samples
                classification_report(labels, guesses, zero_division=1),
            )
        )
        # Note: F1 scores range from 0 to 1, with higher scores being generally better
        print("Model evaluation report saved to `{}`".format(EVALUATION_PATH))
    # Plot the standard confusion matrix (counts)
    plot_confusion_matrix(labels, guesses, CONFUSION_MATRIX_PATH)
    # # Calculate the percentage values by dividing each element in the confusion matrix
    # # by the sum of the corresponding row (i.e., the total number of instances of that class)
    # # and multiplying by 100. The np.round function is used to round the results to two decimal places.
    # confmat_percent = np.round(confmat / confmat.sum(axis=1)[:, np.newaxis] * 100, 2)
    # # Plot the relative percentage confusion matrix (%)
    # plot_confusion_matrix(confmat_percent, CONFUSION_MATRIX_PERCENT_PATH)


def plot_confusion_matrix(labels, guesses, outpath):
    """
    Generate a colorized confusion matrix image
    """
    # Generate the confusion matrix plot display
    cmd = ConfusionMatrixDisplay.from_predictions(labels, guesses)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(9, 9))
    cmd.plot(ax=ax)
    # Customize the axis labels
    ax.set_xlabel("Model Output", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Label", fontsize=12, fontweight="bold", labelpad=10)
    # Save the confusion matrix image
    plt.savefig(outpath)
    # Close the figure
    plt.close(fig)


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
    # Close the figure
    plt.close(fig)


def plot_accuracy(completed_epochs, train_accuracies, test_accuracies):
    """
    Generate a plot showing the accuracy-per-epoch
    for both the training and test datasets
    """
    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(completed_epochs, train_accuracies, color="blue")
    plt.scatter(completed_epochs, test_accuracies, color="red")
    plt.legend(["Train Accuracy", "Test Accuracy"], loc="lower right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy (%)")
    # Force integer X-axis tick marks,
    # since fractional epochs aren't a thing
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(ACCURACY_PLOT_PATH)
    print("Accuracy plot saved to: `{}`\n".format(ACCURACY_PLOT_PATH))
    # Close the figure
    plt.close(fig)


def print_best(completed_epochs, avg_test_losses, test_accuracies):
    """
    Print the epoch with the best loss and accuracy results
    """
    best_epoch = np.argmin(avg_test_losses)
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
