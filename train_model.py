import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning Framework
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Local scripts
from utils import currentTime, timeSince
from animal_data_loaders import build_data_loaders
from AnimalPathsDataset import ALL_CATEGORIES, N_FEATURES

# Check for CUDA / GPU Support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup tunable constants
N_EPOCHS = 1
BATCH_SIZE = 1
PLOT_EVERY = 50
N_HIDDEN = 128  # tunable
N_CATEGORIES = 3  # winterhome, summerhome, migrating
# If you set this too high, it might explode.
# If too low, it might not learn
LEARNING_RATE = 0.005
# Setup path for saving model
SAVED_MODEL_PATH = "results/model.pth"
# Setup output paths
FIGURES_OUTPUT_DIR = "figures/"
PREDICTIONS_OUTPUT_DIR = "predictions/"
# To be set before training begins
START = None

# class RNN(nn.Module):
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def save_model(net):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # Save the current state of the Model
    # so we can load the latest state later on
    torch.save(
        {"model_state_dict": net.state_dict()},
        SAVED_MODEL_PATH,
    )


def load_model():
    """
    Load and return the saved, pre-trained Model
    """
    print("Loading the saved model: `{}`".format(SAVED_MODEL_PATH))
    saved_state = torch.load(SAVED_MODEL_PATH)
    net = Net().to(DEVICE)
    net.load_state_dict(saved_state["model_state_dict"])
    print("Model loaded.")
    return net


def evaluate_performance(completed_epochs, avg_train_losses, avg_test_losses=None):
    # print("train_counter", train_counter)
    # print("train_losses", train_losses)
    # print("test_counter", test_counter)
    # print("test_losses", test_losses)
    FIGURE_OUTPUT = FIGURES_OUTPUT_DIR + "loss.png"

    fig = plt.figure()
    plt.scatter(completed_epochs, avg_train_losses, color="blue")
    plt.scatter(completed_epochs, avg_test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Negative Log Likelihood (NLL) Loss")
    plt.savefig(FIGURE_OUTPUT)
    print("Performance evaluation saved to: `{}`".format(FIGURE_OUTPUT))


# def write_output_csv(predictions, sample_csv, epoch, test_dataset_name):
#     """
#     Write model predictions to output submission CSV
#     """
#     metadata = pd.read_csv(sample_csv)
#     csv_name = "{}_predictions_epoch{}.csv".format(test_dataset_name, epoch)
#     output_csv = PREDICTIONS_OUTPUT_DIR + csv_name
#     print("Write the predicted output to: {}...".format(output_csv))
#     # print("\t predictions length: {}".format(len(predictions)))
#     # print("\t metadata length: {}".format(len(metadata)))
#     with open(output_csv, "w", newline="") as csvfile:
#         fieldnames = ["filename", "sequence", "Tx", "Ty", "Tz", "Qx", "Qy", "Qz", "Qw"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for i in range(0, len(predictions)):
#             row = {
#                 "filename": metadata.iloc[i, SPD.FILENAME_COLUMN],
#                 "sequence": metadata.iloc[i, SPD.SEQUENCE_COLUMN],
#                 "Tx": predictions[i][0],
#                 "Ty": predictions[i][1],
#                 "Tz": predictions[i][2],
#                 "Qx": predictions[i][3],
#                 "Qy": predictions[i][4],
#                 "Qz": predictions[i][5],
#                 "Qw": predictions[i][6],
#             }
#             writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # on/off flag for whether script should run in "load" or "train" mode
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-ts", "--synthetic", action="store_true")
    parser.add_argument("-tr", "--real", action="store_true")
    args = parser.parse_args()
    LOAD_MODEL = args.load
    TEST_SYNTHETIC = args.synthetic
    TEST_REAL = args.real
    #############################################
    ## Print start time to keep track of runtime
    #############################################
    print("Start: {}".format(currentTime()))
    ###########################
    ## Initialize the CNN model
    ###########################
    print("Running with device: {}".format(DEVICE))
    # Send model to GPU device (if CUDA-compatible)
    rnn = Net(N_FEATURES, N_HIDDEN, N_CATEGORIES).to(DEVICE)
    # specify the loss function
    # For the loss function nn.NLLLoss is appropriate, since the last layer of the RNN is nn.LogSoftmax.
    # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    criterion = nn.NLLLoss()
    # cuDNN uses nondeterministic algorithms which are disabled here
    torch.backends.cudnn.enabled = False
    # For repeatable experiments we have to set random seeds
    # for anything using random number generation
    random_seed = 1
    torch.manual_seed(random_seed)
    # configure batch and downscale sizes
    batch_size_train = BATCH_SIZE
    batch_size_test = BATCH_SIZE
    #########################
    ## Initialize the output
    #########################
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    ######################
    ######################
    ## Train the Network
    ######################
    ######################
    def train(category_tensor, inputs_tensor):
        hidden = rnn.initHidden()
        print("hidden:", hidden)
        print("inputs_tensor:", inputs_tensor)
        rnn.zero_grad()
        for i in range(inputs_tensor.size()[0]):
            output, hidden = rnn(inputs_tensor[i], hidden)
        loss = criterion(output, category_tensor)
        loss.backward()
        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)
        return output, loss.item()

    def train_batch(epoch):
        print("\nStart Training for Epoch #{}...".format(epoch))
        current_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            # Convert inputs/labels (aka data/targets)
            # to float values, and send Tensors to GPU device (if CUDA-compatible)
            # inputs = batch["features"].float().to(DEVICE)
            # labels = batch["label"].float().to(DEVICE)
            inputs_tensor = batch["features"].to(DEVICE)
            category_tensor = batch["label"].to(DEVICE)
            output, loss = train(category_tensor, inputs_tensor)
            current_loss += loss
            # Print iter number, loss, name and guess
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output, ALL_CATEGORIES)
                correct = "✓" if guess == category else "✗ (%s)" % category
                print(
                    "%d %d%% (%s) %.4f %s / %s %s"
                    % (
                        iter,
                        iter / n_iters * 100,
                        timeSince(START),
                        loss,
                        line,
                        guess,
                        correct,
                    )
                )
            # Add current loss avg to list of losses
            if iter % PLOT_EVERY == 0:
                train_losses.append(current_loss / PLOT_EVERY)
                current_loss = 0
                #####################
                ## Save the Model  ##
                #####################
                save_model(net)
        print("Finished Training for Epoch #{}.".format(epoch))

    ################################
    ################################
    ### Test the Whole Test Dataset
    ################################
    ################################
    # def test(epoch=1, sample_csv=VALIDATION_CSV):
    #     print("\nStart Testing for Epoch {}...".format(epoch))
    #     # Initialize array to store all predictions
    #     predictions = []
    #     # Let us look at how the network performs on the whole dataset.
    #     test_loss = 0
    #     correct = 0
    #     # since we're not training, we don't need to calculate the gradients for our outputs
    #     with torch.no_grad():
    #         # for data in test_loader:
    #         for i, batch in enumerate(test_loader, 0):
    #             # Convert inputs/labels (aka data/targets)
    #             # to float values, and send Tensors to GPU device (if CUDA-compatible)
    #             inputs = batch["image"].float().to(DEVICE)
    #             labels = batch["pose"].float().to(DEVICE)
    #             # labels = batch["pose"].float().reshape(1, 7)
    #             # calculate outputs by running images through the network
    #             outputs = net(inputs)
    #             # store the predicted outputs
    #             prediction = outputs.cpu().numpy().flatten()
    #             predictions.append(prediction)
    #             # calculate the loss
    #             test_loss = criterion(outputs, labels)
    #             # store the loss value for this batch
    #             test_losses.append(test_loss.item())
    #             ## Consider prediction to be correct
    #             ## if `test_loss` is "close enough" to a perfect score of 0.0
    #             close_enough = 0.001
    #             if test_loss <= close_enough:
    #                 correct += 1
    #             # print some statistics based on the log interval
    #             if i % LOG_INTERVAL == 0:
    #                 print(
    #                     "Test: [{}/{} ({}%)]\tLoss: {}".format(
    #                         i,  # i * len(batch),
    #                         len(test_loader.dataset),
    #                         100.0 * i / len(test_loader),
    #                         test_loss.item(),
    #                     )
    #                 )
    #                 test_counter.append(
    #                     (i * batch_size_test) + ((epoch - 1) * len(test_loader.dataset))
    #                 )
    #     print(
    #         "Test set: Avg. loss: {}, Accuracy: {}/{} ({}%)".format(
    #             np.mean(test_losses),
    #             correct,
    #             len(test_loader.dataset),
    #             100.0 * (correct / len(test_loader.dataset)),
    #         )
    #     )
    #     print("Finished Testing for Epoch {}.".format(epoch))
    #     # Write the predicted poses to an output CSV
    #     # in the submission format expected
    #     test_dataset_name = test_loader.dataset.root_dir.split("/")[1]
    #     # write_output_csv(predictions, sample_csv, epoch, test_dataset_name)

    ####################################
    ####################################
    ## Perform the Training and Testing
    ####################################
    ####################################
    # All it takes to train this network is:
    # - show it a bunch of examples,
    # - have it make guesses,
    # - and tell it if it’s wrong.
    if LOAD_MODEL == True:
        #################################
        # Load the previously saved model
        #################################
        net = load_model()
        # Update test dataset, overwriting `test_loader` variable
        print("Testing for images in: {}".format(TEST_SYNTHETIC_ROOT))
        test_loader = build_final_test_data_loader(
            batch_size_test,
            img_downscale_size,
            TEST_SYNTHETIC_CSV,
            TEST_SYNTHETIC_ROOT,
        )
        # Test the loaded model on the synthetic data
        test(1, TEST_SYNTHETIC_CSV)
    else:
        #################################################################
        ## Load the custom SatellitePoseDataset into PyTorch DataLoaders
        #################################################################
        loaders = build_data_loaders()
        train_loader = loaders["train"]
        test_loader = loaders["test"]
        #####################################
        ## Train the model from the beginning
        #####################################
        completed_epochs = []
        # Store running averages of train/test losses for each epoch
        avg_train_losses = []
        avg_test_losses = []
        # Make epochs 1-indexed for better prints
        epoch_range = range(1, N_EPOCHS + 1)
        # Set the training start time
        START = currentTime()
        # Train and test for each epoch
        for epoch in epoch_range:
            train_batch(epoch)
            # test(epoch, VALIDATION_CSV)
            train_loss = np.mean(train_losses)
            # test_loss = np.mean(test_losses)
            print("[Epoch {}] Avg. Train Loss: {}".format(epoch, train_loss))
            print("[Epoch {}] Avg. Test Loss: {}".format(epoch, test_loss))
            # keep track of stats for each epoch
            avg_train_losses.append(train_loss)
            # avg_test_losses.append(test_loss)
            completed_epochs.append(epoch)
            # reset losses before next epoch
            train_losses = []
            # test_losses = []
        ##############################################################
        ## Output model performance evaluation chart across all epochs
        ##############################################################
        evaluate_performance(completed_epochs, avg_train_losses)  # , avg_test_losses)

    ############
    ## The End
    ############
    print("\nEnd: {}".format(currentTime()))
