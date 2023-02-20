import os
import sys
import csv
import argparse
import numpy as np

# Machine Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim

# Local scripts
from AnimalDataLoaders import build_data_loaders
from AnimalPathsDataset import ALL_CATEGORIES, OUTPUT_FIELDNAMES
from save_and_load import save_model, load_model, write_output_csv, plot_loss
from utils import (
    color,
    start_script,
    finish_script,
    time_since,
    category_from_output,
    reformat_features,
)


# Check for CUDA / GPU Support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running with device: {}".format(DEVICE))
# Setup tunable constants
N_EPOCHS = 20
BATCH_SIZE = 1
LOG_INTERVAL = 100  # how many train/test records between print statements
# Model parameters
INPUT_SIZE = 1  # one-dimensional with N values
HIDDEN_SIZE = 10  # 128, tunable
OUTPUT_SIZE = 3  # N_categories: winterhome, summerhome, migrating
# Optimizer hyperparameters
LEARNING_RATE = 0.005  # 0.01
# Initialize the Loss function
criterion = nn.CrossEntropyLoss()

# Ensure deterministic behavior:
# cuDNN uses nondeterministic algorithms which are disabled here
torch.backends.cudnn.enabled = False
# For repeatable experiments we have to set random seeds
# for anything using random number generation
random_seed = 1111
torch.manual_seed(random_seed)


class Net(nn.Module):
    """ Basic RNNModel """

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        ### TODO: compare RNN and LSTM
        ## self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(2)
        out, hidden = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def Optimizer(net):
    """
    Create optimizer with specified hyperparameters
    """
    # momentum = 0.5
    # return optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=momentum)
    return optim.Adam(net.parameters(), lr=LEARNING_RATE)


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
    ##########################################
    ## Set start time to keep track of runtime
    ##########################################
    script_start = start_script()
    ###########################
    ## Initialize the RNN model
    ###########################
    # Send model to GPU device (if CUDA-compatible)
    rnn = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    optimizer = Optimizer(rnn)
    #########################
    ## Initialize the output
    #########################
    train_losses = []
    test_losses = []

    ####################
    ####################
    ## Train the Model
    ####################
    ####################
    def train(category_tensor, inputs_tensor):
        ## When using PyTorch's built-in RNN or LSTM modules,
        ## we don't need to define the initHidden() function explicitly.
        ## The hidden state is automatically initialized by the PyTorch module.
        # hidden = rnn.initHidden()
        ##############################
        ##############################
        # Clear the gradients of all optimized tensors
        # to prepare for a new backpropagation pass
        optimizer.zero_grad()
        # Forward pass
        output = rnn(inputs_tensor)
        # Loss function expects category_tensor input to be torch.Long dtype
        category_tensor = category_tensor.to(torch.long)
        # Compute loss
        loss = criterion(output, category_tensor)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Return the prediction and loss
        return output, loss.item()

    def train_batch(epoch):
        print("\nStart Training for Epoch #{}...".format(epoch))
        for i, batch in enumerate(train_loader, 0):
            # Send Tensors to GPU device (if CUDA-compatible)
            inputs_tensor = batch["features"].to(DEVICE)
            category_tensor = batch["label"].to(DEVICE)
            output, loss = train(category_tensor, inputs_tensor)
            train_losses.append(loss)
            # Print iter number, loss, name and guess
            if i % LOG_INTERVAL == 0:
                # Get the predicted category string from the RNN output
                guess, guess_i = category_from_output(output, ALL_CATEGORIES)
                # Convert the label tensor to the category string
                label = int(category_tensor.item())
                category = ALL_CATEGORIES[label]
                # Check if the prediction matches the label
                is_correct = guess == category
                correct = "✓" if is_correct else "✗ (%s)" % category
                print(
                    "Train Epoch:%d %d%% (%s) Loss:%.4f %s / %s %s"
                    % (
                        epoch,
                        i / len(train_loader) * 100,
                        time_since(script_start),
                        loss,
                        "inputs_tensor",  # inputs_tensor.numpy()
                        guess,
                        correct,
                    )
                )
        print("Finished Training for Epoch #{}.".format(epoch))
        #####################
        ## Save the Model  ##
        #####################
        save_model(epoch, rnn, optimizer)

    ####################
    ####################
    ### Test the Model
    ####################
    ####################
    def test(epoch=1):
        print("\nStart Testing for Epoch {}...".format(epoch))
        # Initialize array to store prediction alongside input features
        csv_out_rows = []
        # Let us look at how the network performs on the whole dataset.
        test_loss = 0
        total_correct = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # for data in test_loader:
            for i, batch in enumerate(test_loader, 0):
                # Send Tensors to GPU device (if CUDA-compatible)
                inputs_tensor = batch["features"].to(DEVICE)
                category_tensor = batch["label"].to(DEVICE)
                # calculate outputs by running images through the network
                output = rnn(inputs_tensor)
                # Put data back on the CPU
                output = output.cpu()
                features = inputs_tensor.cpu()
                # Get the predicted category string from the RNN output
                guess, guess_i = category_from_output(output, ALL_CATEGORIES)
                # Convert the label tensor to the category string
                label = int(category_tensor.item())
                category = ALL_CATEGORIES[label]
                # Is the prediction correct?
                is_correct = guess == category
                # Keep track of how many guesses are correct
                if is_correct:
                    total_correct += 1
                # Store prediction alongside input features for CSV out
                features = reformat_features(
                    features, test_loader.dataset.get_individuals()
                )
                csv_out_rows.append([is_correct, guess, category] + features)
                # loss function expects category_tensor input to be torch.Long dtype
                category_tensor = category_tensor.to(torch.long)
                # calculate the loss
                test_loss = criterion(output, category_tensor)
                # store the loss value for this batch
                test_losses.append(test_loss.item())
                # Print iter number, loss, name and guess
                if i % LOG_INTERVAL == 0:
                    # Check if the prediction matches the label
                    correct = "✓" if is_correct else "✗ (%s)" % category
                    print(
                        "Test Epoch:%d %d%% (%s) Loss:%.4f %s / %s %s"
                        % (
                            epoch,
                            i / len(test_loader) * 100,
                            time_since(script_start),
                            test_loss,
                            "inputs_tensor",  # inputs_tensor.numpy()
                            guess,
                            correct,
                        )
                    )
        print("Finished Testing for Epoch {}.".format(epoch))
        print("Test:")
        print("\tAvg. Loss: {}".format(np.mean(test_losses)))
        percent_correct = total_correct / len(test_loader) * 100
        percent_correct = round(percent_correct, 2)
        print("\t{}Accuracy: {}%{}".format(color.BOLD, percent_correct, color.END))
        # Write the predicted poses to an output CSV
        # in the submission format expected
        write_output_csv(epoch, csv_out_rows, OUTPUT_FIELDNAMES)

    ####################################
    ####################################
    ## Perform the Training and Testing
    ####################################
    ####################################
    if LOAD_MODEL == True:
        ###############################################
        # Load the previously saved model and optimizer
        ###############################################
        rnn, optimizer = load_model()
        # TODO: implement new data loader just for final test input
        test_loader = build_final_test_data_loader()
        # Test the loaded model on the final test data
        test()
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
        # Train and test for each epoch
        for epoch in epoch_range:
            train_batch(epoch)
            test(epoch)
            train_loss = np.mean(train_losses)
            test_loss = np.mean(test_losses)
            print("[Epoch {}] Avg. Train Loss: {}".format(epoch, train_loss))
            print("[Epoch {}] Avg. Test Loss: {}".format(epoch, test_loss))
            # keep track of stats for each epoch
            avg_train_losses.append(train_loss)
            avg_test_losses.append(test_loss)
            completed_epochs.append(epoch)
            # reset losses before next epoch
            train_losses = []
            test_losses = []
        ##############################################################
        ## Output model performance evaluation chart across all epochs
        ##############################################################
        plot_loss(completed_epochs, avg_train_losses, avg_test_losses)

    ############
    ## The End
    ############
    finish_script(script_start)
