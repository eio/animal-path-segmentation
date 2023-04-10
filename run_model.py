import argparse
from numpy import mean

# Machine Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Local scripts
from utils import start_script, finish_script
from save_and_load import load_model, plot_loss, plot_accuracy
from train_process import train_process
from test_process import test_process
from AnimalPathsDataset import N_FEATURES, N_CATEGORIES
from AnimalDataLoaders import (
    build_data_loaders,
    build_final_test_data_loader,
)

# Check for CUDA / GPU Support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running with device: {}".format(DEVICE))

# Setup tunable constants
N_EPOCHS = 100
BATCH_SIZE = 1
LOG_INTERVAL = 1
# Model parameters
INPUT_SIZE = N_FEATURES  # number of features / covariates
HIDDEN_SIZE = 64  # tunable hyperparameter
OUTPUT_SIZE = N_CATEGORIES  # "Winter", "Spring", "Summer", "Autumn"
# Optimizer hyperparameters
INIT_LEARNING_RATE = 0.001  # == LR
# LR Scheduler hyperparameters
LR_FACTOR = 0.1  # decrease LR by factor of {}
LR_PATIENCE = 10  # wait {} epochs before decreasing
LR_MIN = 1e-6  # minimum learning rate
# SCHEDULER_STEP = 90  # every {} epochs...
# SCHEDULER_GAMMA = 0.1  # ...multiply LR by {}

# Ensure deterministic behavior:
# cuDNN uses nondeterministic algorithms which are disabled here
torch.backends.cudnn.enabled = False
# For repeatable experiments we have to set random seeds
# for anything using random number generation
random_seed = 1111
torch.manual_seed(random_seed)


class Model(nn.Module):
    """ Basic RNN Model """

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.RNN(input_size, hidden_size, batch_first=True)
        ### TODO: compare RNN and LSTM
        # self.model = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the sequences into a PackedSequence object
        packed_inputs = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        # Pass the packed sequences through the RNN
        packed_outputs, _ = self.model(packed_inputs)
        # Unpack the sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # Pass the outputs through a linear layer to get the final predictions
        logits = self.fc(outputs)
        return logits
        # # Apply softmax activation function to convert logits to probabilities
        # probs = nn.functional.softmax(logits, dim=-1)
        # return probs


def Criterion():
    """
    Create the loss function
    """
    return nn.CrossEntropyLoss()


def Optimizer(model):
    """
    Create optimizer with specified hyperparameters
    """
    # return optim.SGD(
    #     model.parameters(),
    #     lr=INIT_LEARNING_RATE,
    #     momentum=0.5,
    # )
    return optim.Adam(
        model.parameters(),
        lr=INIT_LEARNING_RATE,
    )


def Scheduler(optimizer):
    """
    Create learning-rate-scheduler with specified hyperparameters
    """
    # return optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=SCHEDULER_STEP,
    #     gamma=SCHEDULER_GAMMA,
    # )
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # reduce LR when the validation loss stops decreasing
        factor=LR_FACTOR,  # reduce LR by factor of {} when loss stops decreasing
        patience=LR_PATIENCE,  # wait for {} epochs before reducing the learning rate
        min_lr=LR_MIN,  # we don't want the learning rate to go below 1e-6
        verbose=True,  # print a message when the learning rate is reduced
    )


def main(LOAD_MODEL=False):
    """
    Main function
    """
    ##########################################
    ## Set start time to keep track of runtime
    ##########################################
    script_start = start_script()
    ###############################################
    ## Initialize the model and learning conditions
    ###############################################
    # Define the model (and send to GPU device, if CUDA-compatible)
    model = Model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    # Define the loss function
    criterion = Criterion()
    # Define the optimizer parameters
    optimizer = Optimizer(model)
    # Define the learning rate scheduler parameters
    scheduler = Scheduler(optimizer)
    #########################################
    ## Initialize losses for loss plot output
    #########################################
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    ###################################
    ###################################
    ## Perform the Training and Testing
    ###################################
    ###################################
    if LOAD_MODEL == True:
        ###############################################
        # Load the previously saved model and optimizer
        ###############################################
        model, optimizer, epoch = load_model(model, optimizer)
        ##################################################
        ## Load the custom AnimalPathsDataset testing data
        ##################################################
        test_loader = build_final_test_data_loader()
        #########################################
        ## Test the loaded model on the test data
        #########################################
        test_losses = test_process(
            model, criterion, test_loader, script_start, DEVICE, LOG_INTERVAL
        )
    else:
        ###################################################
        ## Load the custom AnimalPathsDataset training data
        ###################################################
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
            # Run the training process
            train_losses, train_accuracy = train_process(
                optimizer,
                model,
                criterion,
                train_loader,
                script_start,
                DEVICE,
                LOG_INTERVAL,
                epoch,
            )
            # Run the testing process
            test_losses, test_accuracy = test_process(
                model,
                criterion,
                test_loader,
                script_start,
                DEVICE,
                LOG_INTERVAL,
                epoch,
            )
            # Find the average train/test losses
            train_loss = mean(train_losses)
            test_loss = mean(test_losses)
            # Adjust the learning rate
            # based on the validation loss
            scheduler.step(test_loss)  # Plateau
            ### Update the optimizer's learning rate
            ## scheduler.step() # StepLR
            # Keep track of average loss for each epoch
            avg_train_losses.append(train_loss)
            avg_test_losses.append(test_loss)
            # Keep track of accuracy for each epoch
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            # Keep track of each completed epoch index
            completed_epochs.append(epoch)
            print("[Epoch {}] Avg. Train Loss: {}".format(epoch, train_loss))
            print("[Epoch {}] Avg. Test Loss: {}".format(epoch, test_loss))
        ##############################################################
        ## Output model performance evaluation chart across all epochs
        ##############################################################
        plot_loss(completed_epochs, avg_train_losses, avg_test_losses)
        plot_accuracy(completed_epochs, train_accuracies, test_accuracies)
    ##########
    ## The End
    ##########
    finish_script(script_start)


if __name__ == "__main__":
    # Check command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--load",
        required=False,
        action="store_true",
        help="on/off flag specifying if script \
        should run in `load` or `train` mode \
        i.e. `load` a pre-trained model (-l), \
        or `train` a new one from scratch (default)",
    )
    args = parser.parse_args()
    # Start the script
    main(args.load)
