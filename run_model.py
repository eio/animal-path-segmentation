import argparse
from numpy import mean

# Machine Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim

# Local scripts
from AnimalDataLoaders import build_data_loaders
from save_and_load import load_model, plot_loss
from train_process import train_process
from test_process import test_process
from utils import start_script, finish_script


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


class Model(nn.Module):
    """ Basic RNN Model """

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.RNN(input_size, hidden_size, batch_first=True)
        ### TODO: compare RNN and LSTM
        # self.model = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(2)
        out, hidden = self.model(x)
        out = self.fc(out[:, -1, :])
        return out


def Optimizer(model):
    """
    Create optimizer with specified hyperparameters
    """
    # momentum = 0.5
    # return optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum)
    return optim.Adam(model.parameters(), lr=LEARNING_RATE)


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
    #######################
    ## Initialize the model
    #######################
    # Send model to GPU device (if CUDA-compatible)
    model = Model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    optimizer = Optimizer(model)
    #########################################
    ## Initialize losses for loss plot output
    #########################################
    train_losses = []
    test_losses = []
    ####################################
    ####################################
    ## Perform the Training and Testing
    ####################################
    ####################################
    if LOAD_MODEL == True:
        ###############################################
        # Load the previously saved model and optimizer
        ###############################################
        model, optimizer = load_model()
        # TODO: implement new data loader just for final test input
        test_loader = build_final_test_data_loader()
        # Test the loaded model on the final test data
        test_process()
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
            # Run the training process
            train_losses = train_process(
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
            test_losses = test_process(
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
            # Keep track of stats for each epoch
            avg_train_losses.append(train_loss)
            avg_test_losses.append(test_loss)
            completed_epochs.append(epoch)
            print("[Epoch {}] Avg. Train Loss: {}".format(epoch, train_loss))
            print("[Epoch {}] Avg. Test Loss: {}".format(epoch, test_loss))
        ##############################################################
        ## Output model performance evaluation chart across all epochs
        ##############################################################
        plot_loss(completed_epochs, avg_train_losses, avg_test_losses)

    ############
    ## The End
    ############
    finish_script(script_start)
