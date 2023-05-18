import argparse
from numpy import mean

# Machine Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Local scripts
from config import Configurator
from utils.misc import start_script, finish_script, get_runtime
from train_process import train_process
from test_process import test_process
from AnimalDataLoaders import (
    build_data_loaders,
    build_final_test_data_loader,
)
from utils.save_and_load import (
    load_model,
    plot_loss,
    plot_accuracy,
    print_best,
)

# Initialize settings and hyperparameters
cfg = Configurator()


class Model(nn.Module):
    """
    Create the model
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        ### TODO: compare RNN and LSTM
        # self.model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True,)
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


def Criterion():
    """
    Create the loss function
    """
    # The log softmax function and the negative log likelihood loss
    # are combined in nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss()


def Optimizer(model):
    """
    Create the optimizer
    """
    return optim.SGD(
        model.parameters(),
        lr=cfg.INIT_LEARNING_RATE,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    # return optim.Adam(
    #     model.parameters(),
    #     lr=cfg.INIT_LEARNING_RATE,
    # )


def Scheduler(optimizer):
    """
    Create the learning-rate scheduler
    """
    # return optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=cfg.SCHEDULER_STEP,
    #     gamma=cfg.SCHEDULER_GAMMA,
    # )
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # reduce LR when the validation loss stops decreasing
        factor=cfg.LR_FACTOR,  # reduce LR by factor of {} when loss stops decreasing
        patience=cfg.LR_PATIENCE,  # wait for {} epochs before reducing the LR
        min_lr=cfg.LR_MIN,  # don't let the learning rate go below {}
        verbose=True,  # print a message when the learning rate is reduced
    )


def main(LOAD_SAVED_MODEL=False):
    """
    Main function
    """
    # Set start time to keep track of runtime
    script_start = start_script()
    ###############################################
    ## Initialize the model and learning conditions
    ###############################################
    # Define the model
    # and send to GPU device (if CUDA-compatible)
    model = Model(
        cfg.INPUT_SIZE,
        cfg.HIDDEN_SIZE,
        cfg.OUTPUT_SIZE,
        cfg.NUM_LAYERS,
        cfg.DROPOUT,
    ).to(cfg.DEVICE)
    # Define the loss function, optimizer, and scheduler
    criterion = Criterion()
    optimizer = Optimizer(model)
    scheduler = Scheduler(optimizer)
    # Initialize losses/accuracies for plot output
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    ###################################
    ###################################
    ## Perform the Training and Testing
    ###################################
    ###################################
    if LOAD_SAVED_MODEL == True:
        #########################################################
        ## Load and test the previously saved model and optimizer
        #########################################################
        model, optimizer, epoch = load_model(model, optimizer)
        # Set the model to evaluation mode
        model.eval()
        # Load the custom AnimalPathsDataset `Testing` data
        test_loader = build_final_test_data_loader(cfg.BATCH_SIZE)
        # Test the loaded model on the test data
        test_process(
            model,
            criterion,
            test_loader,
            script_start,
            cfg.DEVICE,
            cfg.LOG_INTERVAL,
        )
    else:
        ###############################
        ## Train the model from scratch
        ###############################
        # Set the model to training mode
        model.train()
        # Load the custom AnimalPathsDataset `Training` data
        loaders = build_data_loaders(cfg.BATCH_SIZE)
        train_loader = loaders["train"]
        test_loader = loaders["test"]
        # Keep track of completed epoch indices for loss plot
        completed_epochs = []  # list of incremental integers
        # Store running averages of train/test losses for each epoch
        avg_train_losses = []
        avg_test_losses = []
        # Make epochs 1-indexed for better prints
        epoch_range = range(1, cfg.N_EPOCHS + 1)
        ###############################
        # Train and test for each epoch
        ###############################
        for epoch in epoch_range:
            # Run the training process
            train_losses, train_accuracy = train_process(
                optimizer,
                model,
                criterion,
                train_loader,
                script_start,
                cfg.DEVICE,
                cfg.LOG_INTERVAL,
                epoch,
            )
            # Run the testing process
            test_losses, test_accuracy = test_process(
                model,
                criterion,
                test_loader,
                script_start,
                cfg.DEVICE,
                cfg.LOG_INTERVAL,
                epoch,
            )
            # Find the average train/test losses
            train_loss = mean(train_losses)
            test_loss = mean(test_losses)
            # Adjust the learning rate
            # based on the validation loss
            scheduler.step(test_loss)  # Plateau
            ## Update the optimizer's learning rate (StepLR)
            ## scheduler.step()
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
            print(
                "[Epoch {}] Current runtime: {} seconds".format(
                    epoch, get_runtime(script_start)
                )
            )
        ##############################################################
        ## Output model performance evaluation chart across all epochs
        ##############################################################
        plot_loss(completed_epochs, avg_train_losses, avg_test_losses)
        plot_accuracy(completed_epochs, train_accuracies, test_accuracies)
        # Print the best epoch along with its loss and accuracy
        print_best(completed_epochs, avg_test_losses, test_accuracies)
    ##########
    ## The End
    ##########
    # Print total script runtime
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
