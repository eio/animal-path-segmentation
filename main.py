import argparse
from numpy import mean

# Local scripts
from models.model import *
from config import Configurator
from utils.misc import start_script, finish_script, get_runtime
from train_and_test.train_process import train_process
from train_and_test.test_process import test_process
from data_loaders.data_loaders import (
    build_data_loaders,
    build_final_test_data_loader,
)
from utils.save_and_load import (
    write_accuracy_and_loss_plots,
    print_best,
    load_model,
)


def load_model_and_test(
    model,
    optimizer,
    criterion,
    cfg,
    script_start,
):
    """
    Load the trained model+optimizer
    and test on the final, unseen test dataset
    """
    model, optimizer, epoch = load_model(
        model,
        optimizer,
        cfg.OUTPUT_DIR,
    )
    # Set the model to evaluation mode
    model.eval()
    # Load the custom dataset for `test.csv` data
    test_loader = build_final_test_data_loader(cfg.BATCH_SIZE)
    # Test the loaded model on the test data
    test_process(
        cfg.OUTPUT_DIR,
        model,
        criterion,
        test_loader,
        script_start,
        cfg.DEVICE,
        cfg.LOG_INTERVAL,
    )


def train_model_from_scratch(
    model,
    optimizer,
    criterion,
    scheduler,
    cfg,
    script_start,
):
    """
    Train the model from scratch
    and test on the validation set
    """
    # Set the model to training mode
    model.train()
    # Initialize losses/accuracies for plot output
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    # Load the custom datasets
    # for `train.csv` and `validation.csv` data
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
            cfg.OUTPUT_DIR,
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
            cfg.OUTPUT_DIR,
            model,
            criterion,
            test_loader,
            script_start,
            cfg.DEVICE,
            cfg.LOG_INTERVAL,
            cfg.SAVE_PREDICTIONS_EVERY,
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
        if epoch % cfg.PLOT_EVERY == 0:
            # Output accuracy and loss plots
            # at the specified interval of epochs
            write_accuracy_and_loss_plots(
                completed_epochs,
                avg_train_losses,
                avg_test_losses,
                train_accuracies,
                test_accuracies,
                cfg.OUTPUT_DIR,
            )
    ##############################################################
    ## Output model performance evaluation chart across all epochs
    ##############################################################
    write_accuracy_and_loss_plots(
        completed_epochs,
        avg_train_losses,
        avg_test_losses,
        train_accuracies,
        test_accuracies,
        cfg.OUTPUT_DIR,
    )
    # Print the Best Epoch along with its loss and accuracy
    print_best(completed_epochs, avg_test_losses, test_accuracies)
    # Now reload the trained model and test with the final test set:
    train_or_test(
        cfg,
        script_start,
        True,  # Load the trained model
    )


def train_or_test(cfg, script_start, LOAD_MODEL_FLAG):
    """
    Perform the Training and Testing
    or just the Loading and Testing
    depending on the flag
    """
    # Define the model and send to GPU device (if CUDA-compatible)
    model = Model(
        cfg.MODEL,
        cfg.INPUT_SIZE,
        cfg.HIDDEN_SIZE,
        cfg.NUM_LAYERS,
        cfg.DROPOUT,
    ).to(cfg.DEVICE)
    # Define the loss function, optimizer, and scheduler
    criterion = Criterion()
    optimizer = Optimizer(model, cfg)
    scheduler = Scheduler(optimizer, cfg)
    # Perform the Training or Testing
    if LOAD_MODEL_FLAG == True:
        load_model_and_test(
            model,
            optimizer,
            criterion,
            cfg,
            script_start,
        )
    else:
        train_model_from_scratch(
            model,
            optimizer,
            criterion,
            scheduler,
            cfg,
            script_start,
        )


def main(LOAD_MODEL_FLAG=False):
    """
    Main function
    """
    # Set start time to keep track of runtime
    script_start = start_script()
    # Initialize settings, hyperparameters, and output directory
    cfg = Configurator()
    # Perform the Training and Testing
    train_or_test(
        cfg,
        script_start,
        LOAD_MODEL_FLAG,
    )
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
