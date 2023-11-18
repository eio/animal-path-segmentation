import torch

# Local scripts
from main import train_or_test
from config import Configurator
from utils.misc import start_script
from utils.consts import *

# Define consts
MODEL_TYPE = "MODEL_TYPE"
OPTIMIZER = "OPTIMIZER"
LEARNING_RATE = "LEARNING_RATE"
DROPOUT = "DROPOUT"
HIDDEN_SIZE = "HIDDEN_SIZE"
NUM_LAYERS = "NUM_LAYERS"
# Use a fixed number of epochs
NUM_EPOCHS_FIXED = 1
# Define hyperparameter options
HYPERPARAMS = {
    MODEL_TYPE: [RNN, LSTM, GRU],
    OPTIMIZER: [ADAM, SGD],
    LEARNING_RATE: [0.01, 0.001, 0.0001],
    DROPOUT: [0.2, 0.3, 0.4],
    HIDDEN_SIZE: [32, 64, 128, 256],
    NUM_LAYERS: [2, 3, 4],
    BATCH_SIZE: [1],
}
# Models should be created, not loaded
LOAD_MODEL_AND_TEST = False


def train_and_evaluate(cfg):
    """
    Train and evaluate this model configuration
    """
    print("\n\n\n")
    print("________________________________________________")
    print("________________________________________________")
    print("          <> GRID SEARCH ITERATION <>           ")
    print("________________________________________________")
    print(f"{vars(cfg)}")
    print("________________________________________________")
    print("________________________________________________")
    print("\n")
    # Get the current time to keep track of subprocess runtimes
    script_start = start_script()
    # Train and evaluate the model based on the provided cfg
    train_or_test(
        LOAD_MODEL_AND_TEST,
        script_start,
        cfg,
    )


def find_best_hyperparams():
    """
    Iterate through all possible hyperparameter combinations
    training and evaluating new models for each one
    """
    best_accuracy = 0.0
    best_hyperparams = {}
    # Perform the grid search
    # (this will take awhile...)
    for model_type in HYPERPARAMS[MODEL_TYPE]:
        for optimizer in HYPERPARAMS[OPTIMIZER]:
            for learning_rate in HYPERPARAMS[LEARNING_RATE]:
                for dropout in HYPERPARAMS[DROPOUT]:
                    for hidden_size in HYPERPARAMS[HIDDEN_SIZE]:
                        for num_layers in HYPERPARAMS[NUM_LAYERS]:
                            for batch_size in HYPERPARAMS[BATCH_SIZE]:
                                # Build the hyperparameter input structure
                                params = {
                                    MODEL_TYPE: model_type,
                                    OPTIMIZER: optimizer,
                                    LEARNING_RATE: learning_rate,
                                    DROPOUT: dropout,
                                    HIDDEN_SIZE: hidden_size,
                                    NUM_LAYERS: num_layers,
                                    BATCH_SIZE: batch_size,
                                    NUM_EPOCHS: NUM_EPOCHS_FIXED,
                                }
                                # Create the config object
                                cfg = Configurator(params)
                                # Train and evaluate the mod config
                                train_and_evaluate(cfg)
                                #############################
                                #############################
                                # TODO: return and compare real accuracies
                                # ...broken kluge for now...
                                accuracy = 0
                                #############################
                                #############################
                                # Check if the current model's accuracy is better than the best
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_hyperparams = {
                                        MODEL_TYPE: model_type,
                                        OPTIMIZER: optimizer_name,
                                        LEARNING_RATE: learning_rate,
                                        DROPOUT: dropout,
                                        HIDDEN_SIZE: hidden_size,
                                        NUM_LAYERS: num_layers,
                                    }
    # Return the best hyperparameter configs
    # and their respective accuracies
    return best_hyperparams, best_accuracy


if __name__ == "__main__":
    # Perform grid search to find the best hyperparameter configs
    best_hyperparams, best_accuracy = find_best_hyperparams()
    # Print the best hyperparameters and accuracy
    print("Best Hyperparameters:")
    print(best_hyperparams)
    print("Best Accuracy:")
    print(best_accuracy)
