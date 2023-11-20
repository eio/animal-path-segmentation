import torch
import json
import subprocess

# Local scripts
from main import train_or_test
from config import Configurator
from utils.misc import start_script, finish_script
from utils.consts import *

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


def train_and_evaluate(params):
    """
    Train and evaluate this model configuration
    """
    print("\n\n")
    print("________________________________________________")
    print("________________________________________________")
    print("          <> GRID SEARCH ITERATION <>           ")
    print()
    # Initialize the config class
    cfg = Configurator(params)
    print(f"Running with config:\n{vars(cfg)}\n")
    # Get the current time to keep track of subprocess runtimes
    script_start = start_script()
    # Train and evaluate the model based on the provided cfg
    train_or_test(
        cfg,
        script_start,
        False,  # Models should be created, not loaded
    )
    # Print total script runtime
    finish_script(script_start)


def main():
    """
    Iterate through all possible hyperparameter combinations,
    training and evaluating new models for each one
    (this will take awhile...)
    """
    # Perform the grid search
    for model_type in HYPERPARAMS[MODEL_TYPE]:
        for optimizer in HYPERPARAMS[OPTIMIZER]:
            for learning_rate in HYPERPARAMS[LEARNING_RATE]:
                for dropout in HYPERPARAMS[DROPOUT]:
                    for hidden_size in HYPERPARAMS[HIDDEN_SIZE]:
                        for num_layers in HYPERPARAMS[NUM_LAYERS]:
                            for batch_size in HYPERPARAMS[BATCH_SIZE]:
                                # Define this iteration's config
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
                                # Train and evaluate this config
                                train_and_evaluate(params)


if __name__ == "__main__":
    print("\nEvaluating all possible model configurations, given:\n")
    print(json.dumps(HYPERPARAMS, indent=2))
    print("\nThis might take awhile.\n")
    # Perform the grid search
    main()
    # Review the contents of the `output/` directory
    # and order all unique model configs
    # by descending final test accuracy
    subprocess.run(["python", "output/show_all_accuracy_scores.py"])
