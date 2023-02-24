# Usage

## Training

	py run_model.py

Training the model consists of:
- showing it examples,
- letting it make guesses,
- and telling it when it’s wrong.

The trained model is also tested on a validation dataset.

The number of epochs and other hyperparameters are defined at the top of `run_model.py`

The training process logic is in `train_process.py`


## Testing

	py run_model.py -l

Adding the `-l` (or `--load`) flag will load the saved model + optimizer state.

The loaded model will then be tested against the final test dataset.

The testing process logic is in `test_process.py`


# Data Inputs

The custom PyTorch dataset logic is stored in `AnimalPathsDataset.py`

This is loaded by PyTorch DataLoaders in `AnimalDataLoaders.py`


# File Outputs

All functions that save files can be found in `save_and_load.py`

- During training, a saved `model+optimizer.pth` file is updated on each completed epoch and saved inside of `results/`. During testing, when the script is run with `-l`, this is the file that is loaded.
- The final loss and accuracy plots for all epochs are stored in `figures/`.
- Predictions for each training epoch are stored in `predictions/epochs/`. During testing, when the script is run with `-l`, the final test predictions are stored as `predictions/predictions_final_test.csv`

# Utilities

Miscellaneous utility functions (e.g. time and data transformations) are stored in `utils.py`
