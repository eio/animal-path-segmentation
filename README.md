# Usage

## Preprocessing

The `preprocessing/` directory contains scripts to add data labels, derive additional features, normalize, and split into training and test datasets.

## Training

	py run_model.py

Training the model consists of:
- showing it examples
- letting it make guesses
- telling it when it’s wrong

The trained model is also tested on a validation dataset.

The number of epochs and other hyperparameters are defined in `config.py`

The training process logic is in `train_process.py`

The testing process logic is in `test_process.py`


## Testing with a trained model

	py run_model.py -l

Adding the `-l` (or `--load`) flag will load the saved model + optimizer state from `output/saved_model/model+optimizer.pth`

The loaded model will then be tested against the final test dataset as specified at the top of `AnimalDataLoaders.py`


# Data Inputs

The custom PyTorch dataset logic is stored in `AnimalPathsDataset.py`

This is loaded by PyTorch DataLoaders in `AnimalDataLoaders.py`, which is also where the input CSV filepaths are defined.


# File Outputs

All functions that save files can be found in `save_and_load.py`

All generated output files can be found in `output/`

- During training, a saved `model+optimizer.pth` file is updated on each completed epoch and saved inside of `output/saved_model/`. During testing, when the script is run with `-l`, this is the file that is loaded.
- When training is finished, the final loss and accuracy plots for all epochs are stored in `output/performance/`.
- When testing a loaded model, the confusion matrix and classification report are also stored in `output/performance/`.
- Predictions for each training epoch are stored in `output/predictions/epochs/`. During testing, when the script is run with `-l`, the final test predictions are stored as `output/predictions/final_results.csv`

# Utilities

Miscellaneous utility functions (e.g. time and data transformations) are stored in `utils.py`
