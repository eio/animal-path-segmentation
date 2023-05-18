# Usage

## Preprocessing

The `preprocessing/` directory contains scripts to add training labels, derive additional features, normalize, and split the data into training and test datasets.

The expected order of these operations is specified with the numerical prefix in subdirectory names:

- `1_add_labels/`
- `2_add_derived_features/`
- `3_normalize_data/`
- `4_split_data/`

## Training

	py run_model.py

Training the model consists of:
- showing it examples
- letting it make guesses
- telling it when it’s wrong

The number of epochs, other hyperparameters, and output intervals are defined in `config.py`

Each epoch, the model trains on `data/train.csv` and tests on `data/validation.csv`

The training process logic is in `train_and_test/train_process.py`

The testing process logic is in `train_and_test/test_process.py`


## Testing with a trained model

	py run_model.py -l

Adding the `-l` (or `--load`) flag will load the saved model + optimizer state from `output/saved_model/model+optimizer.pth`

The loaded model will then be tested against the final test dataset (`data/test.csv`) as specified at the top of `data_loaders/data_loaders.py`


# Data Inputs

The PyTorch Custom Dataset logic is in `data_loaders/AnimalPathsDataset.py`

This is loaded by PyTorch DataLoaders in `data_loaders/data_loaders.py`, which is also where the input CSV filepaths are defined.

The actual data files (`train.csv`, `validation.csv`, and `test.csv`) should be stored in `data/`.


# File Outputs

All functions that save files can be found in `utils/save_and_load.py`

All generated output files can be found in `output/`

- During training, a saved `model+optimizer.pth` file is updated on each completed epoch and saved inside of `output/saved_model/`. During testing, when the script is run with `-l`, this is the file that is loaded.
- When training is finished, the final loss and accuracy plots for all epochs are stored in `output/performance/`.
- When testing a loaded model, the confusion matrix and classification report are also stored in `output/performance/`.
- Predictions for each training epoch are stored in `output/predictions/epochs/`. During testing, when the script is run with `-l`, the final test predictions are stored as `output/predictions/final_results.csv`

# Utilities

Miscellaneous utility functions (e.g. time and data transformations) are stored in `utils/misc.py`

Widely-used constants, normalization code, and normalization config JSON are also stored in `utils/`
