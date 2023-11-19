# Usage

## Preprocessing

The `preprocessing/` directory contains scripts to add training labels, derive additional features, downsample to daily positions, normalize, and split the data into training, validation, and test datasets.

The expected order of these operations is specified with the numerical prefix in subdirectory names:

- `1_add_labels/`
- `2_add_derived_features/`
- `3_normalize_data/`
- `4_split_data/`

They can be run in sequence with:

	cd preprocessing/
	python run_all_preproc.py

Before running this, the events data `crane_events_20220223.csv` and the labels data `segmentations/*.csv` should be added to the `data/raw_inputs/` directory.

The output of each step in the process is the input for the next step, finishing with `4_split_data/` which produces the `train.csv`, `validation.csv`, and `test.csv` files in `4_split_data/output`.

Once produced, the final split output files should be moved to the top-level `data/` directory.

NOTE: `__add_environmental_features/` is not integrated into the workflow for now, but would go after `2_add_derived_features/`

## Training

	python main.py

Training the model consists of:
- showing it examples
- letting it make guesses
- telling it when it’s wrong

The number of epochs, other hyperparameters, and output intervals are defined in `config.py`

Each epoch, the model trains on `data/train.csv` and tests on `data/validation.csv`

The training process logic is in `train_and_test/train_process.py`

The testing process logic is in `train_and_test/test_process.py`

Training can also be initiated with `grid_search.py`. This script will step through multiple model configurations, training various models and then comparing their final accuracies. Depending on the number of hyperparameter options specified, this could take a very long time to complete.

## Testing with a trained model

	python main.py -l

Adding the `-l` (or `--load`) flag will load the saved model + optimizer state from `output/saved_model/model+optimizer.pth`

The loaded model will then be tested against the final test dataset (`data/test.csv`) as specified at the top of `data_loaders/data_loaders.py`


# Data Inputs

The PyTorch Custom Dataset logic is in `data_loaders/AnimalPathsDataset.py`

This is loaded by PyTorch DataLoaders in `data_loaders/data_loaders.py`, which is also where the input CSV filepaths are defined.

The actual data files (`train.csv`, `validation.csv`, and `test.csv`) should be stored in `data/`.


# File Outputs

All functions that save files can be found in `utils/save_and_load.py`

All generated output files can be found in `output/{UNIQUE_PARAMS_STRING}/`

- During training, a saved `model+optimizer.pth` file is updated on each completed epoch and saved inside of `output/{UNIQUE_PARAMS_STRING}/saved_model/`. During testing, when the script is run with `-l`, this is the file that is loaded.
- When training is finished, the final loss and accuracy plots for all epochs are stored in `output/{UNIQUE_PARAMS_STRING}/performance/`.
- When testing a loaded model, the confusion matrix and classification report are also stored in `output/{UNIQUE_PARAMS_STRING}/performance/`.
- Predictions for each training epoch are stored in `output/{UNIQUE_PARAMS_STRING}/predictions/epochs/`. During testing, when the script is run with `-l`, the final test predictions are stored as `output/{UNIQUE_PARAMS_STRING}/predictions/final_results.csv`

# Utilities

Miscellaneous utility functions (e.g. time and data transformations) are stored in `utils/misc.py`

Widely-used constants, normalization code, and normalization config JSON are also stored in `utils/`

Additional tools for investigating, visualizing, and reporting on results are stored in `tools/`
