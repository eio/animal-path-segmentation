# Usage

	py train_model.py

# Outputs

All functions that save or load files can be found in `save_and_load.py`

During training, a saved `model+optimizer.pth` file will be updated on each completed training epoch and saved inside of `results/`

Predictions for each epoch are stored in `predictions/`

The final train/test loss plot for all epochs is stored in `figures/`