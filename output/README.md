When `run_model.py` is run, a directory will be created here.
The name of the created directory will be a unique string composed of hyperparameter options in `config.py`

The following sub-directories will be created inside of the new directory:
- `/saved_model/`
- `/performance/`
- `/predictions/`
- `/predictions/epochs/`

Inside of these directories will be the saved model+optimizer state, performance plots and metrics, and raw prediction CSVs.