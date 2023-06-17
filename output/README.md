When `run_model.py` is run, a directory will be created here.

The name of the created directory will be a unique string composed of hyperparameter options in `config.py`

The following sub-directories will be created inside of the new directory:
- `/saved_model/`
- `/performance/`
- `/predictions/`
- `/predictions/epochs/`

Inside of these directories will be the saved model+optimizer state, performance plots and metrics, and raw prediction CSVs.

Testing the saved model by running `run_model.py -l` will generate additional outputs in the subdirectories above, including `/performance/model_evaluation.txt`

Running `show_all_accuracy_scores.py` will walk through each directory per model run, parse `model_evaluation.txt` to find the accuracy score percentage, and print each model run with its associated accuracy score (ordered from best to worst score).