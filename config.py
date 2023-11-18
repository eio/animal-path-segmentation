import torch

# Local scripts
from utils.consts import *
from utils.save_and_load import write_config_json, make_directory

# Hyperparameters to be applied IF running `main.py`
# (i.e. this is ignored if running `grid_search.py`)
DEFAULT_HYPERPARAMS = {
    MODEL_TYPE: GRU,  # RNN, LSTM, GRU
    OPTIMIZER: ADAM,  # ADAM, SGD
    HIDDEN_SIZE: 16,  # 32, 64, 128, 256, ...
    NUM_LAYERS: 2,  # 3, 4, ...
    LEARNING_RATE: 0.01,  # 0.001, 0.0001
    DROPOUT: 0.1,  # 0.1 to 0.5
    NUM_EPOCHS: 1,
    BATCH_SIZE: 1,
}


class Configurator(object):
    """
    Initialize PyTorch settings and hyperparameters
    """

    def __init__(self, cfg=DEFAULT_HYPERPARAMS):
        ####################################
        ## Details defined in `run_model.py`
        ####################################
        self.MODEL = cfg[MODEL_TYPE]
        self.OPTIMIZER = cfg[OPTIMIZER]
        #################
        ## Print settings
        #################
        # Print every {} epochs
        self.LOG_INTERVAL = 1  # epochs
        ####################
        ## Training settings
        ####################
        self.N_EPOCHS = cfg[NUM_EPOCHS]
        self.BATCH_SIZE = cfg[BATCH_SIZE]
        #############################
        ## Evaluation output settings
        #############################
        # Output accuracy/loss plots every {} epochs
        self.PLOT_EVERY = 100  # epochs
        # Save an output CSV of predictions every {} epochs
        self.SAVE_PREDICTIONS_EVERY = 100  # epochs
        #################
        ## Model settings
        #################
        # `input_size` is the number of features/ covariates
        self.INPUT_SIZE = N_FEATURES
        # `hidden_size` controls the width of each RNN layer
        # (i.e., the number of neurons in each layer)
        self.HIDDEN_SIZE = cfg[HIDDEN_SIZE]
        # `num_layers` controls the depth of the RNN
        # (i.e., the number of stacked RNN layers)
        self.NUM_LAYERS = cfg[NUM_LAYERS]  # default = 1
        # The recommended values for dropout probability are
        # between 0.1 and 0.5, depending on the task and model size
        self.DROPOUT = cfg[DROPOUT]  # default = 0
        # Guideline: if the model has less than tens-of-thousands of trainable parameters,
        # regularization may not be needed. For an RNN:
        # trainable_params = ((input_size + hidden_size) * hidden_size + hidden_size) * num_layers
        #####################
        ## Optimizer settings
        #####################
        self.INIT_LEARNING_RATE = cfg[LEARNING_RATE]
        self.MOMENTUM = 0.5  # SGD only
        self.WEIGHT_DECAY = 0.001  # SGD or ADAM
        #####################
        ## Scheduler settings
        #####################
        # Hyperparameters for `lr_scheduler.ReduceLROnPlateau`
        self.LR_FACTOR = 0.1  # decrease LR by factor of {}
        self.LR_PATIENCE = 10  # wait {} epochs before decreasing
        self.LR_MIN = 1e-6  # minimum learning rate
        # # Hyperparameters for `lr_scheduler.StepLR`:
        # self.SCHEDULER_STEP = 90  # every {} epochs...
        # self.SCHEDULER_GAMMA = 0.1  # ...multiply LR by {}
        #################################
        ## Create unique output directory
        #################################
        self.OUTPUT_DIR = "output/{}_{}_e{}_hs{}_nl{}_d{}_lr{}_m{}_wd{}/".format(
            self.MODEL,
            self.OPTIMIZER,
            self.N_EPOCHS,
            self.HIDDEN_SIZE,
            self.NUM_LAYERS,
            self.DROPOUT,
            self.INIT_LEARNING_RATE,
            self.MOMENTUM,
            self.WEIGHT_DECAY,
        )
        self.create_output_directories()
        #############################
        ## Save current configuration
        #############################
        self.save()
        ################
        ## PyTorch setup
        ################
        self.DEVICE = self.get_device()
        self.repeatable()

    def save(self):
        """
        Save current configuration to JSON file in `output/`
        """
        write_config_json(self)

    def repeatable(self):
        """
        Ensure deterministic/repeatable behavior across multiple runs
        """
        # cuDNN uses nondeterministic algorithms which are disabled here
        torch.backends.cudnn.enabled = False
        # Set random seed to ensure repeatable experiments
        # for anything using random number generation
        torch.manual_seed(1111)

    def get_device(self):
        """
        Check for CUDA / GPU support
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running with device: {}".format(device))
        return device

    def create_output_directories(self):
        make_directory(self.OUTPUT_DIR)
        make_directory(self.OUTPUT_DIR + SAVED_MODEL_DIR)
        make_directory(self.OUTPUT_DIR + PERFORMANCE_DIR)
        make_directory(self.OUTPUT_DIR + PREDICTIONS_DIR)
        make_directory(self.OUTPUT_DIR + EPOCHS_PREDICTIONS_DIR)
