import torch

# Local scripts
from AnimalPathsDataset import N_FEATURES, N_CATEGORIES
from save_and_load import write_config_json


class Configurator(object):
    """
    Initialize PyTorch settings and hyperparameters
    """

    def __init__(self):
        #################
        ## Print settings
        #################
        # Print every {} epochs
        self.LOG_INTERVAL = 1
        ########################
        # Input feature settings
        ########################
        # Each position record is assigned to a "burst" of movement,
        # where each burst lasts for the specified time duration.
        # Model performance may change with different burst definitions.
        self.BURST_TIME_THRESHOLD = 300  # seconds
        ####################
        ## Training settings
        ####################
        self.N_EPOCHS = 100
        self.BATCH_SIZE = 1
        #################
        ## Model settings
        #################
        # `input_size` is the number of features/ covariates
        self.INPUT_SIZE = N_FEATURES
        # `hidden_size` controls the width of each RNN layer
        # (i.e., the number of neurons in each layer)
        self.HIDDEN_SIZE = 32
        # `output_size` is the number of possible categories:
        # len(["Winter", "Spring", "Summer", "Autumn"])
        self.OUTPUT_SIZE = N_CATEGORIES
        # `num_layers` controls the depth of the RNN
        # (i.e., the number of stacked RNN layers)
        self.NUM_LAYERS = 2  # default = 1
        # The recommended values for dropout probability are
        # between 0.1 and 0.5, depending on the task and model size
        self.DROPOUT = 0.2  # default = 0
        # Guideline: if the model has less than
        # tens-of-thousands of trainable parameters,
        # regularization may not be needed. For an RNN:
        # trainable_params = ((input_size + hidden_size) * hidden_size + hidden_size) * num_layers
        #####################
        ## Optimizer settings
        #####################
        self.INIT_LEARNING_RATE = 0.001
        #####################
        ## Scheduler settings
        #####################
        # Hyperparameters for `lr_scheduler.ReduceLROnPlateau`
        self.LR_FACTOR = 0.1  # decrease LR by factor of {}
        self.LR_PATIENCE = 10  # wait {} epochs before decreasing
        self.LR_MIN = 1e-6  # minimum learning rate
        # Hyperparameters for `lr_scheduler.StepLR`
        # SCHEDULER_STEP = 90  # every {} epochs...
        # SCHEDULER_GAMMA = 0.1  # ...multiply LR by {}
        #####################
        ## Save configuration
        #####################
        # Save the current configuration to file
        # alongside the final model results
        self.save()
        ################
        ## PyTorch setup
        ################
        self.DEVICE = self.get_device()
        self.repeatable()

    def save(self):
        """
        Save current configuration to an output file
        """
        write_config_json(self)

    def repeatable(self):
        """
        Ensure deterministic/repeatable behavior across multiple runs
        """
        # cuDNN uses nondeterministic algorithms which are disabled here
        torch.backends.cudnn.enabled = False
        # To ensure repeatable experiments, we set a random seed
        # for anything using random number generation
        random_seed = 1111
        torch.manual_seed(random_seed)

    def get_device(self):
        """
        Check for CUDA / GPU support
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running with device: {}".format(device))
        return device
