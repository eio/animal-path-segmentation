import torch

# Local scripts
from AnimalPathsDataset import N_FEATURES, N_CATEGORIES


class Configurator(object):
    """
    Initialize PyTorch settings and hyperparameters
    """

    def __init__(self):
        #################
        ## Print settings
        #################
        self.LOG_INTERVAL = 1  # print every {} epochs
        ####################
        ## Training settings
        ####################
        self.N_EPOCHS = 100
        self.BATCH_SIZE = 1
        #################
        ## Model settings
        #################
        self.INPUT_SIZE = N_FEATURES  # number of features / covariates
        self.HIDDEN_SIZE = 32  # tunable hyperparameter
        self.OUTPUT_SIZE = N_CATEGORIES  # "Winter", "Spring", "Summer", "Autumn"
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
        ###################
        ## PyTorch settings
        ###################
        self.DEVICE = self.get_device()
        self.repeatable()

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
