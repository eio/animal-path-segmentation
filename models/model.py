# Machine Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Local scripts
from utils.consts import RNN, LSTM, GRU, SGD, ADAM, N_CATEGORIES


class Model(nn.Module):
    """
    Create the model
    """

    def __init__(self, model_name, input_size, hidden_size, num_layers, dropout):
        """
        Initialize the model.
        Parameters:
            - model_name (str): Specify either 'rnn' or 'lstm'.
            - input_size (int): Size of the input features.
            - hidden_size (int): Number of features in the hidden state.
            - num_layers (int): Number of recurrent layers.
            - dropout (float): Dropout probability for the RNN or LSTM layers.
        """
        super(Model, self).__init__()
        # `output_size` is the number of possible class categories:
        # len(["Winter", "Spring", "Summer", "Autumn"])
        output_size = N_CATEGORIES
        # Linear layer to scale the output
        self.fc = nn.Linear(hidden_size, output_size)
        # Create the model
        if model_name == RNN:
            self.model = nn.RNN(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif model_name == LSTM:
            self.model = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif model_name == GRU:
            self.model = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        else:
            raise ValueError(
                "Invalid model architecture. Please choose 'RNN' or 'LSTM' or 'GRU'."
            )

    def forward(self, x, lengths):
        # Pack the sequences into a PackedSequence object
        packed_inputs = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        # Pass the packed sequences through the RNN
        packed_outputs, _ = self.model(packed_inputs)
        # Unpack the sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # Pass the outputs through a linear layer to get the final predictions
        logits = self.fc(outputs)
        return logits


def Criterion():
    """
    Create the loss function
    """
    # The log softmax function and the negative log likelihood loss
    # are combined in nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss()


def Optimizer(model, cfg):
    """
    Create the optimizer
    """
    if cfg.OPTIMIZER == SGD:
        return optim.SGD(
            model.parameters(),
            lr=cfg.INIT_LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER == ADAM:
        return optim.Adam(
            model.parameters(),
            lr=cfg.INIT_LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            # betas=(0.9, 0.999),
            # eps=1e-08,
        )
    else:
        raise Exception("Please provide a valid optimizer name.")


def Scheduler(optimizer, cfg):
    """
    Create the learning-rate scheduler
    """
    # return optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=cfg.SCHEDULER_STEP,
    #     gamma=cfg.SCHEDULER_GAMMA,
    # )
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # reduce LR when the validation loss stops decreasing
        factor=cfg.LR_FACTOR,  # reduce LR by factor of {} when loss stops decreasing
        patience=cfg.LR_PATIENCE,  # wait for {} epochs before reducing the LR
        min_lr=cfg.LR_MIN,  # don't let the learning rate go below {}
        verbose=True,  # print a message when the learning rate is reduced
    )
