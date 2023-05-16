from numpy import count_nonzero as count_true
from torch import tensor, long as torch_long

# Local scripts
from save_and_load import save_model
from utils.misc import (
    color,
    time_since,
    categories_from_label,
    categories_from_output,
)
from utils.consts import N_CATEGORIES


def train(model, optimizer, criterion, labels_tensor, inputs_tensor):
    ## When using PyTorch's built-in RNN or LSTM modules,
    ## we don't need to define the initHidden() function explicitly.
    ## The hidden state is automatically initialized by the PyTorch module.
    # hidden = model.initHidden()
    ##############################
    ##############################
    # Clear the gradients of all optimized tensors
    # to prepare for a new backpropagation pass
    optimizer.zero_grad()
    # Get the sequence length of this input
    # (i.e. number of waypoints in the trajectory)
    # where inputs_tensor.shape = [batch_size, seq_length, num_Features]
    seq_length = tensor([inputs_tensor.shape[1]])
    # Forward pass
    output_tensor = model(inputs_tensor, seq_length)
    # Compute loss
    # output_tensor.shape = [batch_size, seq_length, num_Categories]
    # labels_tensor.shape = [batch_size, seq_length]
    # NOTE: output_tensor.view(-1, X) reshapes the tensor to have X columns
    # which should match the OUTPUT_SIZE defined in `run_model.py`
    loss = criterion(
        output_tensor.view(-1, N_CATEGORIES),
        labels_tensor.view(-1, N_CATEGORIES),
    )
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    # Return the prediction and loss
    return output_tensor, loss.item()


def train_process(
    optimizer, model, criterion, train_loader, script_start, device, log_interval, epoch
):
    print("\nStart Training for Epoch #{}...".format(epoch))
    print("\nCurrent learning rate: {}".format(optimizer.param_groups[0]["lr"]))
    # Initialize losses
    train_losses = []
    # Initialize correct prediction count
    total_correct = 0
    # Iterate through the training data
    for i, batch in enumerate(train_loader, 0):
        # Send Tensors to GPU device (if CUDA-compatible)
        inputs_tensor = batch["features"].to(device)
        labels_tensor = batch["labels"].to(device)
        # Make prediction, compute the loss, and update model with optimizer
        output, loss = train(model, optimizer, criterion, labels_tensor, inputs_tensor)
        # Store the loss value for this batch
        train_losses.append(loss)
        # Get the predicted category string from the model output
        guesses = categories_from_output(output)
        # Convert the labels tensor to the category strings
        labels = categories_from_label(labels_tensor)
        # Check if the predictions array matches the labels array
        is_correct = guesses == labels
        # Count the number of correct predictions
        correct = count_true(is_correct)
        # Keep a running tally of correct guesses
        total_correct += correct
        # Print details about this training step
        if i % log_interval == 0:
            print(
                "Trajectory: %d , Correct: %d/%d , Accuracy: %.2f%% , Loss: %.4f , Progress: %d%% (%s)"
                % (
                    (i + 1),
                    correct,
                    len(is_correct),
                    correct / len(is_correct) * 100,
                    loss,
                    (i + 1) / len(train_loader) * 100,
                    time_since(script_start),
                )
            )
    print("Finished Training for Epoch #{}.".format(epoch))
    percent_correct = total_correct / train_loader.dataset.total_records() * 100
    train_accuracy = round(percent_correct, 2)
    print("\t{}Train Accuracy: {}%{}".format(color.BOLD, train_accuracy, color.END))
    #####################
    ## Save the Model  ##
    #####################
    save_model(epoch, model, optimizer)
    # Return the train losses and accuracy from this epoch
    return train_losses, train_accuracy
