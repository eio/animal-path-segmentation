from torch import long as torch_long

# Local scripts
from save_and_load import save_model
from utils import color, time_since, category_from_label, category_from_output


def train(model, optimizer, criterion, label_tensor, inputs_tensor):
    ## When using PyTorch's built-in RNN or LSTM modules,
    ## we don't need to define the initHidden() function explicitly.
    ## The hidden state is automatically initialized by the PyTorch module.
    # hidden = model.initHidden()
    ##############################
    ##############################
    # Clear the gradients of all optimized tensors
    # to prepare for a new backpropagation pass
    optimizer.zero_grad()
    # Forward pass
    output = model(inputs_tensor)
    # Loss function expects label_tensor input to be torch.Long dtype
    label_tensor = label_tensor.to(torch_long)
    # Compute loss
    loss = criterion(output, label_tensor)
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    # Return the prediction and loss
    return output, loss.item()


def train_process(
    optimizer, model, criterion, train_loader, script_start, device, log_interval, epoch
):
    print("\nStart Training for Epoch #{}...".format(epoch))
    # Initialize losses
    train_losses = []
    # Initialize correct prediction count
    total_correct = 0
    for i, batch in enumerate(train_loader, 0):
        # Send Tensors to GPU device (if CUDA-compatible)
        inputs_tensor = batch["features"].to(device)
        label_tensor = batch["label"].to(device)
        # Make prediction, compute the loss, and update model with optimizer
        output, loss = train(model, optimizer, criterion, label_tensor, inputs_tensor)
        # Store the loss value for this batch
        train_losses.append(loss)
        # Print iter number, loss, name, and guess
        if i % log_interval == 0:
            # Get the predicted category string from the model output
            guess = category_from_output(output)
            # Convert the label tensor to the category string
            label = category_from_label(label_tensor)
            # Check if the prediction matches the label
            is_correct = guess == label
            # Keep track of how many guesses are correct
            if is_correct:
                total_correct += 1
            # Generate print string to indicate success
            correct = "✓" if is_correct else "✗ (%s)" % label
            print(
                "Train Epoch:%d %d%% (%s) Loss:%.4f %s / %s %s"
                % (
                    epoch,
                    i / len(train_loader) * 100,
                    time_since(script_start),
                    loss,
                    "inputs_tensor",  # inputs_tensor.numpy()
                    guess,
                    correct,
                )
            )
    print("Finished Training for Epoch #{}.".format(epoch))
    # percent_correct = total_correct / len(train_loader) * 100
    # percent_correct = round(percent_correct, 2)
    # print("{}Train Accuracy: {}%{}".format(color.BOLD, percent_correct, color.END))
    #####################
    ## Save the Model  ##
    #####################
    save_model(epoch, model, optimizer)
    # Return the train losses from this epoch
    return train_losses
