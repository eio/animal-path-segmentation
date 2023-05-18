import numpy as np
from torch import (
    no_grad,
    tensor,
    argmax,
    long as torch_long,
)
import os, sys

# Get the absolute path to the directory containing the current script
# and append to sys.path the subdirectory containing the local module to import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
# Local scripts
from utils.consts import N_CATEGORIES
from utils.save_and_load import (
    write_output_csv,
    write_performance_eval,
)
from utils.misc import (
    color,
    time_since,
    categories_from_label,
    categories_from_output,
    inverse_normalize_features,
    make_csv_output_rows,
)

# Specify the epochs interval
# to save an output CSV of predictions
SAVE_PREDICTIONS_EVERY = 20  # epochs


def test(model, criterion, labels_tensor, inputs_tensor):
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
    # ##################
    # ## HPC DIFF START
    # ##################
    # # Loss function expects labels_tensor input to be torch.Long dtype
    # labels_tensor = labels_tensor.to(torch_long)
    # # Convert the one-hot encoded target tensor to class labels
    # labels = argmax(labels_tensor, dim=2).squeeze()
    # # Compute loss
    # loss = criterion(
    #     output_tensor.view(-1, N_CATEGORIES),
    #     labels.view(-1),
    # )
    # #################
    # ## HPC DIFF END
    # #################
    # Return the prediction and loss
    return output_tensor, loss.item()


def test_process(
    model,
    criterion,
    test_loader,
    script_start,
    device,
    log_interval,
    SAVE_PREDICTIONS_EVERY=1,
    epoch=None,
):
    # Determine if this is the final test
    # or just one of many validation epochs
    final_test = False
    if epoch == None:
        final_test = True
        epoch = 1
    # Check if an output CSV should be produced this epoch
    WRITE_OUTPUT_CSV = False
    if (epoch % SAVE_PREDICTIONS_EVERY == 0) or (final_test == True):
        WRITE_OUTPUT_CSV = True
    print("\nStart Testing for Epoch {}...".format(epoch))
    # Initialize losses
    test_losses = []
    # Initialize array to store prediction alongside input features
    csv_out_rows = []
    # Initialize correct prediction count
    total_correct = 0
    # Initialize lists of predictions and ground truth
    # to generate evaluation metrics like the confusion matrix
    all_guesses = []
    all_labels = []
    # Since we're not training,
    # we don't need to calculate the gradients for our outputs
    with no_grad():
        # Iterate through the test data
        for i, batch in enumerate(test_loader, 0):
            # Send Tensors to GPU device (if CUDA-compatible)
            inputs_tensor = batch["features"].to(device)
            labels_tensor = batch["labels"].to(device)
            # Test the model by making a prediction and computing the loss
            output, loss = test(model, criterion, labels_tensor, inputs_tensor)
            # Store the loss value for this batch
            test_losses.append(loss)
            # Put output data back on the CPU
            output = output.cpu()
            # Get the predicted category string from the model output
            guesses = categories_from_output(output)
            # Convert the label tensor to the category string
            labels = categories_from_label(labels_tensor)
            # Check if the predictions array matches the labels array
            is_correct = guesses == labels
            # Count the number of correct predictions
            correct = np.count_nonzero(is_correct)
            # Keep a running tally of correct guesses
            total_correct += correct
            # Generate CSV output rows if needed
            if WRITE_OUTPUT_CSV:
                # Get the non-normalized features for output in the CSV
                orig_features = inverse_normalize_features(inputs_tensor)
                # Build the CSV output rows with predictions and input features
                rows = make_csv_output_rows(
                    is_correct, guesses, labels, batch["id"], orig_features
                )
                # Store the CSV output row for writing later
                csv_out_rows += rows
            # If it's the final model evaluation, then
            # keep track of guesses and labels for performance metrics
            if final_test:
                all_guesses.append(guesses)
                all_labels.append(labels)
            # Print details about this testing step
            if i % log_interval == 0:
                print(
                    "Trajectory: %d , Correct: %d/%d , Accuracy: %.2f%% , Loss: %.4f , Progress: %d%% (%s)"
                    % (
                        (i + 1),
                        correct,
                        len(is_correct),
                        correct / len(is_correct) * 100,
                        loss,
                        (i + 1) / len(test_loader) * 100,
                        time_since(script_start),
                    )
                )
    print("Finished Testing for Epoch {}.".format(epoch))
    print("Test:")
    print("\tAvg. Loss: {}".format(np.mean(test_losses)))
    percent_correct = total_correct / test_loader.dataset.total_records() * 100
    test_accuracy = round(percent_correct, 2)
    print("\t{}Test Accuracy: {}%{}".format(color.BOLD, test_accuracy, color.END))
    if WRITE_OUTPUT_CSV:
        # Determine the output predictions CSV filename
        if final_test == True:
            outname = "final_results.csv"
        else:
            outname = "epochs/epoch_{}.csv".format(epoch)
        # Write the predicted path segmentation labels to an output CSV
        write_output_csv(outname, csv_out_rows)
    if final_test:
        # Generate confusion matrix and other performance metrics
        write_performance_eval(
            np.concatenate(all_labels),
            np.concatenate(all_guesses),
        )
    # Return the test losses and accuracy from this epoch
    return test_losses, test_accuracy
