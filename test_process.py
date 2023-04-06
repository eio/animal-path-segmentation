from numpy import mean, count_nonzero as count_true
from torch import no_grad, tensor, long as torch_long

# Local scripts
from AnimalPathsDataset import OUTPUT_FIELDNAMES
from save_and_load import write_output_csv
from utils import (
    color,
    time_since,
    categories_from_label,
    categories_from_output,
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
    # Loss function expects labels_tensor input to be torch.Long dtype
    labels_tensor = labels_tensor.to(torch_long)
    # Compute loss
    # output_tensor.shape = [batch_size, seq_length, num_Categories]
    # labels_tensor.shape = [batch_size, seq_length]
    # NOTE: output_tensor.view(-1, X) reshapes the tensor to have X columns
    # which should match the OUTPUT_SIZE defined in `run_model.py`
    loss = criterion(output_tensor.view(-1, 4), labels_tensor.view(-1))
    # Return the prediction and loss
    return output_tensor, loss.item()


def test_process(
    model,
    criterion,
    test_loader,
    script_start,
    device,
    log_interval,
    epoch=None,
):
    # Determine if this is the final test
    # or just one of many validation epochs
    final_test = False
    if epoch == None:
        final_test = True
        epoch = 1
    print("\nStart Testing for Epoch {}...".format(epoch))
    # Initialize losses
    test_losses = []
    # Initialize array to store prediction alongside input features
    csv_out_rows = []
    # Initialize correct prediction count
    total_correct = 0
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
            # Put data back on the CPU
            output = output.cpu()
            features = inputs_tensor.cpu()
            # Get the predicted category string from the model output
            guesses = categories_from_output(output)
            # Convert the label tensor to the category string
            labels = categories_from_label(labels_tensor)
            # Check if the predictions array matches the labels array
            is_correct = guesses == labels
            # Count the number of correct predictions
            correct = count_true(is_correct)
            # Keep a running tally of correct guesses
            total_correct += correct
            # Generate the CSV output rows
            rows = make_csv_output_rows(
                is_correct, guesses, labels, batch["id"], features
            )
            # Store the CSV output row for writing later
            csv_out_rows += rows
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
    print("\tAvg. Loss: {}".format(mean(test_losses)))
    percent_correct = total_correct / test_loader.dataset.total_records() * 100
    test_accuracy = round(percent_correct, 2)
    print("\t{}Test Accuracy: {}%{}".format(color.BOLD, test_accuracy, color.END))
    # Determine the output predictions CSV filename
    if final_test == True:
        outname = "final_test.csv"
    else:
        outname = "epochs/epoch_{}.csv".format(epoch)
    # Check if an output CSV should be produced this epoch
    if (epoch % SAVE_PREDICTIONS_EVERY == 0) or (final_test == True):
        # Write the predicted path segmentation labels to an output CSV
        write_output_csv(outname, csv_out_rows, OUTPUT_FIELDNAMES)
    # Return the test losses and accuracy from this epoch
    return test_losses, test_accuracy
