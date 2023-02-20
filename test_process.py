from torch import no_grad, long as torch_long
from numpy import mean

# Local scripts
from AnimalPathsDataset import OUTPUT_FIELDNAMES
from save_and_load import write_output_csv
from utils import (
    color,
    time_since,
    category_from_label,
    category_from_output,
    reformat_features,
)


def test(model, criterion, label_tensor, inputs_tensor):
    # Forward pass
    output = model(inputs_tensor)
    # Loss function expects label_tensor input to be torch.Long dtype
    label_tensor = label_tensor.to(torch_long)
    # Compute loss
    loss = criterion(output, label_tensor)
    # Return the prediction and loss
    return output, loss.item()


def make_csv_output_row(is_correct, guess, label, features, individuals):
    # Prepare input features for the output CSV row
    features = reformat_features(features, individuals)
    # Store prediction with input features in the output CSV row
    row = [is_correct, guess, label] + features
    return row


def test_process(
    model,
    criterion,
    test_loader,
    script_start,
    device,
    log_interval,
    epoch=1,
):
    print("\nStart Testing for Epoch {}...".format(epoch))
    # Initialize losses
    test_losses = []
    # Initialize correct prediction count
    total_correct = 0
    # Initialize array to store prediction alongside input features
    csv_out_rows = []
    # Since we're not training,
    # we don't need to calculate the gradients for our outputs
    with no_grad():
        # for data in test_loader:
        for i, batch in enumerate(test_loader, 0):
            # Send Tensors to GPU device (if CUDA-compatible)
            inputs_tensor = batch["features"].to(device)
            label_tensor = batch["label"].to(device)
            # Test the model by making a prediction and computing the loss
            output, loss = test(model, criterion, label_tensor, inputs_tensor)
            # Store the loss value for this batch
            test_losses.append(loss)
            # Put data back on the CPU
            output = output.cpu()
            features = inputs_tensor.cpu()
            # Get the predicted category string from the model output
            guess = category_from_output(output)
            # Convert the label tensor to the category string
            label = category_from_label(label_tensor)
            # Is the prediction correct?
            is_correct = guess == label
            # Keep track of how many guesses are correct
            if is_correct:
                total_correct += 1
            # Generate the CSV output row
            row = make_csv_output_row(
                is_correct,
                guess,
                label,
                features,
                test_loader.dataset.get_individuals(),
            )
            # Store the CSV output row for writing later
            csv_out_rows.append(row)
            # Print epoch, loss, and guess
            if i % log_interval == 0:
                # Generate print string to indicate success
                correct = "✓" if is_correct else "✗ (%s)" % label
                print(
                    "Test Epoch:%d %d%% (%s) Loss:%.4f %s / %s %s"
                    % (
                        epoch,
                        i / len(test_loader) * 100,
                        time_since(script_start),
                        loss,
                        "inputs_tensor",  # inputs_tensor.numpy()
                        guess,
                        correct,
                    )
                )
    print("Finished Testing for Epoch {}.".format(epoch))
    print("Test:")
    print("\tAvg. Loss: {}".format(mean(test_losses)))
    percent_correct = total_correct / len(test_loader) * 100
    percent_correct = round(percent_correct, 2)
    print("\t{}Test Accuracy: {}%{}".format(color.BOLD, percent_correct, color.END))
    # Write the predicted poses to an output CSV
    # in the submission format expected
    write_output_csv(epoch, csv_out_rows, OUTPUT_FIELDNAMES)
    # Return the test losses from this epoch
    return test_losses
