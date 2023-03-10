import math
import torch
from datetime import datetime
from numpy import array as np_array

# Local scripts
from AnimalPathsDataset import ALL_CATEGORIES


class color:
    # Color codes for terminal output
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def categories_from_label(labels_tensor):
    """
    Convert the integer labels_tensor
    to a list of the respective category strings
    (where labels_tensor.shape = [batch_size, seq_length])
    """
    # Map label integers to label strings
    categories = [ALL_CATEGORIES[i] for i in labels_tensor.flatten().int()]
    return np_array(categories)


def categories_from_output(output_tensor):
    """
    Convert from "likelihood" output_tensor
    to a list of the predicted category strings
    (where output_tensor.shape = [batch_size, seq_length, num_Categories])
    """
    # Get max values and indices along the last (i.e. Categories) dimension
    max_values, max_indices = torch.max(output_tensor, dim=-1)
    # Map max indices to labels
    categories = [ALL_CATEGORIES[i] for i in max_indices.flatten()]
    return np_array(categories)


def reformat_features(features, individuals):
    """
    Reformat features input Tensor
    for easier interpretation in output CSV
    """
    # Flatten features tensor into normal Python list
    features = list(features.numpy().flatten())
    # Get the animal ID back from numerical representation
    animal = int(features[0])
    features[0] = individuals[animal]
    return features


def make_csv_output_rows(is_correct, guess, label, identifier, features_tensor):
    """
    Build the final output row for the predictions CSV,
    including:
    - the prediction's correctness (boolean)
    - the prediction itself (string)
    - the original label (string)
    - the "individual-local-identifier" / animal ID (string)
    - the input features vector values
    """
    # Initialize list of output rows
    rows = []
    # Combine first three lists into list of tuples
    data = list(zip(is_correct, guess, label))
    # Since features_tensor has a single batch dimension,
    # we use squeeze() to remove it and get a 2D tensor
    # with shape (seq_length, 6)
    features = features_tensor.squeeze(0)
    # Iterate over data and features, building and saving each row
    for i, (d, f) in enumerate(zip(data, features)):
        # Combine the data into a single list for the CSV output row
        row = list(d) + list(identifier) + f.tolist()
        rows.append(row)
    # Return the list of output rows
    return rows


def current_time():
    """
    Return the current Unix time
    """
    return datetime.now().timestamp()


def human_time(ts):
    """
    Convert from Unix time
    to a human-readable datetime string
    """
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def time_since(since):
    """
    Return the elapsed time
    from the provided timestamp until now
    """
    s = current_time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def start_script():
    """
    Print the time the script starts
    and keep track of it to compute runtime when it finishes
    """
    script_start = current_time()
    print("Start: {}".format(human_time(script_start)))
    return script_start


def finish_script(script_start):
    """
    Print the time the script finishes
    and the total script runtime
    """
    end = current_time()
    print("\nEnd: {}".format(human_time(end)))
    runtime = round(end - script_start, 3)
    print("Runtime: {} seconds\n".format(runtime))
