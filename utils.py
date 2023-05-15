import math
from datetime import datetime
from numpy import array as np_array
from torch import tensor, eye, max as torch_max

# Local scripts
from consts import SEASON_LABELS, N_CATEGORIES

# Reverse SEASON_LABELS dictionary so that onehot tuples are keys
ONEHOT_LABELS = {tuple(v): k for k, v in SEASON_LABELS.items()}
# Create identity matrix
ONEHOT_MATRIX = eye(N_CATEGORIES)


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


def onehot_to_string(onehot):
    """
    Convert the one-hot encoded season tensor
    to the corresponding season string label
    """
    key = tuple(onehot.tolist())
    return ONEHOT_LABELS[key]


def categories_from_label(labels_tensor):
    """
    Convert the labels_tensor to a list of the
    respective seasonal category strings
    (where labels_tensor.shape = [batch_size, seq_length])
    """
    # Map label tensors to label strings
    categories = []
    for sequence in labels_tensor:
        for label in sequence:
            # Convert the one-hot tensor to a string label
            categories.append(onehot_to_string(label))
    return np_array(categories)


def categories_from_output(output_tensor):
    """
    Convert from probability / "likelihood" output_tensor
    to a list of the predicted category strings
    (where output_tensor.shape = [batch_size, seq_length, num_Categories])
    """
    # Get max values and indices along the last (i.e. Categories) dimension
    max_values, max_indices = torch_max(output_tensor, dim=-1)
    # Map max indices to labels
    categories = []
    for sequence in max_indices:
        for index in sequence:
            # Get the one-hot tensor from the most likely index value
            label = ONEHOT_MATRIX[index]
            # Convert the one-hot tensor to a string label
            categories.append(onehot_to_string(label))
    return np_array(categories)


def make_csv_output_rows(is_correct, guess, label, identifier, features):
    """
    Build the final output row for the predictions CSV,
    including:
    - the prediction's correctness (boolean)
    - the prediction itself (string)
    - the original label (string)
    - the "individual-local-identifier" / animal ID (string)
    - the input feature values
    """
    # Initialize list of output rows
    rows = []
    # Combine first three lists into list of tuples
    data = list(zip(is_correct, guess, label))
    # Iterate over data and features, building and saving each row
    for i, (d, f) in enumerate(zip(data, features)):
        # Combine the data into a single list for the CSV output row
        row = list(d) + identifier.tolist() + f
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
