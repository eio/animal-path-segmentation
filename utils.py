import math
from datetime import datetime

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


def category_from_label(label_tensor):
    """
    Convert the 1x1 label input Tensor
    to the respective category string
    """
    label = int(label_tensor.item())
    category = ALL_CATEGORIES[label]
    return category


def category_from_output(output):
    """
    Convert from a 1x3 Tensor "likelihood"
    to the predicted category value and index
    # https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return ALL_CATEGORIES[category_i]


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
