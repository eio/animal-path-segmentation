import math
from datetime import datetime


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


def category_from_output(output, all_categories):
    """
    Convert from a 1x3 Tensor "likelihood"
    to the predicted category value and index
    """
    # https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
