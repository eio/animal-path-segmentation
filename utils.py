import math
from datetime import datetime
import matplotlib.pyplot as plt


class color:
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


def time_since(since):
    now = datetime.now().timestamp()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def current_time():
    return datetime.now().timestamp()


def human_time(ts):
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def category_from_output(output, all_categories):
    # https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def plot_loss(output, completed_epochs, avg_train_losses, avg_test_losses):
    fig = plt.figure()
    plt.scatter(completed_epochs, avg_train_losses, color="blue")
    plt.scatter(completed_epochs, avg_test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Entropy (CE) Loss")
    plt.savefig(output)
    print("Performance evaluation saved to: `{}`".format(output))
