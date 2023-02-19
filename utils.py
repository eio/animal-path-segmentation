import math
from datetime import datetime


def categoryFromOutput(output, all_categories):
    # https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def timeSince(since):
    now = datetime.now().timestamp()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def currentTime():
    return datetime.now().timestamp()


def humanTime(ts):
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
