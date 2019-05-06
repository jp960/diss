import argparse
import re

import numpy as np
from matplotlib import pyplot as plt
from six.moves import xrange
from glob import glob


def extract_test_data():
    files = sorted(glob('/home/janhavi/Documents/Final Year/DISS/data/results/test/*.png'))
    files = [re.sub("/home/janhavi/Documents/Final Year/DISS/data/results/test/", "", file) for file in files]
    files = [re.sub(".png", "", file) for file in files]
    batches = list()
    losses = list()

    for file in files:
        split = file.split("_")
        batches.append(float(split[1]))
        losses.append(float(split[3]))

    return list(zip(batches, losses))


def extract_data(file_path):
    loc_epochs = list()
    loc_losses = list()
    loc_sample_epochs = list()
    loc_sample_losses = list()
    batches = list()
    f = open(file_path, "r")
    lines = f.readlines()
    first_line = lines[2].strip()
    batch_num = int(first_line.split(":")[1].split("[")[2].split("/")[1].split("]")[0])
    for line in lines[2:]:
        line = line.strip()
        bits = line.split(":")
        if bits[0] == "Epoch":
            loc_epochs.append(float(bits[1].split("[")[1].split("]")[0]))
            batches.append(float(bits[3]))
        elif bits[0] == "[Sample] g_loss":
            loc_sample_losses.append(float(bits[1]))
        else:
            print(bits)
    f.close()

    loc_epochs = list(dict.fromkeys(loc_epochs))
    [loc_losses.append(np.mean(batches[i:i + batch_num])) for i in xrange(0, len(batches), batch_num)]
    [loc_sample_epochs.append(float(i + 1)) for i in range(len(loc_sample_losses))]

    return loc_epochs, loc_losses, loc_sample_epochs, loc_sample_losses


def get_graph_title(file_path):
    split = file_path.split("/")
    file_name = split[len(split) - 1].split(".")[0]
    file_name = re.sub("output", "", file_name)
    file_name = re.sub(".txt", "", file_name)
    info = file_name.split("_")
    if info[0] == "SUNRGBD":
        train_size = "4934"
    elif info[0] == "NYU":
        train_size = "1449"
    else:
        train_size = info[0]
    batch_size = info[1]
    num_epochs = info[2]
    lr = info[3]
    return "{0} samples, Batch size: {1}, No. Epochs: {2}, Learning Rate: 0.{3}".format(train_size, batch_size,
                                                                                               num_epochs, lr)


def plot_graph(x, y, xlabel, ylabel, title):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.axis([0, max(x), 0, max(y)])
    plt.show()


def plot_test_graph(pairs, xlabel, ylabel, title):
    sorted_pairs = sorted(pairs, key=lambda tup: int(tup[0]))
    x = list()
    y = list()
    [x.append(tup[0]) for tup in sorted_pairs]
    [y.append(tup[1]) for tup in sorted_pairs]

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.axis([0, max(x), 0, max(y)])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', dest='file_path')
    args = parser.parse_args()
    graph_title = get_graph_title(args.file_path)
    epochs, losses, sample_epochs, sample_losses = extract_data(args.file_path)
    plot_graph(epochs, losses, "Epoch", "Loss", graph_title)
    plot_graph(sample_epochs, sample_losses, "Sample", "Loss", "Sample losses for: \n {0}".format(graph_title))
    pairs = extract_test_data()
    plot_test_graph(pairs, "Batch", "Loss", "Testing results of mean loss per batch")
    epochs, losses = extract_test_data()
    plot_graph(epochs, losses, "Batch", "Loss", "Testing results of mean loss per batch")

